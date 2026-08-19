"""
eval/ragas_eval.py

用 RAGAS 计算标准化的RAG评估指标:
- Faithfulness(忠实度): 回答是否完全基于检索到的内容,越高说明幻觉越少
- ResponseRelevancy(相关性): 回答是否切题
- LLMContextRecall / FactualCorrectness: 需要人工标注的标准答案(ground truth)才能算,
  用于衡量"检索是否召回了足够信息"和"答案在事实层面是否正确"

前置条件:
1. 已经用 eval/ingest_once.py 上传过文档(且数据是干净的,没有重复)
2. 后端在跑 (uvicorn app.main:app --reload)
3. pip install ragas langchain-openai
"""
import json
import sys
from pathlib import Path

import requests
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from langchain_openai import OpenAIEmbeddings as LangchainOpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness, answer_relevancy, context_recall, factual_correctness
from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings
from ragas.run_config import RunConfig

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings

BASE_URL = "http://localhost:8000"

# question + 人工核对过的标准答案(ground truth)。
# 第1题("this paper"指代不明)故意不给ground truth,只能参与不需要参考答案的指标。
TEST_CASES = [
    {"question": "What type of neural network architecture does this paper introduce?",
     "reference": None},
    {"question": "How many parameters does GPT-3 have?",
     "reference": "GPT-3 has 175 billion parameters."},
    {"question": "What are the two main components combined in RAG?",
     "reference": "RAG combines a retriever and a generator."},
    {"question": "How does the Vision Transformer process an image?",
     "reference": "It splits the image into fixed-size patches, linearly embeds them with position embeddings, and feeds the sequence into a standard Transformer encoder."},
    {"question": "What is the core idea behind diffusion models?",
     "reference": "Diffusion models learn to reverse a gradual noising process, generating data by denoising step by step from pure noise."},
    {"question": "What problem does LoRA aim to solve when fine-tuning large models?",
     "reference": "LoRA reduces the number of trainable parameters and GPU memory needed by injecting low-rank matrices instead of updating all model weights."},
    {"question": "What technique is used to align the model with human preferences?",
     "reference": "Reinforcement Learning from Human Feedback (RLHF)."},
    {"question": "What does chain-of-thought prompting encourage the model to do?",
     "reference": "It encourages the model to break multi-step problems into intermediate reasoning steps."},
    {"question": "Who released the LLaMA models?",
     "reference": "Meta AI."},
    {"question": "What architecture does Mamba use instead of attention?",
     "reference": "Mamba uses a selective state space model (SSM) architecture instead of attention."},
]

def attach_llm_and_embeddings(metrics, llm, embeddings):
    """
    显式地把llm和embeddings挂到每个metric对象上，
    不依赖 evaluate() 内部自动传递(这一步在当前ragas版本下不可靠)。
    """
    run_config = RunConfig()
    for metric in metrics:
        if isinstance(metric, MetricWithLLM):
            metric.llm = llm
        if isinstance(metric, MetricWithEmbeddings):
            metric.embeddings = embeddings
        metric.init(run_config)
    return metrics

def collect_answer_and_contexts(question: str) -> dict:
    """调 /query 接口(SSE)，拼出完整答案 + 拿到引用chunk内容作为retrieved_contexts"""
    resp = requests.post(
        f"{BASE_URL}/query",
        json={"question": question},
        timeout=60,
        stream=True,
    )
    resp.raise_for_status()

    answer = ""
    contexts = []
    event_type = "message"
    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        if raw_line.startswith("event:"):
            event_type = raw_line[len("event:"):].strip()
        elif raw_line.startswith("data:"):
            data = json.loads(raw_line[len("data:"):].strip())
            if event_type == "message":
                answer += data.get("delta", "")
            elif event_type == "sources":
                contexts = [s["content"] for s in data.get("sources", [])]

    return {"response": answer, "retrieved_contexts": contexts or [""]}


def build_dataset():
    """收集系统的真实回答，组装成RAGAS要求的格式"""
    records = []
    for case in TEST_CASES:
        print(f"收集: {case['question'][:50]}...")
        result = collect_answer_and_contexts(case["question"])
        record = {
            "user_input": case["question"],
            "retrieved_contexts": result["retrieved_contexts"],
            "response": result["response"],
        }
        if case["reference"]:
            record["reference"] = case["reference"]
        records.append(record)
    return records


def main():
    print("正在收集系统回答与检索上下文...\n")
    all_records = build_dataset()

    # 有ground truth的样本走完整指标；没有的(第1题)只能算不需要参考答案的Faithfulness
    with_reference = [r for r in all_records if "reference" in r]
    without_reference = [r for r in all_records if "reference" not in r]

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(model=settings.openai_chat_model, api_key=settings.openai_api_key)
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
    LangchainOpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )
)

    print(f"\n{'='*60}")
    print(f"评估 {len(with_reference)} 条有标准答案的样本 (完整指标)")
    print(f"{'='*60}")

    metrics_with_ref = attach_llm_and_embeddings(
        [faithfulness, answer_relevancy, context_recall, factual_correctness],
        evaluator_llm,
        evaluator_embeddings,
    )

    if with_reference:
        ds = EvaluationDataset.from_list(with_reference)
        result = evaluate(dataset=ds, metrics=metrics_with_ref)  # 不再需要传llm/embeddings，已经挂在metric上了
        print(result)
        result.to_pandas().to_csv("eval/ragas_results_with_reference.csv", index=False)
        print("已保存: eval/ragas_results_with_reference.csv")

    if without_reference:
        metrics_no_ref = attach_llm_and_embeddings([faithfulness], evaluator_llm, evaluator_embeddings)
        ds2 = EvaluationDataset.from_list(without_reference)
        for attempt in range(3):
            try:
                result2 = evaluate(dataset=ds2, metrics=metrics_no_ref)
                print(result2)
                break
            except Exception as e:
                print(f"第{attempt + 1}次尝试失败: {e}")


if __name__ == "__main__":
    main()
