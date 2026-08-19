"""
eval/query_test.py

只负责提问，可以反复运行调试。不会重新上传文档。
前提：已经运行过一次 eval/ingest_once.py。
"""
import json

import requests

BASE_URL = "http://localhost:8000"

TEST_QUESTIONS = [
    {"pdf_filename": "01_attention_is_all_you_need.pdf",
     "question": "What type of neural network architecture does this paper introduce?"},
    {"pdf_filename": "02_language_models_are_few_shot_learners_gpt3.pdf",
     "question": "How many parameters does GPT-3 have?"},
    {"pdf_filename": "03_retrieval_augmented_generation_rag.pdf",
     "question": "What are the two main components combined in RAG?"},
    {"pdf_filename": "04_vision_transformer_vit.pdf",
     "question": "How does the Vision Transformer process an image?"},
    {"pdf_filename": "05_denoising_diffusion_probabilistic_models.pdf",
     "question": "What is the core idea behind diffusion models?"},
    {"pdf_filename": "06_lora_low_rank_adaptation.pdf",
     "question": "What problem does LoRA aim to solve when fine-tuning large models?"},
    {"pdf_filename": "07_instructgpt_rlhf.pdf",
     "question": "What technique is used to align the model with human preferences?"},
    {"pdf_filename": "08_chain_of_thought_prompting.pdf",
     "question": "What does chain-of-thought prompting encourage the model to do?"},
    {"pdf_filename": "09_llama_open_and_efficient_foundation_language_models.pdf",
     "question": "Who released the LLaMA models?"},
    {"pdf_filename": "10_mamba_linear_time_sequence_modeling.pdf",
     "question": "What architecture does Mamba use instead of attention?"},
]


def parse_sse_response(resp) -> dict:
    answer = ""
    sources = []
    error = None
    event_type = "message"

    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        if raw_line.startswith("event:"):
            event_type = raw_line[len("event:"):].strip()
        elif raw_line.startswith("data:"):
            try:
                data = json.loads(raw_line[len("data:"):].strip())
            except json.JSONDecodeError:
                continue
            if event_type == "message":
                answer += data.get("delta", "")
            elif event_type == "sources":
                sources = data.get("sources", [])
            elif event_type == "error":
                error = data.get("message")

    return {"answer": answer, "sources": sources, "error": error}

def debug_retrieval(question: str):
    """诊断用：直接调后端看实际检索到了哪些chunk内容,不经过LLM生成"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from app.services import embeddings, vector_store
    from app.config import settings

    query_vector = embeddings.embed_query(question)
    matches = vector_store.query_similar(query_vector, top_k=settings.top_k)

    print(f"\n{'='*70}")
    print(f"诊断问题: {question}")
    print(f"检索到 {len(matches)} 个chunk:")
    for i, m in enumerate(matches, 1):
        print(f"\n--- chunk {i} (score={m['score']:.3f}, source={m['source']}) ---")
        print(m['content'][:300])   

def main():
    for i, case in enumerate(TEST_QUESTIONS, start=1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] {case['question']}")

        resp = requests.post(
            f"{BASE_URL}/query",
            json={"question": case["question"]},
            timeout=60,
            stream=True,
        )
        if resp.status_code != 200:
            print(f"[ERROR {resp.status_code}] {resp.text}")
            continue

        result = parse_sse_response(resp)
        if result["error"]:
            print(f"[系统返回错误] {result['error']}")
            continue

        print(f"系统回答: {result['answer']}")
        if result["sources"]:
            top = result["sources"][0]
            mark = "✓" if case["pdf_filename"] in top["source"] else "⚠"
            print(f"引用来源: {top['source']} (相似度 {top['score']:.2f}) {mark}")


if __name__ == "__main__":
    main()
    debug_retrieval("What are the two main components combined in RAG?")        
    debug_retrieval("What technique is used to align the model with human preferences?")