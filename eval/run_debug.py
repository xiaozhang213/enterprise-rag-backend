
"""
eval/run_debug.py

直接使用 eval/sample_docs/pdfs/ 下已有的PDF（ml_arxiv_10_papers数据集），
批量上传到RAG系统，再逐个提问，打印结果人工核对。
不依赖 datasets 库，只用 requests，环境要求最简单。
"""
import json
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
PDF_DIR = SCRIPT_DIR / "sample_docs" / "pdfs"
BASE_URL = "http://localhost:8000"

# 每篇论文配1个可以人工核对答案的问题，文件名要和你实际拷贝进去的一致
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


def check_pdfs_exist():
    pdf_paths = list(PDF_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError(
            f"在 {PDF_DIR} 没找到任何PDF，请先把 ml_arxiv_10_papers 里的文件拷贝进这个目录"
        )
    print(f"检测到 {len(pdf_paths)} 份PDF:")
    for p in pdf_paths:
        print(f"  - {p.name} ({p.stat().st_size / 1024:.1f} KB)")
    return pdf_paths


def check_backend_alive():
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        resp.raise_for_status()
        print(f"后端服务正常: {resp.json()}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            f"连不上后端 {BASE_URL}，请先运行 uvicorn app.main:app --reload\n原始错误: {e}"
        )


def upload_all_pdfs(pdf_paths):
    print(f"\n批量上传 {len(pdf_paths)} 份PDF...")
    files = [("files", (p.name, open(p, "rb"), "application/pdf")) for p in pdf_paths]

    resp = requests.post(f"{BASE_URL}/ingest/batch", files=files, timeout=300)
    resp.raise_for_status()
    result = resp.json()

    print(f"上传结果: 成功 {result['succeeded']}/{result['total']}")
    for r in result["results"]:
        mark = "✓" if r["status"] == "success" else "✗"
        print(f"  {mark} {r['filename']}", end="")
        if r["status"] == "success":
            print(f" ({r['chunks_created']} 个片段)")
        else:
            print(f" — {r['error']}")


def parse_sse_response(resp) -> dict:
    """解析SSE格式的响应，拼出完整答案和来源"""
    answer = ""
    sources = []
    error = None

    event_type = "message"
    for raw_line in resp.iter_lines(decode_unicode=True):
        if raw_line is None or raw_line == "":
            continue
        if raw_line.startswith("event:"):
            event_type = raw_line[len("event:"):].strip()
        elif raw_line.startswith("data:"):
            data_str = raw_line[len("data:"):].strip()
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if event_type == "message":
                answer += data.get("delta", "")
            elif event_type == "sources":
                sources = data.get("sources", [])
            elif event_type == "error":
                error = data.get("message")

    return {"answer": answer, "sources": sources, "error": error}


def run_questions():
    print(f"\n{'='*70}")
    print("开始逐个提问")
    print(f"{'='*70}")

    for i, case in enumerate(TEST_QUESTIONS, start=1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}] {case['question']}")
        print(f"（预期来源: {case['pdf_filename']}）")

        try:
            resp = requests.post(
                f"{BASE_URL}/query",
                json={"question": case["question"]},
                timeout=60,
                stream=True,  # 关键：SSE响应必须用stream模式读取，不能直接.json()
            )
        except requests.exceptions.RequestException as e:
            print(f"[请求失败] {e}")
            continue

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
            match_mark = "✓" if case["pdf_filename"] in top["source"] else "⚠"
            print(f"实际引用来源: {top['source']} (相似度 {top['score']:.2f}) {match_mark}")
        else:
            print("（没有返回引用来源）")


if __name__ == "__main__":
    pdf_paths = check_pdfs_exist()
    check_backend_alive()
    upload_all_pdfs(pdf_paths)
    run_questions()
