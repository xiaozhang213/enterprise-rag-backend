"""
eval/build_golden.py

从公开数据集 Vectara Open RAG Benchmark (HF: vectara/open_ragbench, CC-BY-NC-4.0)
构建 eval/golden_set.jsonl，而不是用LLM自己出题——避免"自己出题自己考"。
全程不调用任何LLM。

数据集文件(pdf/arxiv/ 下，均为 {key: value} 的JSON对象)：
- queries.json : {query_id: {"query", "type": abstractive|extractive, "source": text|text-image|text-table|text-table-image}}
- answers.json : {query_id: "answer text"}
- qrels.json   : {query_id: {"doc_id": arxiv_id, "section_id": int}}   每个query只对应一篇文档
- pdf_urls.json: {arxiv_id: "https://arxiv.org/pdf/<id>"}

流程：
1. 打印每个文件的一条原始样例(确认真实字段名)
2. 从"有query的论文"(约400篇，另外600篇是无query的hard negative)中按seed=42抽 N_DOCS 篇，
   下载PDF到 eval/sample_docs/pdfs_bench/(已存在则跳过)
3. 只保留相关文档在抽样集合里的query，按 type 分层抽样约 N_QUERIES 条
4. 追加10条手写的不可回答问题(type="unanswerable")
5. 写出 eval/golden_set.jsonl 并打印各类型数量

运行：./.venv/bin/python eval/build_golden.py
"""
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

SCRIPT_DIR = Path(__file__).resolve().parent
PDF_DIR = SCRIPT_DIR / "sample_docs" / "pdfs_bench"
OUT_PATH = SCRIPT_DIR / "golden_set.jsonl"

DATASET_ID = "vectara/open_ragbench"
DATASET_DIR = "pdf/arxiv"
SEED = 42
N_DOCS = 25
N_QUERIES = 60
ARXIV_DELAY_SECONDS = 3  # arXiv 要求批量下载时请求间隔约3秒
USER_AGENT = "enterprise-rag-backend-eval/1.0 (golden set builder)"

# 和抽样论文、以及现有10篇ML论文都无关的问题，期望系统拒答
UNANSWERABLE_QUESTIONS = [
    "How long should I knead bread dough when making a sourdough loaf?",
    "What is the best way to make a classic Italian carbonara without cream?",
    "At what internal temperature is a roast chicken safe to eat?",
    "Who won the FIFA World Cup in 2018?",
    "How many players are on the field for each team in a rugby union match?",
    "What is the offside rule in ice hockey?",
    "How many days of paid vacation do new employees get per year?",
    "What is the company's parental leave policy?",
    "How do I submit a reimbursement request for business travel expenses?",
    "What is the dress code for the office on Fridays?",
]


def load_dataset_files() -> dict:
    files = {}
    for name in ["queries", "answers", "qrels", "pdf_urls"]:
        path = hf_hub_download(DATASET_ID, f"{DATASET_DIR}/{name}.json", repo_type="dataset")
        with open(path, encoding="utf-8") as f:
            files[name] = json.load(f)
    return files


def print_raw_examples(files: dict) -> None:
    print(f"=== 原始样例 ({DATASET_ID}) ===")
    qid = next(iter(files["queries"]))
    for name in ["queries", "answers", "qrels"]:
        print(f"\n{name}.json ({len(files[name])} 条):")
        print(json.dumps({qid: files[name][qid]}, ensure_ascii=False, indent=2))
    doc_id = files["qrels"][qid]["doc_id"]
    print(f"\npdf_urls.json ({len(files['pdf_urls'])} 条):")
    print(json.dumps({doc_id: files["pdf_urls"][doc_id]}, indent=2))
    print()


def pdf_filename(doc_id: str) -> str:
    # 入库时 source 就是文件名，expected_sources 必须和它完全一致
    return f"{doc_id}.pdf"


def download_pdfs(doc_ids: list[str], urls: dict) -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    for i, doc_id in enumerate(doc_ids, start=1):
        target = PDF_DIR / pdf_filename(doc_id)
        if target.exists():
            print(f"  [{i}/{len(doc_ids)}] 已存在，跳过 {target.name}")
            continue

        resp = session.get(urls[doc_id], timeout=120)
        resp.raise_for_status()
        if not resp.content.startswith(b"%PDF"):
            raise RuntimeError(f"{urls[doc_id]} 返回的不是PDF (content-type={resp.headers.get('content-type')})")

        # 先写临时文件再改名，中途中断不会留下半个PDF被当成"已存在"
        tmp = target.with_suffix(".pdf.part")
        tmp.write_bytes(resp.content)
        tmp.rename(target)
        print(f"  [{i}/{len(doc_ids)}] 已下载 {target.name} ({len(resp.content) // 1024} KB)")
        time.sleep(ARXIV_DELAY_SECONDS)


def stratified_sample(query_ids: list[str], queries: dict, n: int, rng: random.Random) -> list[str]:
    """按 type 均分名额；某一类不够时，剩余名额由其他类补足"""
    by_type = defaultdict(list)
    for qid in sorted(query_ids):  # 先排序，保证同一seed结果可复现
        by_type[queries[qid]["type"]].append(qid)
    for ids in by_type.values():
        rng.shuffle(ids)

    types = sorted(by_type)
    quota = {t: n // len(types) for t in types}
    for t in types[: n % len(types)]:
        quota[t] += 1

    picked = {t: by_type[t][: quota[t]] for t in types}
    shortfall = n - sum(len(v) for v in picked.values())
    for t in types:
        if shortfall <= 0:
            break
        extra = by_type[t][len(picked[t]) : len(picked[t]) + shortfall]
        picked[t] += extra
        shortfall -= len(extra)

    return [qid for t in types for qid in picked[t]]


def main():
    files = load_dataset_files()
    queries, answers, qrels, urls = files["queries"], files["answers"], files["qrels"], files["pdf_urls"]
    print_raw_examples(files)

    rng = random.Random(SEED)

    positive_docs = sorted({rel["doc_id"] for rel in qrels.values()} & set(urls))
    sampled_docs = sorted(rng.sample(positive_docs, N_DOCS))
    print(f"从 {len(positive_docs)} 篇有query的论文中抽取 {N_DOCS} 篇 (seed={SEED})，下载到 {PDF_DIR}")
    download_pdfs(sampled_docs, urls)

    doc_set = set(sampled_docs)
    candidate_ids = [qid for qid, rel in qrels.items() if rel["doc_id"] in doc_set and qid in queries and qid in answers]
    print(f"\n抽样论文覆盖的query: {len(candidate_ids)} 条，类型分布 "
          f"{dict(Counter(queries[q]['type'] for q in candidate_ids))}")

    selected = stratified_sample(candidate_ids, queries, N_QUERIES, rng)

    rows = [
        {
            "id": f"orb-{qid}",
            "question": queries[qid]["query"],
            "reference_answer": answers[qid],
            "expected_sources": [pdf_filename(qrels[qid]["doc_id"])],
            "type": queries[qid]["type"],
            # 数据集的 source 字段：问题依据的内容模态。我们的解析器只抽文本，
            # 依赖图片/表格的题目大概率答不好，分析结果时需要按它拆开看
            "modality": queries[qid]["source"],
            "verified": False,
        }
        for qid in selected
    ]
    rows += [
        {
            "id": f"unans-{i:02d}",
            "question": q,
            "reference_answer": "REFUSE",
            "expected_sources": [],
            "type": "unanswerable",
            "modality": None,
            "verified": False,
        }
        for i, q in enumerate(UNANSWERABLE_QUESTIONS, start=1)
    ]

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n已写入 {OUT_PATH.relative_to(SCRIPT_DIR.parent)}，共 {len(rows)} 条")
    print("按 type:", dict(Counter(r["type"] for r in rows)))
    print("按 type x modality:", dict(Counter((r["type"], r["modality"]) for r in rows if r["modality"])))
    print("涉及论文数:", len({s for r in rows for s in r["expected_sources"]}))


if __name__ == "__main__":
    main()
