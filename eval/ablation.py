"""
eval/ablation.py

检索策略消融实验：同一份golden set、同一个生成调用，只换检索方式，比较四种配置：
  A  vector only        embed_query -> vector_store.query_similar
  B  hybrid             retrieval.hybrid_search (vector + BM25, RRF)
  C  hybrid + condense  follow_up行先用上一条golden问题做2轮历史改写问题，其余行等同B
  D  agentic            agent_graph (plan -> retrieve -> grade -> rewrite*)

全程进程内调用 app.services，不走HTTP，不需要启动服务。详见 eval/README_eval.md。

  ./.venv/bin/python eval/ablation.py                  # 全量
  ./.venv/bin/python eval/ablation.py --limit 5        # 冒烟测试
  ./.venv/bin/python eval/ablation.py --no-ragas       # 只算检索/拒答/延迟指标
"""
import argparse
import json
import logging
import math
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd
from openai import RateLimitError

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

from app.config import settings  # noqa: E402
from app.routers.agentic_query import NOT_FOUND_ANSWER  # noqa: E402
from app.services import agent_graph, embeddings, llm, retrieval, vector_store  # noqa: E402

logger = logging.getLogger("ablation")

ALL_CONFIGS = ["A", "B", "C", "D"]
CONFIG_NAMES = {"A": "vector only", "B": "hybrid", "C": "hybrid + condense", "D": "agentic"}
K = 4  # hit@K / MRR@K 的截断位置，和默认 top_k 一致
DECLINE_PHRASES = ["no relevant information was found", "未找到相关信息"]
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_recall", "factual_correctness"]
DEFAULT_GOLDEN = EVAL_DIR / "golden_set_ml.jsonl"
CACHE_PATH = EVAL_DIR / "results" / "cache.json"

T = TypeVar("T")


@dataclass
class Retrieved:
    query: str  # 实际用于检索和生成的问题(C的follow_up行是改写后的)
    chunks: list[dict]
    llm_calls: int = 0  # 检索阶段的chat调用次数(不含embedding)
    refused: bool = False  # 只有D会在检索阶段直接拒答


@dataclass
class Record:
    config: str
    id: str
    type: str
    question: str
    query_used: str
    retrieved_sources: list[str]
    contexts: list[str]
    answer: str
    refused: bool
    latency_s: float
    llm_calls: int
    scores: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------- golden set

def load_golden(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select_rows(rows: list[dict], limit: int | None) -> list[dict]:
    """按type轮流取，保证 --limit 很小时也能覆盖每种题型"""
    if not limit or limit >= len(rows):
        return rows
    by_type: dict[str, list[dict]] = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r)
    picked: list[dict] = []
    while len(picked) < limit:
        for bucket in by_type.values():
            if bucket and len(picked) < limit:
                picked.append(bucket.pop(0))
    return picked


def history_for(row: dict, previous: dict | None) -> list[dict]:
    """follow_up行：用golden文件里的上一条问答构造2轮历史"""
    if not row.get("follow_up") or previous is None:
        return []
    return [
        {"role": "user", "content": previous["question"]},
        {"role": "assistant", "content": previous["reference_answer"]},
    ]


# ---------------------------------------------------------------- retrieval configs

def retrieve_vector(question: str) -> Retrieved:
    return Retrieved(question, vector_store.query_similar(embeddings.embed_query(question), top_k=settings.top_k))


def retrieve_hybrid(question: str) -> Retrieved:
    vector = embeddings.embed_query(question)
    return Retrieved(question, retrieval.hybrid_search(vector, question, top_k=settings.top_k))


def retrieve_condensed(question: str, history: list[dict]) -> Retrieved:
    standalone = llm.condense_question(question, history)  # 无历史时直接返回原问题，不调LLM
    result = retrieve_hybrid(standalone)
    result.llm_calls = 1 if history else 0
    return result


def retrieve_agentic(question: str) -> Retrieved:
    deps = agent_graph.default_deps()
    calls = 0
    inner = deps.llm_json

    def counted(prompt: str) -> dict:
        nonlocal calls
        calls += 1
        return inner(prompt)

    deps.llm_json = counted
    state = agent_graph.build_graph(deps).invoke({"question": question})
    return Retrieved(question, state.get("chunks", []), llm_calls=calls, refused=bool(state.get("refused")))


def run_retrieval(config: str, question: str, history: list[dict]) -> Retrieved:
    if config == "A":
        return retrieve_vector(question)
    if config == "B":
        return retrieve_hybrid(question)
    if config == "C":
        return retrieve_condensed(question, history)
    if config == "D":
        return retrieve_agentic(question)
    raise ValueError(f"unknown config {config}")


# ---------------------------------------------------------------- generation + one question

def generate_answer(question: str, chunks: list[dict]) -> str:
    """llm.generate_answer 在 llm.py 里被注释掉了；拼接流式输出等价于同一个非流式调用
    (同prompt、同模型、同temperature)"""
    return "".join(llm.generate_answer_stream(question, chunks))


def is_refusal(answer: str) -> bool:
    text = answer.casefold()
    return any(p.casefold() in text for p in DECLINE_PHRASES)


def with_retry(fn: Callable[[], T], attempts: int = 5, base_delay: float = 2.0) -> T:
    for attempt in range(attempts):
        try:
            return fn()
        except RateLimitError:
            if attempt == attempts - 1:
                raise
            delay = base_delay * 2**attempt
            logger.warning("OpenAI rate limit, retrying in %.0fs", delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def run_question(config: str, row: dict, history: list[dict]) -> Record:
    start = time.perf_counter()
    retrieved = run_retrieval(config, row["question"], history)
    if retrieved.refused:
        answer, gen_calls = NOT_FOUND_ANSWER, 0  # 和 /query/agentic 的行为一致：拒答时不调生成
    else:
        answer, gen_calls = generate_answer(retrieved.query, retrieved.chunks), 1
    latency = time.perf_counter() - start
    return Record(
        config=config,
        id=row["id"],
        type=row["type"],
        question=row["question"],
        query_used=retrieved.query,
        retrieved_sources=[c["source"] for c in retrieved.chunks],
        contexts=[c["content"] for c in retrieved.chunks],
        answer=answer,
        refused=retrieved.refused or is_refusal(answer),
        latency_s=round(latency, 3),
        llm_calls=retrieved.llm_calls + gen_calls,
    )


# ---------------------------------------------------------------- cache

class Cache:
    """(config, id) -> 生成结果。问题文本变了就视为失效，避免改了golden还读到旧答案"""

    def __init__(self, path: Path, enabled: bool):
        self.path, self.enabled = path, enabled
        self.lock = threading.Lock()
        self.data: dict[str, dict] = {}
        if enabled and path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def key(config: str, row_id: str) -> str:
        return f"{config}|{row_id}"

    def get(self, config: str, row: dict) -> Record | None:
        if not self.enabled:
            return None
        hit = self.data.get(self.key(config, row["id"]))
        if hit is None or hit["question"] != row["question"]:
            return None
        return Record(**{**hit, "scores": {}})

    def put(self, record: Record) -> None:
        if not self.enabled:
            return
        with self.lock:
            self.data[self.key(record.config, record.id)] = {**asdict(record), "scores": {}}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(self.path)


def collect_records(configs: list[str], rows: list[dict], all_rows: list[dict], cache: Cache, workers: int) -> tuple[list[Record], list[str]]:
    previous = {r["id"]: (all_rows[i - 1] if i > 0 else None) for i, r in enumerate(all_rows)}
    tasks = [(c, r) for c in configs for r in rows]
    failed: list[str] = []

    def work(task: tuple[str, dict]) -> Record | None:
        config, row = task
        cached = cache.get(config, row)
        if cached is not None:
            return cached
        try:
            record = with_retry(lambda: run_question(config, row, history_for(row, previous[row["id"]])))
        except Exception:
            logger.exception("FAILED %s/%s, skipped", config, row["id"])
            failed.append(f"{config}/{row['id']}")
            return None
        cache.put(record)
        return record

    if any(cache.get(c, r) is None for c, r in tasks):
        # 预热一次Pinecone/OpenAI连接，否则冷启动开销全算在最先跑的配置(A)头上，延迟对比失真
        retrieve_hybrid("warm up")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(work, tasks))
    return [r for r in results if r is not None], failed


# ---------------------------------------------------------------- retrieval metrics

def reciprocal_rank(sources: list[str], expected: list[str], k: int = K) -> float:
    for rank, src in enumerate(sources[:k], start=1):
        if src in expected:
            return 1.0 / rank
    return 0.0


def source_recall(sources: list[str], expected: list[str], k: int = K) -> float:
    """multi_hop题需要多篇文档：top-k覆盖了期望文档的比例"""
    return len(set(sources[:k]) & set(expected)) / len(expected) if expected else math.nan


# ---------------------------------------------------------------- RAGAS

def ragas_scores(records: list[Record], references: dict[str, str]) -> None:
    """只对可回答题算RAGAS；分数写回 record.scores。evaluate 内部失败的样本记为NaN"""
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import FactualCorrectness, answer_relevancy, context_recall, faithfulness
    from ragas.run_config import RunConfig

    from ragas_eval import attach_llm_and_embeddings  # 复用 eval/ragas_eval.py 的挂载逻辑

    targets = [r for r in records if r.type != "unanswerable"]
    if not targets:
        return
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=settings.openai_chat_model, api_key=settings.openai_api_key))
    evaluator_emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=settings.openai_embedding_model, api_key=settings.openai_api_key)
    )
    # LangChain的ChatOpenAI包装只返回1个generation，answer_relevancy默认要3个(会刷warning然后照样用1个)，
    # 显式设成1，让实际行为和配置一致
    answer_relevancy.strictness = 1
    metrics = attach_llm_and_embeddings(
        [faithfulness, answer_relevancy, context_recall, FactualCorrectness()], evaluator_llm, evaluator_emb
    )
    dataset = EvaluationDataset.from_list([
        {
            "user_input": r.question,
            "retrieved_contexts": r.contexts or [""],
            "response": r.answer,
            "reference": references[r.id],
        }
        for r in targets
    ])
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        run_config=RunConfig(max_workers=16, timeout=120, max_retries=6),
        raise_exceptions=False,
        show_progress=False,
    )
    df = result.to_pandas()
    columns = {m: next((c for c in df.columns if c.startswith(m)), None) for m in RAGAS_METRICS}
    for record, (_, row) in zip(targets, df.iterrows()):
        record.scores = {m: float(row[c]) if c else math.nan for m, c in columns.items()}


# ---------------------------------------------------------------- tables

def to_frame(records: list[Record], golden: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        expected = golden[r.id]["expected_sources"]
        answerable = r.type != "unanswerable"
        rows.append({
            "config": r.config,
            "id": r.id,
            "type": r.type,
            "follow_up": bool(golden[r.id].get("follow_up")),
            "question": r.question,
            "query_used": r.query_used,
            "retrieved_sources": "|".join(r.retrieved_sources),
            "answer": r.answer,
            "refused": r.refused,
            "latency_s": r.latency_s,
            "llm_calls": r.llm_calls,
            "hit@4": float(any(s in expected for s in r.retrieved_sources[:K])) if answerable else math.nan,
            "rr@4": reciprocal_rank(r.retrieved_sources, expected) if answerable else math.nan,
            "source_recall@4": source_recall(r.retrieved_sources, expected) if answerable else math.nan,
            **{m: r.scores.get(m, math.nan) for m in RAGAS_METRICS},
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(by, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        ans, unans = g[g["type"] != "unanswerable"], g[g["type"] == "unanswerable"]
        rows.append({
            **dict(zip(by, keys)),
            "n": len(g),
            "hit@4": ans["hit@4"].mean(),
            "MRR@4": ans["rr@4"].mean(),
            "src_recall@4": ans["source_recall@4"].mean(),
            "refusal_unans": unans["refused"].mean() if len(unans) else math.nan,
            "false_refusal_ans": ans["refused"].mean() if len(ans) else math.nan,
            "lat_p50": float(np.percentile(g["latency_s"], 50)),
            "lat_p95": float(np.percentile(g["latency_s"], 95)),
            "llm_calls": g["llm_calls"].mean(),
            **{m: ans[m].mean() for m in RAGAS_METRICS},
        })
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return "–" if math.isnan(v) else f"{v:.3f}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def write_outputs(df: pd.DataFrame, out_dir: Path, args: argparse.Namespace, failed: list[str]) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "results.csv", index=False)

    overall = summarize(df, ["config"])
    overall.insert(1, "name", overall["config"].map(CONFIG_NAMES))
    per_type = summarize(df, ["config", "type"])[
        ["config", "type", "n", "hit@4", "MRR@4", "src_recall@4", "refusal_unans", "false_refusal_ans",
         "faithfulness", "factual_correctness", "llm_calls"]
    ]
    summary = "\n\n".join([
        f"# Ablation results ({datetime.now():%Y-%m-%d %H:%M})",
        f"golden: `{args.golden.name}`, rows: {df['id'].nunique()}, configs: {','.join(args.configs)}, "
        f"model: `{settings.openai_chat_model}`, top_k: {settings.top_k}, workers: {args.workers}, "
        f"ragas: {'off' if args.no_ragas else 'on'}",
        "## By config\n\n" + markdown_table(overall),
        "## By config x type\n\n" + markdown_table(per_type),
        f"Failed (skipped): {', '.join(failed) if failed else 'none'}",
    ])
    (out_dir / "summary.md").write_text(summary + "\n", encoding="utf-8")
    return summary


# ---------------------------------------------------------------- main

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", default=",".join(ALL_CONFIGS), type=lambda s: [c.strip().upper() for c in s.split(",")])
    p.add_argument("--limit", type=int, default=None, help="只跑N行(按题型轮流取)")
    p.add_argument("--out", type=Path, default=EVAL_DIR / "results")
    p.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    p.add_argument("--workers", type=int, default=4, help="并发数；测准确延迟请用1")
    p.add_argument("--no-cache", action="store_true", help="不读也不写生成结果缓存")
    p.add_argument("--no-ragas", action="store_true", help="跳过RAGAS(省时间和费用)")
    args = p.parse_args()
    unknown = set(args.configs) - set(ALL_CONFIGS)
    if unknown:
        p.error(f"unknown configs: {sorted(unknown)}")
    return args


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    # 需求指定沿用 ragas_eval.py 的 Langchain*Wrapper，ragas 0.3.9 对它们报的弃用警告直接屏蔽
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
    warnings.filterwarnings("ignore", message=".*Wrapper is deprecated.*")
    args = parse_args()
    all_rows = load_golden(args.golden)
    rows = select_rows(all_rows, args.limit)
    golden = {r["id"]: r for r in all_rows}
    cache = Cache(CACHE_PATH, enabled=not args.no_cache)

    t0 = time.perf_counter()
    print(f"Running configs {args.configs} on {len(rows)} rows ({args.golden.name})...")
    records, failed = collect_records(args.configs, rows, all_rows, cache, args.workers)
    print(f"Generation done in {time.perf_counter() - t0:.0f}s ({len(records)} ok, {len(failed)} failed)")

    if not args.no_ragas:
        t1 = time.perf_counter()
        ragas_scores(records, {r["id"]: r["reference_answer"] for r in all_rows})
        print(f"RAGAS done in {time.perf_counter() - t1:.0f}s")

    out_dir = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
    summary = write_outputs(to_frame(records, golden), out_dir, args, failed)
    print("\n" + summary)
    print(f"\nWrote {out_dir / 'results.csv'} and {out_dir / 'summary.md'} (total {time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
