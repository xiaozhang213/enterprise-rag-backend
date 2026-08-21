"""
app/services/keyword_index.py

BM25关键词检索，与向量检索互补。向量检索擅长语义相似，
但容易漏掉依赖精确术语匹配的查询(缩写、专有名词等)，BM25直接补上这块。

索引持久化到本地磁盘(pickle)，避免每次重启都要从Pinecone重新拉取全部内容
(Pinecone不支持高效的"列出所有向量"操作)。

已知局限: 部署在Render免费层时，本地磁盘在实例重启/重新部署后会被清空，
需要重新运行 ingest_once.py。生产环境应该把这份索引挪到持久化存储。
"""
import pickle
from pathlib import Path
from threading import Lock

from typing import Optional

from rank_bm25 import BM25Okapi

INDEX_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "bm25_index.pkl"
_lock = Lock()

_documents: list[dict] = []  # [{"id": str, "content": str, "source": str}, ...]
_bm25: Optional[BM25Okapi] = None


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _rebuild_bm25() -> None:
    global _bm25
    _bm25 = BM25Okapi([_tokenize(d["content"]) for d in _documents]) if _documents else None


def _save() -> None:
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(_documents, f)


def _load() -> None:
    global _documents
    if INDEX_PATH.exists():
        with open(INDEX_PATH, "rb") as f:
            _documents = pickle.load(f)
        _rebuild_bm25()


_load()


def add_documents(chunk_ids: list[str], chunks: list[str], source: str) -> None:
    """加入新摄入的chunk，用跟向量库相同的确定性ID，方便后续结果合并去重"""
    with _lock:
        existing_ids = {d["id"] for d in _documents}
        for chunk_id, content in zip(chunk_ids, chunks):
            if chunk_id in existing_ids:
                continue  # 幂等设计，跟vector_store的确定性ID是同一个原则
            _documents.append({"id": chunk_id, "content": content, "source": source})
        _rebuild_bm25()
        _save()


def search(query: str, top_k: int) -> list[dict]:
    """返回 [{id, content, source, score}]，score归一化到0-1(除以本次结果里的最高分)"""
    if _bm25 is None or not _documents:
        return []

    scores = _bm25.get_scores(_tokenize(query))
    max_score = max(scores) if len(scores) and max(scores) > 0 else 1.0

    ranked = sorted(zip(_documents, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {"id": d["id"], "content": d["content"], "source": d["source"], "score": s / max_score}
        for d, s in ranked
        if s > 0
    ]


def clear() -> None:
    global _documents, _bm25
    with _lock:
        _documents = []
        _bm25 = None
        if INDEX_PATH.exists():
            INDEX_PATH.unlink()