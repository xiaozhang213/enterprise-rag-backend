"""
向量数据库服务，封装 Pinecone 的读写。

设计上把"存储"完全隔离在这一层：
以后要换成 pgvector / Qdrant / Weaviate，只需要重写这个文件，
routers 和其他 services 完全不用动——这是面试里很好的
"依赖抽象/可替换设计"的例子。
"""
import uuid

from pinecone import Pinecone, ServerlessSpec

from app.config import settings

_pc = Pinecone(api_key=settings.pinecone_api_key)


def _ensure_index_exists() -> None:
    existing_indexes = [idx["name"] for idx in _pc.list_indexes()]
    if settings.pinecone_index_name not in existing_indexes:
        _pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.pinecone_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )


def get_index():
    _ensure_index_exists()
    return _pc.Index(settings.pinecone_index_name)


def upsert_chunks(chunks: list[str], embeddings: list[list[float]], source: str) -> int:
    """把chunk文本+向量+来源写入向量库"""
    index = get_index()
    vectors = []
    for chunk, vector in zip(chunks, embeddings):
        vectors.append(
            {
                "id": str(uuid.uuid4()),
                "values": vector,
                "metadata": {"content": chunk, "source": source},
            }
        )
    if vectors:
        index.upsert(vectors=vectors)
    return len(vectors)


def query_similar(query_vector: list[float], top_k: int) -> list[dict]:
    """检索最相似的 top_k 个chunk，返回文本+来源+相似度分数"""
    index = get_index()
    result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    matches = []
    for match in result.get("matches", []):
        metadata = match.get("metadata", {})
        matches.append(
            {
                "content": metadata.get("content", ""),
                "source": metadata.get("source", "unknown"),
                "score": match.get("score", 0.0),
            }
        )
    return matches
