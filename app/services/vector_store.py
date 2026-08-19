"""
向量数据库服务，封装 Pinecone 的读写。

设计上把"存储"完全隔离在这一层：
以后要换成 pgvector / Qdrant / Weaviate，只需要重写这个文件，
routers 和其他 services 完全不用动——这是面试里很好的
"依赖抽象/可替换设计"的例子。
"""
import uuid
import hashlib

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


UPSERT_BATCH_SIZE = 100

def _generate_chunk_id(source: str, chunk_index: int) -> str:
    """
    用 来源文件名+chunk序号 生成确定性ID，而不是随机UUID。
    好处：同一份文件重复上传时，ID完全相同，Pinecone会直接覆盖旧向量，
    而不是不断新增重复数据——这样"重复运行ingest脚本"这件事本身就变得安全、幂等。
    """
    raw = f"{source}::chunk_{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()


def upsert_chunks(chunks: list[str], embeddings: list[list[float]], source: str) -> int:
    index = get_index()
    vectors = [
        {
            "id": _generate_chunk_id(source, i),
            "values": vector,
            "metadata": {"content": chunk, "source": source},
        }
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings))
    ]

    for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
        batch = vectors[i:i + UPSERT_BATCH_SIZE]
        if batch:
            index.upsert(vectors=batch)

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
