"""
Embedding 服务：把文本变成向量。
封装成独立模块的好处：以后想换成 BGE 本地模型或其他厂商API，
只需要改这一个文件，不影响 ingest/query 的业务逻辑。
"""
from openai import OpenAI

from app.config import settings

_client = OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量embedding，减少API调用次数"""
    if not texts:
        return []
    response = _client.embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
    )
    # OpenAI 返回顺序与输入顺序一致
    return [item.embedding for item in response.data]


def embed_query(text: str) -> list[float]:
    """单条查询embedding，语义上和批量分开更清晰"""
    return embed_texts([text])[0]
