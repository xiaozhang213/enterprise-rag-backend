"""
Embedding 服务：把文本变成向量。
加入分批处理 + 指数退避重试，应对OpenAI的速率限制。
"""
import time

from openai import OpenAI, RateLimitError

from app.config import settings

_client = OpenAI(api_key=settings.openai_api_key)

EMBEDDING_BATCH_SIZE = 100  # 每次API调用最多处理100个chunk，避免撞TPM限制
MAX_RETRIES = 3


def _embed_batch_with_retry(texts: list[str]) -> list[list[float]]:
    for attempt in range(MAX_RETRIES):
        try:
            response = _client.embeddings.create(
                model=settings.openai_embedding_model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except RateLimitError:
            wait_time = 2 ** attempt  # 1s, 2s, 4s 指数退避
            print(f"[embedding] 遇到限流，{wait_time}s 后重试 (第{attempt + 1}次)")
            time.sleep(wait_time)
    raise RuntimeError("embedding 请求多次重试后仍然失败（限流）")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """分批处理，避免单次请求chunk数量过多导致限流或超出请求体大小限制"""
    if not texts:
        return []

    all_embeddings = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + EMBEDDING_BATCH_SIZE]
        all_embeddings.extend(_embed_batch_with_retry(batch))
    return all_embeddings


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]