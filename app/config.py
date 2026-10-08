"""
全局配置。
所有可调参数集中在这里，方便面试时讲清楚"我做了哪些可配置的设计"。
"""
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenAI
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str = "enterprise-kb"
    pinecone_dimension: int = 1536

    # Retrieval / Chunking
    top_k: int = 4
    chunk_size: int = 600
    chunk_overlap: int = 80
    vector_weight: float = 0.6
    keyword_weight: float = 0.4

    # Agentic检索(/query/agentic)
    agent_max_rewrites: int = 1
    agent_max_subqueries: int = 3

    # API Key鉴权(仅用于写操作/ingest)，留空则关闭鉴权(方便本地开发)
    api_key: Optional[str] = None

    # 速率限制
    query_rate_limit: str = "30/hour"
    ingest_rate_limit: str = "10/hour"

    # CORS
    allowed_origins: str = "http://localhost:3000"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


settings = Settings()
