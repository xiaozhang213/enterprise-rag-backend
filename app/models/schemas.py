"""
Pydantic 数据模型，定义接口的请求/响应结构。
用 TypeHint + Pydantic 而不是裸dict，是北美后端工程规范里的加分项。
"""
from typing import Optional

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    filename: str
    chunks_created: int
    status: str = "success"


class ChatTurn(BaseModel):
    role: str  # "user" 或 "assistant"
    content: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户提出的问题")
    top_k: Optional[int] = Field(None, description="可选，覆盖默认检索数量")
    history: list[ChatTurn] = Field(default_factory=list, description="可选，此前的对话历史，用于多轮追问改写")


class SourceChunk(BaseModel):
    content: str
    source: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
