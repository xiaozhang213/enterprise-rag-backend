"""
/query 路由：接收用户问题，完成"检索 -> 生成答案"全流程。
"""
from fastapi import APIRouter

from app.config import settings
from app.models.schemas import QueryRequest, QueryResponse, SourceChunk
from app.services import embeddings, llm, vector_store

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    top_k = request.top_k or settings.top_k

    query_vector = embeddings.embed_query(request.question)
    matches = vector_store.query_similar(query_vector, top_k=top_k)

    if not matches:
        return QueryResponse(
            answer="知识库中暂无文档，或未找到相关信息，请先上传文档。",
            sources=[],
        )

    answer = llm.generate_answer(request.question, matches)

    sources = [
        SourceChunk(content=m["content"], source=m["source"], score=m["score"])
        for m in matches
    ]
    return QueryResponse(answer=answer, sources=sources)
