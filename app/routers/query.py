"""
/query 路由：接收用户问题，完成"检索 -> 生成答案"全流程。
"""
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.config import settings
from app.models.schemas import QueryRequest
from app.services import embeddings, llm, vector_store

from app.services import retrieval

import logging

logger = logging.getLogger("enterprise_rag.query")


router = APIRouter(prefix="/query", tags=["query"])


def _sse_event(event: str, data: dict) -> str:
    """把数据包装成SSE协议格式：event类型 + JSON数据 + 空行分隔"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("")
async def query_knowledge_base(request: QueryRequest):
    top_k = request.top_k or settings.top_k

    # 检索阶段不是耗时瓶颈，保持同步调用，只对LLM生成做流式
    try:
        query_vector = embeddings.embed_query(request.question)
        matches = retrieval.hybrid_search(query_vector, request.question, top_k=top_k)
    except Exception:
        logger.exception("检索失败: question=%s", request.question)  # 加这一行
        raise HTTPException(status_code=503, detail="检索服务暂时不可用，请稍后重试")
   
    def event_stream():
        if not matches:
            yield _sse_event(
                "message", {"delta": "知识库中暂无文档，或未找到相关信息，请先上传文档。"}
            )
            yield _sse_event("sources", {"sources": []})
            yield _sse_event("done", {})
            return

        try:
            for delta in llm.generate_answer_stream(request.question, matches):
                yield _sse_event("message", {"delta": delta})
        except Exception:
            yield _sse_event("error", {"message": "生成答案时出错，请稍后重试"})
            return

        sources = [
            {"content": m["content"], "source": m["source"], "score": m["score"]}
            for m in matches
        ]
        yield _sse_event("sources", {"sources": sources})
        yield _sse_event("done", {})

    return StreamingResponse(event_stream(), media_type="text/event-stream")