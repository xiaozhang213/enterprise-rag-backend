"""
/query 路由：接收用户问题，完成"检索 -> 生成答案"全流程。
"""
import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.config import settings
from app.rate_limit import limiter
from app.models.schemas import QueryRequest
from app.services import embeddings, llm, retrieval, vector_store

logger = logging.getLogger("enterprise_rag.query")


router = APIRouter(prefix="/query", tags=["query"])


def _sse_event(event: str, data: dict) -> str:
    """把数据包装成SSE协议格式：event类型 + JSON数据 + 空行分隔"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("")
@limiter.limit(settings.query_rate_limit)
async def query_knowledge_base(request: Request, body: QueryRequest):
    # 参数名 request: QueryRequest 改成 body，避免跟slowapi需要的 Request 对象重名
    top_k = body.top_k or settings.top_k
    history = [turn.model_dump() for turn in body.history]

    # 检索阶段不是耗时瓶颈，保持同步调用，只对LLM生成做流式
    try:
        # 有历史时先把追问改写成独立问题，否则"那它多少钱？"这种问题检索不到东西
        standalone_question = llm.condense_question(body.question, history)
        query_vector = embeddings.embed_query(standalone_question)
        matches = retrieval.hybrid_search(query_vector, standalone_question, top_k=top_k)
    except Exception:
        logger.exception("检索失败: question=%s", body.question)
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
            for delta in llm.generate_answer_stream(standalone_question, matches, history=history):
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