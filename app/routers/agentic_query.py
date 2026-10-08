"""
app/routers/agentic_query.py

/query/agentic：和 /query 用同一个请求结构、同一套SSE格式，区别是检索阶段
换成 agent_graph 的"拆分 -> 检索 -> 评估 -> (改写重检)"流程。
保留成独立接口而不是替换 /query，方便直接对比两种检索方式的效果。

SSE事件顺序：step* -> message* -> sources -> done，出错时发 error。
"""
import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.config import settings
from app.models.schemas import QueryRequest
from app.rate_limit import limiter
from app.services import agent_graph, llm

logger = logging.getLogger("enterprise_rag.agentic_query")

router = APIRouter(prefix="/query/agentic", tags=["agentic-query"])

# 和 llm.SYSTEM_PROMPT 里约定的"资料不足"回答保持一致
NOT_FOUND_ANSWER = "According to the available information, no relevant information was found"


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("")
@limiter.limit(settings.query_rate_limit)
async def agentic_query(request: Request, body: QueryRequest):
    history = [turn.model_dump() for turn in body.history]

    # 同步生成器：StreamingResponse会放到线程池里跑，图里的阻塞LLM/检索调用不会卡住事件循环
    def event_stream():
        try:
            question = llm.condense_question(body.question, history)
            graph = agent_graph.build_graph(agent_graph.default_deps(top_k=body.top_k))
            final: dict = {}
            # stream_mode="updates"：每个节点跑完就拿到它的输出，step事件可以实时推给前端
            for update in graph.stream({"question": question}, stream_mode="updates"):
                for node_output in update.values():
                    for line in node_output.get("trace", []):
                        yield _sse_event("step", {"message": line})
                    final.update(node_output)
        except Exception:
            logger.exception("agentic检索失败: question=%s", body.question)
            yield _sse_event("error", {"message": "检索服务暂时不可用，请稍后重试"})
            return

        if final.get("refused"):
            yield _sse_event("message", {"delta": NOT_FOUND_ANSWER})
            yield _sse_event("sources", {"sources": []})
            yield _sse_event("done", {})
            return

        chunks = final["chunks"]
        try:
            for delta in llm.generate_answer_stream(question, chunks, history=history):
                yield _sse_event("message", {"delta": delta})
        except Exception:
            logger.exception("agentic生成失败: question=%s", body.question)
            yield _sse_event("error", {"message": "生成答案时出错，请稍后重试"})
            return

        sources = [{"content": c["content"], "source": c["source"], "score": c["score"]} for c in chunks]
        yield _sse_event("sources", {"sources": sources})
        yield _sse_event("done", {})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
