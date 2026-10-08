"""
app/services/agent_graph.py

用LangGraph显式编排的"agentic检索"：不是让LLM自由调用工具(ReAct)，
而是一张固定形状的状态图，每一步做什么、最多循环几次都是确定的：

    START -> plan -> retrieve -> grade --(充分 / 达到改写上限)--> finalize -> END
                        ^                |
                        |              (不充分)
                        +--- rewrite <---+

- plan:     把问题拆成 <= max_subqueries 个子查询(多跳/对比类问题需要分别检索)
- retrieve: 每个子查询跑一次 hybrid_search，按chunk id合并去重
- grade:    LLM判断现有chunk够不够回答，不够的话说明缺什么(missing)
- rewrite:  根据missing改写子查询，再检索一轮
- finalize: 仍不充分则 refused=True，由路由层直接返回"未找到"，不让LLM硬编

外部依赖(LLM、检索)全部通过 AgentDeps 注入，测试时可以传fake，无需网络。
"""
import json
import operator
from dataclasses import dataclass
from functools import partial
from typing import Annotated, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

# 节点名。注意 state 的 key 不能和节点名重名，否则LangGraph会报错。
PLAN = "plan"
RETRIEVE = "retrieve"
GRADE = "grade"
REWRITE = "rewrite"
FINALIZE = "finalize"


class AgentState(TypedDict, total=False):
    question: str
    sub_queries: list[str]
    per_query: dict[str, int]  # 子查询 -> 命中chunk数，方便trace/调试
    rewrites_used: int
    sufficient: bool
    missing: list[str]
    chunks: list[dict]
    refused: bool
    # 每个节点只返回自己新增的那一行，靠 operator.add 累加，而不是互相覆盖
    trace: Annotated[list[str], operator.add]


@dataclass
class AgentDeps:
    llm_json: Callable[[str], dict]  # prompt -> 解析好的JSON对象
    search: Callable[[str], list[dict]]  # query -> [{"id", "content", "source", "score"}, ...]
    max_rewrites: int = 1
    max_subqueries: int = 3


PLAN_PROMPT = """Split the user's question into at most {n} standalone search queries for a document knowledge base.
Use one query if the question is about a single topic; use several only if it covers distinct topics (e.g. a comparison).
Respond with JSON only: {{"sub_queries": ["...", "..."]}}

Question: {question}
"""

GRADE_PROMPT = """Decide whether the retrieved passages below contain enough information to fully answer the question.
Respond with JSON only: {{"sufficient": true or false, "missing": ["specific information still needed", ...]}}
Use an empty "missing" list when sufficient is true.

Question: {question}

Passages:
{passages}
"""

REWRITE_PROMPT = """The previous search queries did not retrieve enough information to answer the question.
Write at most {n} new search queries that target the missing information.
Respond with JSON only: {{"sub_queries": ["...", "..."]}}

Question: {question}
Previous queries: {previous}
Missing information: {missing}
"""


def _dedupe_queries(queries, limit: int) -> list[str]:
    """去空白、大小写不敏感去重，保留原始顺序，截断到limit个"""
    seen = set()
    result = []
    for q in queries or []:
        if not isinstance(q, str):
            continue
        q = q.strip()
        key = q.casefold()
        if q and key not in seen:
            seen.add(key)
            result.append(q)
    return result[:limit]


def _format_passages(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{i} | {c['source']}]\n{c['content']}" for i, c in enumerate(chunks, start=1)
    )


def plan(state: AgentState, deps: AgentDeps) -> dict:
    question = state["question"]
    out = deps.llm_json(PLAN_PROMPT.format(n=deps.max_subqueries, question=question))
    sub_queries = _dedupe_queries(out.get("sub_queries"), deps.max_subqueries)
    if not sub_queries:  # LLM输出不可用时退化成"原问题检索一次"，等价于普通 /query
        sub_queries = [question]
    return {
        "sub_queries": sub_queries,
        "rewrites_used": 0,
        "chunks": [],
        "per_query": {},
        "trace": [f"Planned {len(sub_queries)} sub-quer{'y' if len(sub_queries) == 1 else 'ies'}: {sub_queries}"],
    }


def retrieve(state: AgentState, deps: AgentDeps) -> dict:
    # 改写后的新一轮检索结果和上一轮的合并：上一轮找到的有效信息不应该丢
    merged = {c["id"]: c for c in state.get("chunks", [])}
    per_query = dict(state.get("per_query", {}))
    new_count = 0

    for q in state["sub_queries"]:
        hits = deps.search(q)
        per_query[q] = len(hits)
        for c in hits:
            existing = merged.get(c["id"])
            if existing is None:
                merged[c["id"]] = c
                new_count += 1
            elif c["score"] > existing["score"]:
                merged[c["id"]] = c

    chunks = sorted(merged.values(), key=lambda c: c["score"], reverse=True)
    return {
        "chunks": chunks,
        "per_query": per_query,
        "trace": [f"Retrieved {new_count} new chunk(s) for {len(state['sub_queries'])} sub-quer"
                  f"{'y' if len(state['sub_queries']) == 1 else 'ies'}; {len(chunks)} unique total"],
    }


def grade(state: AgentState, deps: AgentDeps) -> dict:
    chunks = state.get("chunks", [])
    if not chunks:  # 什么都没检索到，不需要再花一次LLM调用去判断
        return {
            "sufficient": False,
            "missing": [state["question"]],
            "trace": ["Graded: insufficient (no chunks retrieved)"],
        }

    out = deps.llm_json(GRADE_PROMPT.format(question=state["question"], passages=_format_passages(chunks)))
    sufficient = out.get("sufficient") is True
    missing = [] if sufficient else [m for m in out.get("missing") or [] if isinstance(m, str) and m.strip()]
    line = "Graded: sufficient" if sufficient else f"Graded: insufficient, missing {missing}"
    return {"sufficient": sufficient, "missing": missing, "trace": [line]}


def rewrite(state: AgentState, deps: AgentDeps) -> dict:
    previous = state["sub_queries"]
    missing = state.get("missing") or [state["question"]]
    out = deps.llm_json(REWRITE_PROMPT.format(
        n=deps.max_subqueries,
        question=state["question"],
        previous=json.dumps(previous, ensure_ascii=False),
        missing=json.dumps(missing, ensure_ascii=False),
    ))
    sub_queries = _dedupe_queries(out.get("sub_queries"), deps.max_subqueries)
    if not sub_queries:
        sub_queries = _dedupe_queries(missing, deps.max_subqueries)
    rewrites_used = state.get("rewrites_used", 0) + 1
    return {
        "sub_queries": sub_queries,
        "rewrites_used": rewrites_used,
        "trace": [f"Rewrite {rewrites_used}/{deps.max_rewrites}: {sub_queries}"],
    }


def finalize(state: AgentState) -> dict:
    refused = not state.get("sufficient", False)
    line = (
        f"Refused: still insufficient after {state.get('rewrites_used', 0)} rewrite(s)"
        if refused
        else f"Answering from {len(state.get('chunks', []))} chunk(s)"
    )
    return {"refused": refused, "trace": [line]}


def route_after_grade(state: AgentState, max_rewrites: int) -> str:
    if state.get("sufficient") or state.get("rewrites_used", 0) >= max_rewrites:
        return FINALIZE
    return REWRITE


def build_graph(deps: AgentDeps):
    graph = StateGraph(AgentState)
    graph.add_node(PLAN, partial(plan, deps=deps))
    graph.add_node(RETRIEVE, partial(retrieve, deps=deps))
    graph.add_node(GRADE, partial(grade, deps=deps))
    graph.add_node(REWRITE, partial(rewrite, deps=deps))
    graph.add_node(FINALIZE, finalize)

    graph.add_edge(START, PLAN)
    graph.add_edge(PLAN, RETRIEVE)
    graph.add_edge(RETRIEVE, GRADE)
    graph.add_conditional_edges(
        GRADE,
        partial(route_after_grade, max_rewrites=deps.max_rewrites),
        {REWRITE: REWRITE, FINALIZE: FINALIZE},
    )
    graph.add_edge(REWRITE, RETRIEVE)
    graph.add_edge(FINALIZE, END)
    return graph.compile()


def default_deps(top_k: Optional[int] = None) -> AgentDeps:
    """生产环境依赖：复用现有的 llm 客户端、embeddings 和 hybrid_search。
    放在函数里延迟import，这样测试导入本模块时不需要OpenAI/Pinecone的key。"""
    from app.config import settings
    from app.services import embeddings, llm, retrieval

    k = top_k or settings.top_k

    def llm_json(prompt: str) -> dict:
        response = llm._client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(response.choices[0].message.content or "{}")
        except json.JSONDecodeError:
            return {}  # 各节点对空结果都有兜底

    def search(query: str) -> list[dict]:
        return retrieval.hybrid_search(embeddings.embed_query(query), query, top_k=k)

    return AgentDeps(
        llm_json=llm_json,
        search=search,
        max_rewrites=settings.agent_max_rewrites,
        max_subqueries=settings.agent_max_subqueries,
    )
