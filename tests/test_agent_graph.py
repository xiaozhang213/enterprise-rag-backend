"""
agent_graph 的离线测试：LLM和检索都用fake注入，不访问网络、不需要API key。
运行：pytest tests/
"""
from app.services.agent_graph import AgentDeps, build_graph


def chunk(cid: str, score: float = 0.5) -> dict:
    return {"id": cid, "content": f"content of {cid}", "source": f"{cid}.pdf", "score": score}


class FakeLLM:
    """按prompt类型返回预设结果；grade按调用次数依次取 grades 里的值。"""

    def __init__(self, sub_queries, grades, rewrites=None):
        self.sub_queries = sub_queries
        self.grades = list(grades)
        self.rewrites = rewrites or [["rewritten query"]]
        self.calls = {"plan": 0, "grade": 0, "rewrite": 0}

    def __call__(self, prompt: str) -> dict:
        if prompt.startswith("Split the user's question"):
            self.calls["plan"] += 1
            return {"sub_queries": self.sub_queries}
        if prompt.startswith("Decide whether"):
            sufficient = self.grades[min(self.calls["grade"], len(self.grades) - 1)]
            self.calls["grade"] += 1
            return {"sufficient": sufficient, "missing": [] if sufficient else ["more detail"]}
        if prompt.startswith("The previous search queries"):
            out = self.rewrites[min(self.calls["rewrite"], len(self.rewrites) - 1)]
            self.calls["rewrite"] += 1
            return {"sub_queries": out}
        raise AssertionError(f"unexpected prompt: {prompt[:40]}")


class FakeSearch:
    def __init__(self, results: dict, default=None):
        self.results = results
        self.default = default if default is not None else []
        self.queries = []

    def __call__(self, query: str) -> list[dict]:
        self.queries.append(query)
        return self.results.get(query, self.default)


def run(llm: FakeLLM, search: FakeSearch, max_rewrites: int = 1, max_subqueries: int = 3) -> dict:
    deps = AgentDeps(llm_json=llm, search=search, max_rewrites=max_rewrites, max_subqueries=max_subqueries)
    return build_graph(deps).invoke({"question": "What is LoRA?"})


def test_sufficient_on_first_pass_skips_rewrite():
    llm = FakeLLM(sub_queries=["LoRA"], grades=[True])
    search = FakeSearch({"LoRA": [chunk("a"), chunk("b")]})

    state = run(llm, search)

    assert state["sufficient"] is True
    assert state["refused"] is False
    assert state["rewrites_used"] == 0
    assert llm.calls == {"plan": 1, "grade": 1, "rewrite": 0}
    assert search.queries == ["LoRA"]
    assert {c["id"] for c in state["chunks"]} == {"a", "b"}
    assert len(state["trace"]) == 4  # plan, retrieve, grade, finalize


def test_insufficient_then_sufficient_after_one_rewrite():
    llm = FakeLLM(sub_queries=["LoRA"], grades=[False, True], rewrites=[["LoRA rank decomposition"]])
    search = FakeSearch({"LoRA": [chunk("a")], "LoRA rank decomposition": [chunk("b")]})

    state = run(llm, search, max_rewrites=2)

    assert state["sufficient"] is True
    assert state["refused"] is False
    assert state["rewrites_used"] == 1
    assert llm.calls == {"plan": 1, "grade": 2, "rewrite": 1}
    assert search.queries == ["LoRA", "LoRA rank decomposition"]
    # 改写后的检索结果和第一轮合并，而不是替换
    assert {c["id"] for c in state["chunks"]} == {"a", "b"}
    assert state["per_query"] == {"LoRA": 1, "LoRA rank decomposition": 1}


def test_never_sufficient_stops_at_max_rewrites_and_refuses():
    llm = FakeLLM(sub_queries=["LoRA"], grades=[False], rewrites=[["q1"], ["q2"], ["q3"]])
    search = FakeSearch({}, default=[chunk("a")])

    state = run(llm, search, max_rewrites=2)

    assert state["sufficient"] is False
    assert state["refused"] is True
    assert state["rewrites_used"] == 2
    assert llm.calls == {"plan": 1, "grade": 3, "rewrite": 2}
    assert search.queries == ["LoRA", "q1", "q2"]
    assert state["trace"][-1].startswith("Refused")


def test_empty_retrieval_refuses_without_grading_llm_call():
    llm = FakeLLM(sub_queries=["LoRA"], grades=[True])
    search = FakeSearch({})  # 知识库里什么都没有

    state = run(llm, search, max_rewrites=1)

    assert state["refused"] is True
    assert llm.calls["grade"] == 0  # 没有chunk时不浪费LLM调用去评估
    assert state["rewrites_used"] == 1


def test_sub_queries_and_chunks_are_deduped():
    llm = FakeLLM(
        sub_queries=["LoRA", " lora ", "RLHF", "LoRA", "InstructGPT", "extra"],
        grades=[True],
    )
    search = FakeSearch({
        "LoRA": [chunk("a", 0.4), chunk("shared", 0.3)],
        "RLHF": [chunk("b", 0.6), chunk("shared", 0.9)],
        "InstructGPT": [chunk("c", 0.5)],
    })

    state = run(llm, search, max_subqueries=3)

    # 大小写/空白不敏感去重，保留顺序，截断到 max_subqueries
    assert state["sub_queries"] == ["LoRA", "RLHF", "InstructGPT"]
    assert search.queries == ["LoRA", "RLHF", "InstructGPT"]
    # 同一chunk被多个子查询命中只保留一份，取最高分，并按分数降序
    ids = [c["id"] for c in state["chunks"]]
    assert ids == ["shared", "b", "c", "a"]
    assert state["chunks"][0]["score"] == 0.9


def test_unusable_plan_output_falls_back_to_original_question():
    llm = FakeLLM(sub_queries=[], grades=[True])
    search = FakeSearch({"What is LoRA?": [chunk("a")]})

    state = run(llm, search)

    assert state["sub_queries"] == ["What is LoRA?"]
    assert state["refused"] is False
