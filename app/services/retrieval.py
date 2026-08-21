"""
app/services/retrieval.py

混合检索：把向量检索(语义)和BM25检索(关键词)的结果加权融合成一个排序列表。

为什么要混合：向量检索容易漏掉依赖精确术语的查询(比如"RLHF"这种缩写，
如果chunk的整体语义embedding没有把这个词的权重体现出来)，BM25靠字面
匹配能补上这类情况。两者按权重结合，覆盖的失败场景比单用一种更广。

为什么不用加权分数相加：向量相似度(cosine, 0-1)和BM25分数不是同一量纲，
直接加权求和会让"只被一种方法命中"的chunk被系统性地打压(哪怕它本来排第一)，
导致真正相关的结果被挤出候选池。RRF只看排名位次、不看原始分数大小，
避免了这种尺度不匹配问题，是搜索领域融合多路召回的标准做法。
"""
from app.config import settings
from app.services import keyword_index, vector_store

from app.services import keyword_index, vector_store

RRF_K = 60  # 业界常用平滑常数，值越大，靠后排名的贡献越被压低

def hybrid_search(query_vector: list[float], query_text: str, top_k: int) -> list[dict]:
    vector_matches = vector_store.query_similar(query_vector, top_k=top_k * 2)
    keyword_matches = keyword_index.search(query_text, top_k=top_k * 2)

    rrf_scores: dict[str, float] = {}
    metadata: dict[str, dict] = {}

    for rank, m in enumerate(vector_matches, start=1):
        rrf_scores[m["id"]] = rrf_scores.get(m["id"], 0.0) + 1 / (RRF_K + rank)
        metadata[m["id"]] = m

    for rank, m in enumerate(keyword_matches, start=1):
        rrf_scores[m["id"]] = rrf_scores.get(m["id"], 0.0) + 1 / (RRF_K + rank)
        metadata.setdefault(m["id"], m)

    ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

    # 不做"当次结果内部"的min-max归一化(会把排名误伪装成绝对置信度)。
    # RRF理论最高分是 vector_weight和keyword_weight两路都命中rank=1时的
    # 1/(RRF_K+1) + 1/(RRF_K+1)，用这个理论上限做归一化，分数在不同次查询之间可比。
    theoretical_max = 2 / (RRF_K + 1)

    results = []
    for cid in ranked_ids:
        m = metadata[cid]
        normalized = min(rrf_scores[cid] / theoretical_max, 1.0)
        results.append({**m, "score": normalized})
    return results
    # ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

    # # RRF原始分数很小(约0.01-0.03量级)，不适合直接当"相似度百分比"展示给用户，
    # # 这里做min-max归一化，只影响展示，不影响排序结果本身
    # raw_scores = [rrf_scores[cid] for cid in ranked_ids]
    # lo, hi = min(raw_scores, default=0), max(raw_scores, default=1)
    # span = (hi - lo) or 1.0

    # results = []
    # for cid in ranked_ids:
    #     m = metadata[cid]
    #     normalized = (rrf_scores[cid] - lo) / span
    #     results.append({**m, "score": normalized})
    # return results
# def hybrid_search(query_vector: list[float], query_text: str, top_k: int) -> list[dict]:
#     vector_matches = vector_store.query_similar(query_vector, top_k=top_k * 2)
#     keyword_matches = keyword_index.search(query_text, top_k=top_k * 2)

#     combined: dict[str, dict] = {}

#     for m in vector_matches:
#         combined[m["id"]] = {
#             "id": m["id"], "content": m["content"], "source": m["source"],
#             "vector_score": m["score"], "keyword_score": 0.0,
#         }

#     for m in keyword_matches:
#         if m["id"] in combined:
#             combined[m["id"]]["keyword_score"] = m["score"]
#         else:
#             combined[m["id"]] = {
#                 "id": m["id"], "content": m["content"], "source": m["source"],
#                 "vector_score": 0.0, "keyword_score": m["score"],
#             }

#     for entry in combined.values():
#         entry["score"] = (
#             settings.vector_weight * entry["vector_score"]
#             + settings.keyword_weight * entry["keyword_score"]
#         )

#     return sorted(combined.values(), key=lambda x: x["score"], reverse=True)[:top_k]