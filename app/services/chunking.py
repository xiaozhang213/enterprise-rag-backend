"""
文本切分（Chunking）服务。

没有直接依赖 LangChain 的 TextSplitter，而是自己实现一个简化版
"递归字符切分"——这是有意的选择：
1. 减少依赖，避免版本冲突
2. 面试时能完整讲清楚切分逻辑，而不是"我调了个库"

策略：优先按段落/句子边界切分，只有边界找不到时才硬切，
并保留 overlap，避免答案跨越两个chunk边界时上下文丢失。
"""
from typing import Optional

from app.config import settings

# 按优先级尝试的切分符：先按段落，再按句子，最后按空格硬切
SEPARATORS = ["\n\n", "\n", "。", ". ", " "]


def _split_by_separator(text: str, separator: str) -> list[str]:
    if separator == "":
        return list(text)
    return text.split(separator)


def recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """递归尝试用不同分隔符切分，直到每块都小于等于 chunk_size"""
    if len(text) <= chunk_size or not separators:
        return [text] if text.strip() else []

    separator, remaining_separators = separators[0], separators[1:]
    pieces = _split_by_separator(text, separator)

    chunks: list[str] = []
    buffer = ""
    for piece in pieces:
        candidate = (buffer + separator + piece) if buffer else piece
        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            if buffer:
                chunks.append(buffer)
            if len(piece) > chunk_size:
                # 这一段本身还是太长，递归用下一级分隔符继续切
                chunks.extend(recursive_split(piece, chunk_size, remaining_separators))
                buffer = ""
            else:
                buffer = piece
    if buffer:
        chunks.append(buffer)
    return chunks


def add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """给相邻chunk之间加上重叠部分，避免语义在切分边界被截断"""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    overlapped = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        overlapped.append(prev_tail + chunks[i])
    return overlapped


def chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> list[str]:
    """对外暴露的主入口"""
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap if overlap is not None else settings.chunk_overlap

    text = text.strip()
    if not text:
        return []

    raw_chunks = recursive_split(text, chunk_size, SEPARATORS)
    raw_chunks = [c.strip() for c in raw_chunks if c.strip()]
    return add_overlap(raw_chunks, overlap)
