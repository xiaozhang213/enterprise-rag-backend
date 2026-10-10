"""
LLM 生成服务。
核心设计点（面试常问）：
1. 明确要求模型"只根据提供的上下文回答"，减少幻觉
2. 找不到相关信息时，要求模型明确说"未找到"，而不是编答案
3. Prompt 和业务逻辑分离，方便后续做 Prompt 迭代/AB测试
4. 多轮追问先做 query condensation：把"那它多少钱？"这类依赖上下文的问题
   改写成独立问题再去检索，否则向量/BM25检索会因为缺主语而检不到东西
"""
from typing import Optional

from openai import OpenAI

from app.config import settings

_client = OpenAI(api_key=settings.openai_api_key)

SYSTEM_PROMPT = """You are an enterprise knowledge base Q&A assistant. You must answer user questions strictly based on the "reference materials" provided below.

Rules:
1. Use only the information from the reference materials; do not use your own knowledge or invent content.
2. If the reference materials do not contain sufficient information to answer the question, explicitly respond with "According to the available information, no relevant information was found" and do not guess.
3. Keep answers concise and accurate, and you may briefly quote key information from the reference materials.
"""

CONDENSE_PROMPT = """Based on the conversation history below, rewrite the user's latest question into a standalone, complete question that does not depend on the conversation context and can be understood independently.
Output only the rewritten question itself without any explanation. If the latest question is already a standalone complete question and does not need rewriting, return it unchanged.

Conversation history:
{history}

Latest question: {question}
"""


def build_context(chunks: list[dict]) -> str:
    """把检索到的chunk拼接成LLM可读的上下文，每段标注来源方便后续引用展示"""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[资料{i} | 来源: {chunk['source']}]\n{chunk['content']}")
    return "\n\n".join(parts)


def condense_question(question: str, history: list[dict]) -> str:
    """
    把追问改写成独立问题，用于检索。没有历史时直接跳过，省一次LLM调用。
    """
    if not history:
        return question

    history_text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)
    user_message = CONDENSE_PROMPT.format(history=history_text, question=question)

    response = _client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[{"role": "user", "content": user_message}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def generate_answer_stream(question: str, chunks: list[dict], history: Optional[list[dict]] = None):
    """
    流式版本：用生成器逐块yield文本，而不是等模型说完整段话再返回。
    history: 此前的对话轮次(role/content)，拼在参考资料之前，让回答保持多轮语境连贯。
    """
    context = build_context(chunks)
    user_message = f"""参考资料：
{context}

用户问题：{question}
"""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history or []:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    stream = _client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=messages,
        temperature=0.2,
        stream=True,  # 关键参数：开启流式
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
# def generate_answer(question: str, chunks: list[dict]) -> str:
#     context = build_context(chunks)

#     user_message = f"""参考资料：
# {context}

# 用户问题：{question}
# """

#     response = _client.chat.completions.create(
#         model=settings.openai_chat_model,
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": user_message},
#         ],
#         temperature=0.2,  # 问答场景要低temperature，减少发挥空间
#     )
#     return response.choices[0].message.content
