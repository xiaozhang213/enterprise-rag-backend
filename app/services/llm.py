"""
LLM 生成服务。
核心设计点（面试常问）：
1. 明确要求模型"只根据提供的上下文回答"，减少幻觉
2. 找不到相关信息时，要求模型明确说"未找到"，而不是编答案
3. Prompt 和业务逻辑分离，方便后续做 Prompt 迭代/AB测试
"""
from openai import OpenAI

from app.config import settings

_client = OpenAI(api_key=settings.openai_api_key)

SYSTEM_PROMPT = """你是一个企业知识库问答助手。你必须严格根据下面提供的"参考资料"来回答用户问题。

规则：
1. 只使用参考资料中的信息回答，不要使用你自己的知识编造内容。
2. 如果参考资料中没有足够信息回答问题，明确回答"根据现有资料，未找到相关信息"，不要猜测。
3. 回答要简洁、准确，可以适当引用参考资料中的关键信息。
"""


def build_context(chunks: list[dict]) -> str:
    """把检索到的chunk拼接成LLM可读的上下文，每段标注来源方便后续引用展示"""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f"[资料{i} | 来源: {chunk['source']}]\n{chunk['content']}")
    return "\n\n".join(parts)


def generate_answer(question: str, chunks: list[dict]) -> str:
    context = build_context(chunks)

    user_message = f"""参考资料：
{context}

用户问题：{question}
"""

    response = _client.chat.completions.create(
        model=settings.openai_chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,  # 问答场景要低temperature，减少发挥空间
    )
    return response.choices[0].message.content
