"""
app/security.py

API Key鉴权，只用于写操作(文档上传)。设计上留了一个"关闭开关"：
如果没在环境变量里配置 API_KEY，鉴权直接跳过——这样本地开发不用每次
带着key麻烦，只有生产环境显式设置了key才会真正生效。
"""
from fastapi import Header, HTTPException, status

from app.config import settings


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if settings.api_key is None:
        return  # 未配置key，视为鉴权关闭(本地开发场景)

    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )