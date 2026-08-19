"""
应用入口。
"""
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.routers import ingest, query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("enterprise_rag")

app = FastAPI(
    title="Enterprise Knowledge Base Q&A System",
    description="A RAG-based Q&A system over enterprise documents.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """兜底异常处理:任何没被主动catch的报错,都统一记日志+返回友好提示,而不是把堆栈信息暴露给用户"""
    logger.exception("Unhandled error on %s %s", request.method, request.url)
    return JSONResponse(status_code=500, content={"detail": "服务器内部错误，请稍后重试"})


app.include_router(ingest.router)
app.include_router(query.router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}