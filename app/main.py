"""
应用入口。
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import ingest, query

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

app.include_router(ingest.router)
app.include_router(query.router)


@app.get("/health")
async def health_check():
    return {"status": "ok"}
