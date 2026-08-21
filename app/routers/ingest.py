"""
/ingest 路由：接收文档上传，完成"解析 -> 切分 -> embedding -> 入库"全流程。
"""
import logging

from fastapi import APIRouter, UploadFile
from pydantic import BaseModel

from app.models.schemas import IngestResponse
from app.services import chunking, document_parser, embeddings, vector_store

from typing import Optional

from app.services import keyword_index

router = APIRouter(prefix="/ingest", tags=["ingest"])
logger = logging.getLogger("enterprise_rag.ingest")

MAX_FILES_PER_BATCH = 10  # 防止demo环境被一次性传爆


class BatchIngestResult(BaseModel):
    filename: str
    status: str
    chunks_created: int = 0
    error: Optional[str] = None
# class BatchIngestResult(BaseModel):
#     filename: str
#     status: str          # "success" 或 "failed"
#     chunks_created: int = 0
#     error: str | None = None


class BatchIngestResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: list[BatchIngestResult]


def _ingest_single_file(filename: str, file_bytes: bytes) -> BatchIngestResult:
    """单个文件的处理逻辑抽成独立函数，方便单个上传和批量上传复用"""
    if not filename:
        return BatchIngestResult(filename="(未命名)", status="failed", error="未提供文件名")
    if not file_bytes:
        return BatchIngestResult(filename=filename, status="failed", error="文件内容为空")

    try:
        raw_text = document_parser.parse_document(filename, file_bytes)
    except ValueError as e:
        return BatchIngestResult(filename=filename, status="failed", error=str(e))
    except Exception:
        logger.exception("解析文件失败: %s", filename)
        return BatchIngestResult(filename=filename, status="failed", error="文件解析失败，请确认文件未损坏")

    if not raw_text.strip():
        return BatchIngestResult(filename=filename, status="failed", error="文档解析结果为空")

    chunks = chunking.chunk_text(raw_text)
    if not chunks:
        return BatchIngestResult(filename=filename, status="failed", error="切分后没有有效内容")

    try:
        chunk_ids = vector_store.generate_chunk_ids(chunks, filename)
        vectors = embeddings.embed_texts(chunks)
        count = vector_store.upsert_chunks(chunks, vectors, source=filename)
        keyword_index.add_documents(chunk_ids, chunks, filename) 
    except Exception:
        logger.exception("向量化或入库失败: %s", filename)
        return BatchIngestResult(filename=filename, status="failed", error="向量库服务暂时不可用")

    logger.info("已入库文档 %s，共 %d 个片段", filename, count)
    return BatchIngestResult(filename=filename, status="success", chunks_created=count)


@router.post("", response_model=IngestResponse)
async def ingest_document(file: UploadFile) -> IngestResponse:
    """保留单文件接口，向后兼容"""
    file_bytes = await file.read()
    result = _ingest_single_file(file.filename, file_bytes)
    if result.status == "failed":
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=result.error)
    return IngestResponse(filename=result.filename, chunks_created=result.chunks_created)


@router.post("/batch", response_model=BatchIngestResponse)
async def ingest_documents_batch(files: list[UploadFile]) -> BatchIngestResponse:
    """批量上传接口:部分失败不影响其他文件继续处理"""
    files = files[:MAX_FILES_PER_BATCH]

    results = []
    for file in files:
        file_bytes = await file.read()
        results.append(_ingest_single_file(file.filename, file_bytes))

    succeeded = sum(1 for r in results if r.status == "success")
    return BatchIngestResponse(
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        results=results,
    )