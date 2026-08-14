"""
/ingest 路由：接收文档上传，完成"解析 -> 切分 -> embedding -> 入库"全流程。
"""
from fastapi import APIRouter, HTTPException, UploadFile

from app.models.schemas import IngestResponse
from app.services import chunking, document_parser, embeddings, vector_store

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse)
async def ingest_document(file: UploadFile) -> IngestResponse:
    file_bytes = await file.read()

    try:
        raw_text = document_parser.parse_document(file.filename, file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="文档解析结果为空，请检查文件内容")

    chunks = chunking.chunk_text(raw_text)
    if not chunks:
        raise HTTPException(status_code=400, detail="文档切分后没有有效内容")

    vectors = embeddings.embed_texts(chunks)
    count = vector_store.upsert_chunks(chunks, vectors, source=file.filename)

    return IngestResponse(filename=file.filename, chunks_created=count)
