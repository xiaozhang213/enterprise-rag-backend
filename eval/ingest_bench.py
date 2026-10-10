"""
eval/ingest_bench.py

把 benchmark 抽样的论文(eval/sample_docs/pdfs_bench/)和原有10篇论文(eval/sample_docs/pdfs/)
一起入库。和 ingest_once.py 一样走 /ingest/batch 接口，这样正在运行的服务进程里的BM25索引
也会同步更新(直接在脚本里调用service函数的话，服务进程内存里的BM25索引看不到新文档)。

chunk ID 由(文件名, chunk序号)哈希得到，重复运行只会覆盖同一批向量，不会产生重复数据。
前提：服务已启动(uvicorn app.main:app)，且已运行过 eval/build_golden.py。
运行：./.venv/bin/python eval/ingest_bench.py
"""
import os
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
PDF_DIRS = [SCRIPT_DIR / "sample_docs" / "pdfs", SCRIPT_DIR / "sample_docs" / "pdfs_bench"]
BASE_URL = "http://localhost:8000"
BATCH_SIZE = 10  # 和 app/routers/ingest.py 的 MAX_FILES_PER_BATCH 保持一致


def main():
    pdf_paths = [p for d in PDF_DIRS for p in sorted(d.glob("*.pdf"))]
    if not pdf_paths:
        raise RuntimeError(f"在 {[str(d) for d in PDF_DIRS]} 没找到PDF，请先运行 eval/build_golden.py")

    headers = {"X-API-Key": os.environ["API_KEY"]} if os.environ.get("API_KEY") else {}
    print(f"检测到 {len(pdf_paths)} 份PDF，分 {-(-len(pdf_paths) // BATCH_SIZE)} 批上传...")

    total_chunks = 0
    failed = []
    for start in range(0, len(pdf_paths), BATCH_SIZE):
        batch = pdf_paths[start : start + BATCH_SIZE]
        handles = [open(p, "rb") for p in batch]
        try:
            files = [("files", (p.name, h, "application/pdf")) for p, h in zip(batch, handles)]
            resp = requests.post(f"{BASE_URL}/ingest/batch", files=files, headers=headers, timeout=600)
            resp.raise_for_status()
        finally:
            for h in handles:
                h.close()

        for r in resp.json()["results"]:
            if r["status"] == "success":
                total_chunks += r["chunks_created"]
                print(f"  ✓ {r['filename']} ({r['chunks_created']} 个片段)")
            else:
                failed.append(r["filename"])
                print(f"  ✗ {r['filename']} — {r['error']}")

    print(f"\n完成：成功 {len(pdf_paths) - len(failed)}/{len(pdf_paths)} 份，共 {total_chunks} 个片段")
    if failed:
        print(f"失败文件(可直接重跑本脚本重试，已成功的会被幂等覆盖): {failed}")


if __name__ == "__main__":
    main()
