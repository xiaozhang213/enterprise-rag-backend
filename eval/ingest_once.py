"""
eval/ingest_once.py

只负责上传文档，只在第一次调试、或者文档集有变化时运行一次。
之后调试提问逻辑，直接用 eval/query_test.py，不要重复跑这个文件。
"""
from pathlib import Path

from pinecone.core.openapi.shared.exceptions import NotFoundException

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
PDF_DIR = SCRIPT_DIR / "sample_docs" / "pdfs"
BASE_URL = "http://localhost:8000"


def main():
    pdf_paths = list(PDF_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError(f"在 {PDF_DIR} 没找到PDF")

    print(f"检测到 {len(pdf_paths)} 份PDF，开始上传...")
    files = [("files", (p.name, open(p, "rb"), "application/pdf")) for p in pdf_paths]

    resp = requests.post(f"{BASE_URL}/ingest/batch", files=files, timeout=300)
    resp.raise_for_status()
    result = resp.json()

    print(f"上传结果: 成功 {result['succeeded']}/{result['total']}")
    for r in result["results"]:
        mark = "✓" if r["status"] == "success" else "✗"
        print(f"  {mark} {r['filename']}", end="")
        print(f" ({r['chunks_created']} 个片段)" if r["status"] == "success" else f" — {r['error']}")

    print("\n上传完成。以后调试直接跑 eval/query_test.py，不要重复运行这个文件。")


if __name__ == "__main__":
    main()
    confirm = input("确认要清空整个向量库索引吗？这会删除所有已上传的文档数据 (y/n): ")
    if confirm.lower() == "y":
        index = get_index()
        try:
            index.delete(delete_all=True)
            print("已清空。请重新运行 eval/ingest_once.py 上传文档。")
        except NotFoundException:
            print("索引本来就是空的，无需清空，可以直接运行 eval/ingest_once.py。")
    else:
        print("已取消。")