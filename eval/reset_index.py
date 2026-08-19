"""
eval/reset_index.py

清空Pinecone索引里的所有向量，用于清理之前重复上传产生的脏数据。
运行后需要重新执行 ingest_once.py 才能继续测试。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.vector_store import get_index

if __name__ == "__main__":
    confirm = input("确认要清空整个向量库索引吗？这会删除所有已上传的文档数据 (y/n): ")
    if confirm.lower() == "y":
        index = get_index()
        index.delete(delete_all=True)
        print("已清空。请重新运行 eval/ingest_once.py 上传文档。")
    else:
        print("已取消。")
