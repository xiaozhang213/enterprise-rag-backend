"""
文档解析服务。
职责单一：把不同格式的文件变成纯文本，不做任何切分/向量化逻辑。
"""
import fitz  # PyMuPDF
import docx


def parse_pdf(file_bytes: bytes) -> str:
    """用 PyMuPDF 提取PDF全文文本"""
    text_parts = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def parse_docx(file_bytes: bytes) -> str:
    """用 python-docx 提取Word文档文本"""
    import io

    document = docx.Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in document.paragraphs)


def parse_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")


def parse_document(filename: str, file_bytes: bytes) -> str:
    """
    根据文件后缀分发到对应解析器。
    后续要支持更多格式（ppt/html等），只需要在这里加一个分支——
    这是面试常问的"如何设计可扩展的解析层"的答案。
    """
    lower_name = filename.lower()
    if lower_name.endswith(".pdf"):
        return parse_pdf(file_bytes)
    elif lower_name.endswith(".docx"):
        return parse_docx(file_bytes)
    elif lower_name.endswith(".txt"):
        return parse_txt(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {filename}")
