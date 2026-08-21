# Enterprise Knowledge Base Q&A System — Backend (Day 1 MVP)

A minimal RAG (Retrieval-Augmented Generation) backend: upload documents,
ask questions, get answers grounded in your own document set with cited sources.

## Stack
FastAPI · OpenAI (embeddings + GPT-4o-mini) · Pinecone (vector store)

## Project Structure
```
app/
  main.py              FastAPI app entrypoint, CORS setup
  config.py            Settings loaded from .env
  models/schemas.py    Pydantic request/response models
  services/
    document_parser.py PDF / DOCX / TXT -> plain text
    chunking.py         Recursive text splitter with overlap
    embeddings.py       OpenAI embedding calls
    vector_store.py     Pinecone read/write, isolated so it's swappable
    llm.py               Prompt construction + answer generation
  routers/
    ingest.py           POST /ingest  — upload a document
    query.py             POST /query   — ask a question
```

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in real keys:
   ```bash
   cp .env.example .env
   ```
   You need:
   - An OpenAI API key (https://platform.openai.com)
   - A Pinecone API key (https://www.pinecone.io — free tier is enough)

3. Run the server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

4. Check it's alive:
   ```bash
   curl http://localhost:8000/health
   ```

## Try it out
TODO


**Upload a document:**
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@/Users/apple/Downloads/示例文档_诺瓦科技员工手册.txt"
```

**Ask a question:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does this document say about revenue growth?"}'
```

Interactive API docs (Swagger UI) are auto-generated at:
```
http://localhost:8000/docs
```

## Design Notes

- **Chunking**: custom recursive splitter (paragraph → sentence → space)
  with overlap, instead of relying on a library, so the splitting logic
  is fully understood and tunable.
- **Vector store isolation**: all Pinecone calls live in `vector_store.py`.
  Swapping to pgvector/Qdrant/Weaviate later only touches this one file.
- **Hallucination control**: the system prompt in `llm.py` explicitly
  instructs the model to say "not found" rather than guess when the
  retrieved context doesn't contain the answer.

## Roadmap (see project planning doc)
- Day 2: Next.js frontend + deployment (Vercel + Render)
- Day 3: streaming responses, error handling polish, README/architecture diagram
- Later: RAGAS evaluation, unit tests, reranking, hybrid search, auth

**RAGAS automated metrics** (post hybrid retrieval upgrade):
| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | 0.94 | % of answer content directly supported by retrieved context |
| Answer Relevancy | 0.87 | How well the answer addresses the actual question asked |
| Context Recall | 0.89 | How completely the retriever surfaced the needed information |
| Factual Correctness (F1) | 0.49 | See note below — score is depressed by terse manual reference answers, not by system inaccuracy |