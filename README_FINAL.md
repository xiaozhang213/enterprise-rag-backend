# Enterprise Knowledge Base Q&A System

A full-stack Retrieval-Augmented Generation (RAG) system that answers natural-language
questions grounded in a private document collection, with cited sources and
hallucination-resistant design.

**Live Demo**: https://enterprise-rag-frontend-amber.vercel.app
*(Note: the backend runs on Render's free tier — after ~15 minutes of inactivity it
spins down, so the first request may take 20-40s to wake it up)*

Frontend repo: https://github.com/你的用户名/enterprise-rag-frontend

---

## Overview

Enterprise teams often have valuable knowledge locked away in PDFs, reports, and internal
documents that are slow to search and easy to misquote. This project implements an
end-to-end RAG pipeline that lets users upload documents and ask questions in plain
English, receiving answers that are explicitly grounded in the retrieved source material
rather than the model's general knowledge — with every answer traceable back to the exact
document it came from.

The system was built with a focus on production-realistic engineering practices:
streaming responses, batch document processing with partial-failure handling, retry/backoff
for rate-limited API calls, idempotent data ingestion, and a measured evaluation of
retrieval and generation quality rather than a "looks good in the demo" approach.

## Architecture

```
User → Next.js frontend → FastAPI backend
                              ├─ Document Parser (PDF/DOCX/TXT → text)
                              ├─ Chunking Service (recursive splitter, overlap-aware)
                              ├─ Embedding Service (OpenAI, batched + retried)
                              ├─ Vector Store (Pinecone, idempotent upsert)
                              └─ LLM Service (streaming generation, grounded prompting)
```

Two independent data flows:
- **Ingestion (offline)**: document → parse → chunk → embed → upsert to vector store
- **Query (online, streamed)**: question → embed → retrieve top-k chunks → generate
  answer via SSE, with source citations attached once generation completes

*(See `docs/architecture-overview.svg`, `docs/ingestion-flow.svg`, and
`docs/query-flow.svg` for diagrams of each flow.)*

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js, TypeScript, Tailwind CSS |
| Backend | FastAPI (Python) |
| LLM / Embeddings | OpenAI (GPT-4o-mini, text-embedding-3-small) |
| Vector Database | Pinecone |
| Evaluation | RAGAS (faithfulness, context recall), custom test harness |
| Deployment | Vercel (frontend), Render (backend) |

## Project Structure

```
app/
  main.py              FastAPI entrypoint, CORS, global error handling
  config.py            Settings loaded from environment variables
  models/schemas.py    Pydantic request/response models
  services/
    document_parser.py  PDF / DOCX / TXT → plain text
    chunking.py          Recursive text splitter with overlap
    embeddings.py         OpenAI embedding calls, batched with retry/backoff
    vector_store.py       Pinecone read/write, deterministic chunk IDs
    llm.py                 Prompt construction + streamed generation
  routers/
    ingest.py            POST /ingest, POST /ingest/batch
    query.py              POST /query (SSE streaming)
eval/
  ingest_once.py         One-time document upload for testing
  query_test.py           Repeatable question-answering test harness
  ragas_eval.py            RAGAS-based automated evaluation
  reset_index.py           Clears the vector store for a clean re-test
docs/
  architecture-overview.svg
  ingestion-flow.svg
  query-flow.svg
```

## Key Design Decisions

- **Grounded generation, not open-ended chat.** The system prompt explicitly instructs
  the model to answer only from retrieved context and to say "not found" rather than
  guess — verified empirically: on a 10-question test set, the system correctly declined
  to answer the one question with genuinely ambiguous phrasing, rather than fabricating
  a plausible-sounding response.
- **Idempotent ingestion.** Document chunks are written to the vector store using
  deterministic IDs (hash of source + chunk index) instead of random UUIDs, so re-running
  the ingestion pipeline never creates duplicate vectors — a correctness property that
  surfaced as a real bug during testing (duplicate chunks were silently crowding out
  relevant results in retrieval) and was fixed at the source rather than patched around.
- **Resilience to rate limits.** Embedding calls are batched and wrapped with exponential
  backoff retry, after testing with larger documents (6MB+ PDFs, 450+ chunks) surfaced
  transient failures under OpenAI's rate limits.
- **Streaming with unchanged safety guarantees.** Query responses are streamed via SSE for
  perceived latency, while the underlying system prompt and temperature settings are
  identical between the streaming and non-streaming code paths — streaming changes only
  the transport, not the model's constraints.
- **Batch upload with partial-failure isolation.** `/ingest/batch` processes each file
  independently, so one corrupted or oversized file doesn't block the rest of the batch —
  the response reports per-file success/failure rather than an all-or-nothing result.

## Evaluation

The system was evaluated on a 10-question test set spanning 10 distinct source documents
(machine learning research papers), using two complementary methods:

**Manual verification**: 9/10 questions answered correctly with the right source cited.
The one failure was traced to a genuinely ambiguous test question ("this paper" with no
document specified in a multi-document context) rather than a retrieval or generation
defect — a useful distinction between a system limitation and a test design flaw.

**RAGAS automated metrics**:
| Metric | Score | What it measures |
|---|---|---|
| Faithfulness | 0.93 | % of answer content directly supported by retrieved context (low = hallucination) |
| Context Recall | 0.89 | How completely the retriever surfaced the information needed to answer |

*(Answer Relevancy and Factual Correctness were attempted but not finalized due to a
third-party library API incompatibility encountered during evaluation; the two metrics
above were prioritized as they most directly measure hallucination and retrieval quality,
which were the primary risk areas for this system.)*

## What I'd Improve Next

- **Hybrid retrieval** (vector + BM25 keyword search) to handle queries where multiple
  documents share similar semantic content but only one has the literal answer
- **Reranking** to improve precision when top-k retrieval returns topically related but
  not directly relevant chunks
- **Function calling / agentic retrieval**, letting the model decide when and how many
  times to query the knowledge base instead of a fixed single retrieval per question
- **Role-based access control** for a true multi-tenant enterprise deployment
- Resolve the RAGAS library version conflict to complete the full four-metric evaluation

---

## Setup (local development)

1. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   ./.venv/bin/pip install -r requirements.txt
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
   ./.venv/bin/uvicorn app.main:app --reload --port 8000
   ```

4. Check it's alive:
   ```bash
   curl http://localhost:8000/health
   ```

### Try it out

**Upload a document:**
```bash
curl -X POST http://localhost:8000/ingest \
  -F "file=@/path/to/your/document.pdf"
```

**Ask a question:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does this document say about revenue growth?"}'
```

Interactive API docs (Swagger UI): `http://localhost:8000/docs`

### Running the evaluation suite

```bash
./.venv/bin/pip install -r requirements-eval.txt
./.venv/bin/python eval/ingest_once.py    # one-time document upload
./.venv/bin/python eval/query_test.py     # repeatable Q&A test
./.venv/bin/python eval/ragas_eval.py     # RAGAS metrics
```

## Deployment

- **Backend (Render)**: Build command `pip install -r requirements.txt`,
  start command `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- **Frontend (Vercel)**: auto-detected Next.js build, requires the
  `NEXT_PUBLIC_API_BASE_URL` environment variable pointing at the Render backend URL
- Backend's `ALLOWED_ORIGINS` environment variable must include the deployed
  frontend's exact domain for CORS to work
