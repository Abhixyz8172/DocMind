# RAG Document Q&A Agent

A document question-answering service built on a **Retrieval-Augmented Generation** pipeline with an **agentic LangGraph workflow**, served as a containerized **FastAPI** microservice.

Upload PDFs / Markdown / text → ask questions → get answers **grounded in your documents, with citations**. If the documents don't contain the answer, the agent says so instead of hallucinating.

## Architecture

```
            ┌───────────── ingestion ─────────────┐
 file ──▶ load ──▶ chunk (paragraph-aware + overlap) ──▶ embed (BGE/GTE) ──▶ ChromaDB
                                                                                  │
            ┌───────────── question answering (LangGraph) ───────────────────────┘
            │
 question ──▶ route ──┬─ DIRECT ───────────────────────────────▶ answer
                      └─ RETRIEVE ─▶ rewrite query ─▶ retrieve top-k ─▶ grade context
                                          ▲                               │
                                          └──── insufficient (retry ≤ 2) ─┤
                                                                          ├─ sufficient ─▶ generate (cited)
                                                                          └─ still no ───▶ "not found" fallback
```

| Concern | Choice |
|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` by default; swap to `bge-large-en-v1.5` / `thenlper/gte-large` via `EMBED_MODEL` |
| Vector store | ChromaDB (persistent, cosine / HNSW) |
| Orchestration | LangGraph state machine (route → rewrite → retrieve → grade → generate / retry / fallback) |
| LLM | OpenAI or Anthropic via `LLM_PROVIDER`, behind a tiny `LLM` interface so adding LLaMA/Ollama is one class |
| API | FastAPI: `/ingest`, `/ask`, `/health` |
| Delivery | Docker + docker-compose, GitHub Actions (tests + image build) |

## Quick start

```bash
cp .env.example .env            # add OPENAI_API_KEY or ANTHROPIC_API_KEY
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Or with Docker: `docker compose up --build`

```bash
# ingest the bundled samples
curl -F "file=@data/sample/kubernetes_basics.md" localhost:8000/ingest
curl -F "file=@data/sample/rag_basics.md"        localhost:8000/ingest

# ask
curl -X POST localhost:8000/ask -H "content-type: application/json" \
     -d '{"question": "How do rolling updates work in Kubernetes?"}'
```

Response includes the `answer`, the retrieved `sources` (with similarity scores) and an agent `trace` showing which graph nodes ran. Interactive docs: http://localhost:8000/docs

## Evaluation

Retrieval quality is measured, not guessed. `eval/run_eval.py` reports **hit-rate@k** and **MRR** over a labelled question set (no API key needed):

```bash
python -m eval.run_eval
EMBED_MODEL=BAAI/bge-large-en-v1.5 python -m eval.run_eval --k 3
python -m eval.run_eval --chunk-size 500 --overlap 100
```

Record your own numbers here after running (and extend `eval/questions.json` with questions over your own documents):

| Embed model | chunk / overlap | k | hit-rate | MRR |
|---|---|---|---|---|
| bge-small-en-v1.5 | 800 / 150 | 4 | _fill in_ | _fill in_ |
| bge-large-en-v1.5 | 800 / 150 | 4 | _fill in_ | _fill in_ |

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

The suite uses a deterministic fake embedder and a scripted fake LLM, so it runs offline in CI. It covers chunking, the vector store, every agent path (answer, retry, fallback, direct) and the HTTP API.

## Project layout

```
app/
  config.py      env-driven settings
  embeddings.py  sentence-transformers wrapper (BGE query prefix handled)
  store.py       ChromaDB wrapper
  ingest.py      loaders + chunking
  llm.py         OpenAI / Anthropic adapters
  agent.py       LangGraph workflow
  main.py        FastAPI app factory
eval/            retrieval evaluation
tests/           pytest suite
```

## Ideas to extend

- Hybrid search (BM25 + vectors) and a cross-encoder reranker
- Streaming responses (SSE), conversation memory
- Fine-tune the embedding model on domain question/passage pairs
- Swap ChromaDB for a managed vector store (e.g. Databricks Vector Search) behind the same `VectorStore` interface
- Deploy the image to AWS ECS / GCP Cloud Run / Kubernetes
