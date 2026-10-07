"""Retrieval evaluation: hit-rate@k and MRR over a small labelled question set.

Needs no LLM / API key — it measures the retriever only (embedder + chunking + store).

    python -m eval.run_eval                       # defaults from env / .env settings
    EMBED_MODEL=BAAI/bge-large-en-v1.5 python -m eval.run_eval
    python -m eval.run_eval --k 3 --chunk-size 500
"""
import argparse
import json
import uuid
from pathlib import Path

import chromadb

from app.config import get_settings
from app.embeddings import HFEmbedder
from app.ingest import ingest_text, load_text
from app.store import VectorStore

ROOT = Path(__file__).resolve().parent.parent


def evaluate(embedder, k: int, chunk_size: int, overlap: int, data_dir: Path, questions: list[dict]):
    store = VectorStore(
        collection=f"eval_{uuid.uuid4().hex[:8]}", client=chromadb.EphemeralClient()
    )
    for f in sorted(data_dir.glob("*")):
        if f.suffix.lower() in {".md", ".txt", ".pdf"}:
            ingest_text(f.name, load_text(f.name, f.read_bytes()), store, embedder, chunk_size, overlap)

    hits, rr_total = 0, 0.0
    for q in questions:
        results = store.query(embedder.embed_query(q["question"]), k=k)
        rank = next(
            (i + 1 for i, r in enumerate(results) if r["source"] == q["expected_source"]), None
        )
        if rank:
            hits += 1
            rr_total += 1 / rank
    n = len(questions)
    return {"questions": n, "k": k, "hit_rate": hits / n, "mrr": rr_total / n}


def main():
    s = get_settings()
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=s.top_k)
    p.add_argument("--chunk-size", type=int, default=s.chunk_size)
    p.add_argument("--overlap", type=int, default=s.chunk_overlap)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "sample")
    p.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.json")
    a = p.parse_args()

    questions = json.loads(a.questions.read_text())
    res = evaluate(HFEmbedder(s.embed_model), a.k, a.chunk_size, a.overlap, a.data, questions)
    print(f"model={s.embed_model} chunk={a.chunk_size} overlap={a.overlap}")
    print(f"hit-rate@{res['k']} = {res['hit_rate']:.2%}   MRR = {res['mrr']:.3f}   (n={res['questions']})")


if __name__ == "__main__":
    main()
