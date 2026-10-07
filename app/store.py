import chromadb


class VectorStore:
    """Thin wrapper around a ChromaDB collection (cosine similarity)."""

    def __init__(self, path: str | None = None, collection: str = "docs", client=None):
        self.client = client or chromadb.PersistentClient(path=path)
        self.col = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, source: str, chunks: list[str], vectors: list[list[float]]) -> int:
        # Re-ingesting the same file replaces its old chunks.
        self.delete_source(source)
        if not chunks:
            return 0
        self.col.add(
            ids=[f"{source}::{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=vectors,
            metadatas=[{"source": source, "chunk": i} for i in range(len(chunks))],
        )
        return len(chunks)

    def delete_source(self, source: str) -> None:
        self.col.delete(where={"source": source})

    def query(self, vector: list[float], k: int = 4) -> list[dict]:
        n = self.col.count()
        if n == 0:
            return []
        res = self.col.query(query_embeddings=[vector], n_results=min(k, n))
        out = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            out.append(
                {
                    "text": doc,
                    "source": meta["source"],
                    "chunk": meta["chunk"],
                    "score": round(1 - dist, 4),  # cosine distance -> similarity
                }
            )
        return out

    def count(self) -> int:
        return self.col.count()
