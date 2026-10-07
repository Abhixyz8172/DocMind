from typing import Protocol

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class HFEmbedder:
    """Sentence-transformers wrapper for BGE / GTE style models (lazy loaded)."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._load().encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        # BGE models are trained with an instruction prefix on the query side.
        if "bge" in self.model_name.lower():
            text = BGE_QUERY_PREFIX + text
        vec = self._load().encode([text], normalize_embeddings=True, show_progress_bar=False)
        return vec[0].tolist()
