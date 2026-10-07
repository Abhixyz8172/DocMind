from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .agent import ask, build_graph
from .config import get_settings
from .embeddings import Embedder, HFEmbedder
from .ingest import ingest_text, load_text
from .llm import LLM, get_llm
from .store import VectorStore


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def create_app(
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
    llm: LLM | None = None,
) -> FastAPI:
    """App factory. Components are injectable (tests) and otherwise built lazily."""
    settings = get_settings()
    app = FastAPI(title="RAG Document Q&A Agent", version="1.0.0")
    state: dict = {"store": store, "embedder": embedder, "llm": llm, "graph": None}

    def get_store() -> VectorStore:
        if state["store"] is None:
            state["store"] = VectorStore(settings.chroma_dir, settings.collection)
        return state["store"]

    def get_embedder() -> Embedder:
        if state["embedder"] is None:
            state["embedder"] = HFEmbedder(settings.embed_model)
        return state["embedder"]

    def get_graph():
        if state["graph"] is None:
            if state["llm"] is None:
                state["llm"] = get_llm(settings)
            state["graph"] = build_graph(get_store(), get_embedder(), state["llm"], settings.top_k)
        return state["graph"]

    @app.get("/health")
    def health():
        return {"status": "ok", "chunks": get_store().count(), "embed_model": settings.embed_model}

    @app.post("/ingest")
    async def ingest(file: UploadFile = File(...)):
        data = await file.read()
        try:
            text = load_text(file.filename or "upload.txt", data)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not text.strip():
            raise HTTPException(status_code=400, detail="No extractable text in file")
        n = ingest_text(
            file.filename, text, get_store(), get_embedder(),
            settings.chunk_size, settings.chunk_overlap,
        )
        return {"source": file.filename, "chunks": n}

    @app.post("/ask")
    def ask_endpoint(req: AskRequest):
        try:
            graph = get_graph()
        except RuntimeError as e:  # e.g. missing API key
            raise HTTPException(status_code=503, detail=str(e))
        return ask(graph, req.question)

    return app


app = create_app()
