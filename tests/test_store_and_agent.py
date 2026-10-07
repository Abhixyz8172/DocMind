from app.agent import ask, build_graph
from app.ingest import ingest_text
from tests.conftest import FakeLLM

DOC_K8S = "A Pod is the smallest deployable unit in Kubernetes. Deployments manage Pods and support rolling updates."
DOC_RAG = "Chunking splits documents into passages. Embeddings turn text into vectors stored in ChromaDB."


def _load(store, embedder):
    ingest_text("k8s.md", DOC_K8S, store, embedder)
    ingest_text("rag.md", DOC_RAG, store, embedder)


def test_retrieval_returns_relevant_source_first(store, embedder):
    _load(store, embedder)
    res = store.query(embedder.embed_query("rolling updates deployments"), k=2)
    assert res[0]["source"] == "k8s.md"


def test_reingest_replaces_old_chunks(store, embedder):
    ingest_text("a.md", "first version", store, embedder)
    ingest_text("a.md", "second version", store, embedder)
    assert store.count() == 1


def test_empty_store_query_returns_empty(store, embedder):
    assert store.query(embedder.embed_query("anything")) == []


def test_agent_retrieve_and_answer(store, embedder):
    _load(store, embedder)
    graph = build_graph(store, embedder, FakeLLM(), k=2)
    out = ask(graph, "What are rolling updates in deployments?")
    assert out["answer"] == "ANSWER based on context"
    assert out["sources"][0]["source"] == "k8s.md"
    assert out["trace"][0] == "route=retrieve"


def test_agent_retries_then_succeeds(store, embedder):
    _load(store, embedder)
    llm = FakeLLM(grades=("NO", "YES"))
    out = ask(build_graph(store, embedder, llm, k=2), "deployments rolling updates")
    assert out["answer"] == "ANSWER based on context"
    assert sum("query=" in t for t in out["trace"]) == 2  # rewrote twice


def test_agent_falls_back_when_context_never_sufficient(store, embedder):
    _load(store, embedder)
    out = ask(build_graph(store, embedder, FakeLLM(grades=("NO",)), k=2), "something unrelated")
    assert "couldn't find" in out["answer"]
    assert out["sources"] == []
    assert out["trace"][-1] == "fallback"


def test_agent_fallback_on_empty_store(store, embedder):
    out = ask(build_graph(store, embedder, FakeLLM(), k=2), "anything")
    assert "couldn't find" in out["answer"]


def test_agent_direct_route_skips_retrieval(store, embedder):
    out = ask(build_graph(store, embedder, FakeLLM(route="DIRECT"), k=2), "hi there")
    assert out["answer"] == "Hello!"
    assert out["trace"] == ["route=direct", "direct answer"]
