from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import FakeLLM


def make_client(store, embedder):
    return TestClient(create_app(store=store, embedder=embedder, llm=FakeLLM()))


def test_health(store, embedder):
    r = make_client(store, embedder).get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_ingest_then_ask(store, embedder):
    c = make_client(store, embedder)
    doc = b"Deployments support rolling updates and rollbacks in Kubernetes."
    r = c.post("/ingest", files={"file": ("k8s.txt", doc, "text/plain")})
    assert r.status_code == 200 and r.json()["chunks"] == 1

    r = c.post("/ask", json={"question": "rolling updates in Kubernetes?"})
    body = r.json()
    assert r.status_code == 200
    assert body["answer"] and body["sources"][0]["source"] == "k8s.txt"


def test_ingest_rejects_unsupported_type(store, embedder):
    r = make_client(store, embedder).post("/ingest", files={"file": ("x.exe", b"abc")})
    assert r.status_code == 400


def test_ask_validates_input(store, embedder):
    r = make_client(store, embedder).post("/ask", json={"question": ""})
    assert r.status_code == 422
