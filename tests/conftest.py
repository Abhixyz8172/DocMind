import hashlib
import math
import re
import uuid

import chromadb
import pytest

from app.store import VectorStore


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder. No model download needed."""

    DIM = 256

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.DIM
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.DIM] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    """Scripted LLM: behaviour is chosen by which system prompt it receives."""

    def __init__(self, route="RETRIEVE", grades=("YES",)):
        self.route = route
        self.grades = list(grades)
        self.calls: list[str] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append(system[:20])
        if system.startswith("You are a router"):
            return self.route
        if system.startswith("Rewrite"):
            return user.splitlines()[0]
        if system.startswith("You check retrieval"):
            return self.grades.pop(0) if len(self.grades) > 1 else self.grades[0]
        if system.startswith("Answer the question"):
            return "ANSWER based on context"
        return "Hello!"


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def store():
    return VectorStore(collection=f"t_{uuid.uuid4().hex[:8]}", client=chromadb.EphemeralClient())
