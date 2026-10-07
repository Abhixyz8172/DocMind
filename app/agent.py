"""LangGraph agent: route -> rewrite -> retrieve -> grade -> (generate | retry | fallback).

The agent decides on its own whether retrieval is needed, rewrites the question
into a search query, checks whether the retrieved context is actually sufficient,
and retries with a new query if not. If it still can't find support it says so
instead of hallucinating.
"""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .embeddings import Embedder
from .llm import LLM
from .store import VectorStore

MAX_ATTEMPTS = 2

ROUTE_SYS = (
    "You are a router. Reply with exactly one word.\n"
    "RETRIEVE: the question needs facts from the user's uploaded documents.\n"
    "DIRECT: greeting, small talk, or a question about you that needs no documents."
)
REWRITE_SYS = (
    "Rewrite the user's question into a short, keyword-rich search query for a "
    "semantic search over documents. Output only the query, nothing else."
)
GRADE_SYS = (
    "You check retrieval quality. Given a question and passages, reply with exactly "
    "one word: YES if the passages contain enough information to answer the question, "
    "otherwise NO."
)
ANSWER_SYS = (
    "Answer the question using ONLY the provided context. Cite the passages you used "
    "inline like [source#chunk]. If the context is insufficient, say you don't know. "
    "Be concise."
)
DIRECT_SYS = "You are a helpful document Q&A assistant. Reply briefly."


class AgentState(TypedDict, total=False):
    question: str
    query: str
    route: str
    docs: list[dict]
    sufficient: bool
    attempts: int
    answer: str
    trace: list[str]


def _fmt_docs(docs: list[dict]) -> str:
    return "\n\n".join(f"[{d['source']}#{d['chunk']}]\n{d['text']}" for d in docs)


def build_graph(store: VectorStore, embedder: Embedder, llm: LLM, k: int = 4):
    def log(state: AgentState, msg: str) -> list[str]:
        return state.get("trace", []) + [msg]

    def route(state: AgentState):
        out = llm.complete(ROUTE_SYS, state["question"]).strip().upper()
        decision = "direct" if out.startswith("DIRECT") else "retrieve"
        return {"route": decision, "attempts": 0, "trace": log(state, f"route={decision}")}

    def rewrite(state: AgentState):
        user = state["question"]
        if state.get("query"):  # retry: tell the model the last query failed
            user += f"\n\nPrevious query that failed to find an answer: {state['query']}\nTry different wording."
        query = llm.complete(REWRITE_SYS, user).strip() or state["question"]
        return {"query": query, "trace": log(state, f"query={query!r}")}

    def retrieve(state: AgentState):
        docs = store.query(embedder.embed_query(state["query"]), k=k)
        return {"docs": docs, "trace": log(state, f"retrieved={len(docs)}")}

    def grade(state: AgentState):
        docs = state.get("docs", [])
        attempts = state.get("attempts", 0) + 1
        if not docs:
            ok = False
        else:
            verdict = llm.complete(
                GRADE_SYS, f"Question: {state['question']}\n\nPassages:\n{_fmt_docs(docs)}"
            )
            ok = verdict.strip().upper().startswith("YES")
        return {
            "sufficient": ok,
            "attempts": attempts,
            "trace": log(state, f"grade={'sufficient' if ok else 'insufficient'} (attempt {attempts})"),
        }

    def after_route(state: AgentState):
        return "direct" if state["route"] == "direct" else "rewrite"

    def after_grade(state: AgentState):
        if state["sufficient"]:
            return "generate"
        return "rewrite" if state["attempts"] < MAX_ATTEMPTS else "fallback"

    def generate(state: AgentState):
        answer = llm.complete(
            ANSWER_SYS,
            f"Context:\n{_fmt_docs(state['docs'])}\n\nQuestion: {state['question']}",
        )
        return {"answer": answer.strip(), "trace": log(state, "generated")}

    def fallback(state: AgentState):
        return {
            "answer": "I couldn't find this in the ingested documents.",
            "docs": [],
            "trace": log(state, "fallback"),
        }

    def direct(state: AgentState):
        return {
            "answer": llm.complete(DIRECT_SYS, state["question"]).strip(),
            "docs": [],
            "trace": log(state, "direct answer"),
        }

    g = StateGraph(AgentState)
    for name, fn in [
        ("route", route),
        ("rewrite", rewrite),
        ("retrieve", retrieve),
        ("grade", grade),
        ("generate", generate),
        ("fallback", fallback),
        ("direct", direct),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "route")
    g.add_conditional_edges("route", after_route, {"direct": "direct", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade", after_grade, {"generate": "generate", "rewrite": "rewrite", "fallback": "fallback"}
    )
    for end_node in ("generate", "fallback", "direct"):
        g.add_edge(end_node, END)
    return g.compile()


def ask(graph, question: str) -> dict:
    final = graph.invoke({"question": question, "trace": []})
    return {
        "answer": final["answer"],
        "sources": [
            {"source": d["source"], "chunk": d["chunk"], "score": d["score"], "text": d["text"]}
            for d in final.get("docs", [])
        ],
        "trace": final.get("trace", []),
    }
