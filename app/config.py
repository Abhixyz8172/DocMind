import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    openai_model: str
    anthropic_model: str
    embed_model: str
    chroma_dir: str
    collection: str
    chunk_size: int
    chunk_overlap: int
    top_k: int


def get_settings() -> Settings:
    """Read settings from environment at call time (keeps tests easy)."""
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "openai").lower(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        embed_model=os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        chroma_dir=os.getenv("CHROMA_DIR", "./chroma_db"),
        collection=os.getenv("COLLECTION", "docs"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "800")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        top_k=int(os.getenv("TOP_K", "4")),
    )
