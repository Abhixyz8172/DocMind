import io
import re

from .embeddings import Embedder
from .store import VectorStore


def load_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((p.extract_text() or "") for p in reader.pages)
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {filename} (use .pdf, .txt or .md)")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Paragraph-aware chunking with character overlap.

    Paragraphs are packed into chunks up to `chunk_size`. A paragraph longer than
    chunk_size is split with a sliding window. The tail of each chunk is repeated
    at the start of the next one so context isn't lost at chunk boundaries.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    for p in paragraphs:
        if len(p) <= chunk_size:
            pieces.append(p)
        else:
            step = chunk_size - overlap
            pieces.extend(p[i : i + chunk_size] for i in range(0, len(p), step))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif len(current) + 2 + len(piece) <= chunk_size:
            current = f"{current}\n\n{piece}"
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{piece}" if tail else piece
            if len(current) > chunk_size:  # keep the hard size bound
                current = current[-chunk_size:]
    if current:
        chunks.append(current)
    return chunks


def ingest_text(
    source: str,
    text: str,
    store: VectorStore,
    embedder: Embedder,
    chunk_size: int = 800,
    overlap: int = 150,
) -> int:
    chunks = chunk_text(text, chunk_size, overlap)
    vectors = embedder.embed_documents(chunks) if chunks else []
    return store.add(source, chunks, vectors)
