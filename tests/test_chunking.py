import pytest

from app.ingest import chunk_text


def test_short_text_single_chunk():
    assert chunk_text("hello world", 100, 10) == ["hello world"]


def test_empty_text():
    assert chunk_text("   \n\n  ", 100, 10) == []


def test_chunks_respect_size_limit():
    text = "\n\n".join(f"Paragraph number {i} " + "word " * 40 for i in range(20))
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_overlap_is_carried_between_chunks():
    text = "\n\n".join(f"unique-marker-{i} " + "filler " * 30 for i in range(10))
    chunks = chunk_text(text, chunk_size=300, overlap=60)
    for a, b in zip(chunks, chunks[1:]):
        assert a[-30:] in b  # tail of previous chunk reappears in the next


def test_long_paragraph_is_split():
    chunks = chunk_text("x" * 2000, chunk_size=500, overlap=100)
    assert len(chunks) >= 4
    assert all(len(c) <= 500 for c in chunks)


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=100, overlap=100)
