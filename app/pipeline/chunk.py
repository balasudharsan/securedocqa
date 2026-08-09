"""Chunking for the e5 embedding model.

Chunks feed intfloat/multilingual-e5-small, which has a 512-token context
window, so word-windows are kept well under it. e5 also REQUIRES asymmetric
prefixes ("passage: " for stored text, "query: " for questions); Qdrant Cloud
Inference does not add them, so we do — from a single place so the two sides
can never drift.
"""

from __future__ import annotations


def chunk_text(
    pages: list[tuple[int, str]],
    max_words: int = 300,
    overlap: int = 40,
) -> list[dict]:
    """Split each page into overlapping word-windows tagged with its page.

    Overlap keeps a sentence that straddles a window boundary intact in at
    least one chunk. Stored text is CLEAN (no "passage: " prefix): the prefix
    is applied only at embedding time, so citations and display show the real
    text. stride is floored at 1 so a bad overlap can't cause an infinite loop.
    """
    stride = max(1, max_words - overlap)
    chunks: list[dict] = []
    for page_number, text in pages:
        words = text.split()
        for start in range(0, len(words), stride):
            window = words[start : start + max_words]
            chunk = " ".join(window).strip()
            if chunk:
                chunks.append({"page": page_number, "text": chunk})
            if start + max_words >= len(words):
                break
    return chunks


def to_passage(chunk_text: str) -> str:
    """Prefix stored text for e5. Kept in one place to match to_query()."""
    return "passage: " + chunk_text


def to_query(question: str) -> str:
    """Prefix a question for e5. The query-side match to to_passage()."""
    return "query: " + question
