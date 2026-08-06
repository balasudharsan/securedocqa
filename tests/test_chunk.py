from app.pipeline.chunk import chunk_text, to_passage, to_query


def test_page_numbers_preserved_across_pages():
    pages = [(1, "alpha beta gamma"), (2, "delta epsilon")]
    chunks = chunk_text(pages, max_words=10, overlap=2)
    assert [c["page"] for c in chunks] == [1, 2]
    assert chunks[0]["text"] == "alpha beta gamma"
    assert chunks[1]["text"] == "delta epsilon"


def test_long_page_produces_multiple_overlapping_chunks():
    words = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_text([(1, words)], max_words=10, overlap=3)
    assert len(chunks) > 1
    assert all(c["page"] == 1 for c in chunks)


def test_consecutive_chunks_share_overlap_words():
    max_words, overlap = 10, 3
    words = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_text([(1, words)], max_words=max_words, overlap=overlap)
    first = chunks[0]["text"].split()
    second = chunks[1]["text"].split()
    assert first[-overlap:] == second[:overlap]


def test_empty_and_whitespace_pages_produce_no_chunks():
    assert chunk_text([(1, ""), (2, "   \n\t  ")]) == []


def test_to_passage_and_to_query_prefixes():
    assert to_passage("hello world") == "passage: hello world"
    assert to_query("what is this?") == "query: what is this?"
