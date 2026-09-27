from ragsync.document_source import DirectorySource, split_into_chunks


def test_split_packs_paragraphs_under_limit():
    text = "para one.\n\npara two.\n\npara three."
    chunks = split_into_chunks(text, max_chars=100)
    assert chunks == ["para one.\n\npara two.\n\npara three."]


def test_split_breaks_when_limit_exceeded():
    text = "a" * 50 + "\n\n" + "b" * 50
    chunks = split_into_chunks(text, max_chars=60)
    assert chunks == ["a" * 50, "b" * 50]


def test_split_hard_splits_oversized_paragraph():
    text = "x" * 25
    chunks = split_into_chunks(text, max_chars=10)
    assert chunks == ["x" * 10, "x" * 10, "x" * 5]


def test_split_empty_text_gives_no_chunks():
    assert split_into_chunks("   \n\n  ") == []


def test_directory_source_loads_txt_and_md(tmp_path):
    (tmp_path / "a.txt").write_text("hello from a")
    (tmp_path / "b.md").write_text("hello from b")
    (tmp_path / "ignore.json").write_text('{"x": 1}')
    source = DirectorySource(str(tmp_path))
    chunks = source.load_chunks()
    assert {c.source_id for c in chunks} == {"a.txt", "b.md"}
    assert {c.text for c in chunks} == {"hello from a", "hello from b"}


def test_directory_source_chunk_ids_are_unique_per_source(tmp_path):
    long_text = "\n\n".join(f"paragraph {i} " + "x" * 100 for i in range(20))
    (tmp_path / "big.txt").write_text(long_text)
    source = DirectorySource(str(tmp_path), max_chars=200)
    chunks = source.load_chunks()
    assert len(chunks) > 1
    assert len({c.chunk_id for c in chunks}) == len(chunks)
