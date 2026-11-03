from docqa.loaders import load_and_split


def test_loads_all_formats_with_citation_metadata(settings):
    chunks, report = load_and_split(settings)
    assert sorted(report.files) == ["handbook.md", "onboarding.docx", "policies/refunds.pdf"]
    by_source = {}
    for c in chunks:
        by_source.setdefault(c.metadata["source"], []).append(c)

    pdf_pages = sorted(c.metadata["page"] for c in by_source["policies/refunds.pdf"])
    assert pdf_pages == [1, 2]  # 1-based, one per page
    assert "billing@example.com" in by_source["policies/refunds.pdf"][1].page_content

    assert "page" not in by_source["onboarding.docx"][0].metadata
    sections = {c.metadata.get("section") for c in by_source["handbook.md"]}
    assert "Handbook > Leave" in sections


def test_metadata_is_minimal_and_stable(settings):
    chunks, _ = load_and_split(settings)
    for c in chunks:
        assert set(c.metadata) <= {"source", "page", "section"}


def test_chunks_respect_size_and_overlap(settings):
    long_text = " ".join(f"sentence{i} is here." for i in range(200))
    (settings.docs_dir / "long.txt").write_text(long_text, encoding="utf-8")
    chunks, _ = load_and_split(settings)
    long_chunks = [c.page_content for c in chunks if c.metadata["source"] == "long.txt"]
    assert len(long_chunks) > 5
    assert all(len(c) <= settings.chunk_size for c in long_chunks)
    # consecutive chunks share text (overlap)
    assert long_chunks[0][-20:].split()[-1] in long_chunks[1]


def test_bad_file_is_reported_not_fatal(settings):
    (settings.docs_dir / "broken.pdf").write_bytes(b"not a pdf")
    chunks, report = load_and_split(settings)
    assert "broken.pdf" in report.errors
    assert chunks
