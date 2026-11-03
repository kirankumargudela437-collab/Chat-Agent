import time

from docqa.indexing import open_record_manager, open_vectorstore, sync_documents


def _sources(vs):
    return sorted({m["source"] for m in vs.get()["metadatas"]})


def test_incremental_sync_add_skip_update_delete(settings, embeddings):
    vs = open_vectorstore(settings, embeddings)
    rm = open_record_manager(settings)

    first = sync_documents(settings, vs, rm)
    assert first.num_added == first.num_chunks > 0
    assert first.num_deleted == 0
    total = len(vs.get()["ids"])

    again = sync_documents(settings, vs, rm)
    assert again.num_added == 0 and again.num_deleted == 0
    assert again.num_skipped == first.num_chunks
    assert len(vs.get()["ids"]) == total  # no duplicates

    # edit one file
    md = settings.docs_dir / "handbook.md"
    md.write_text(md.read_text().replace("25 days", "30 days"))
    edited = sync_documents(settings, vs, rm)
    assert edited.num_added >= 1 and edited.num_deleted >= 1
    texts = " ".join(vs.get()["documents"])
    assert "30 days" in texts and "25 days" not in texts

    # delete one file
    (settings.docs_dir / "onboarding.docx").unlink()
    removed = sync_documents(settings, vs, rm)
    assert removed.num_deleted >= 1
    assert "onboarding.docx" not in _sources(vs)


def test_index_persists_across_processes(settings, embeddings):
    sync_documents(settings, open_vectorstore(settings, embeddings), open_record_manager(settings))
    # fresh objects pointing at the same data dir see the same index and skip everything
    res = sync_documents(settings, open_vectorstore(settings, embeddings), open_record_manager(settings))
    assert res.num_added == 0 and res.num_skipped == res.num_chunks


def test_collection_is_per_embedding_model(settings):
    from docqa.config import Settings

    other = Settings(docs_dir=settings.docs_dir, data_dir=settings.data_dir, embed_provider="openai")
    assert other.collection_name != settings.collection_name
