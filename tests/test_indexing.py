import pytest

from docqa import loaders
from docqa.config import Settings
from docqa.indexing import EmptySyncError, open_record_manager, open_vectorstore, sync_documents


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


def test_file_that_fails_to_load_keeps_its_chunks(settings, embeddings, monkeypatch):
    vs, rm = open_vectorstore(settings, embeddings), open_record_manager(settings)
    sync_documents(settings, vs, rm)
    before = len(vs.get()["ids"])

    real = loaders.load_file

    def locked(path, rel, use_docling=False):
        if rel == "onboarding.docx":
            raise PermissionError("file is locked")
        return real(path, rel, use_docling)

    monkeypatch.setattr(loaders, "load_file", locked)
    res = sync_documents(settings, vs, rm)
    assert res.num_deleted == 0 and res.kept == ("onboarding.docx",)
    assert "kept its previous chunks" in res.summary()
    assert "onboarding.docx" in _sources(vs) and len(vs.get()["ids"]) == before

    # once it loads again it is just "unchanged"
    monkeypatch.setattr(loaders, "load_file", real)
    again = sync_documents(settings, vs, rm)
    assert again.num_added == 0 and again.num_deleted == 0


def test_empty_folder_does_not_wipe_index(settings, embeddings):
    vs, rm = open_vectorstore(settings, embeddings), open_record_manager(settings)
    sync_documents(settings, vs, rm)
    for p in list(settings.docs_dir.rglob("*")):
        if p.is_file():
            p.unlink()

    with pytest.raises(EmptySyncError):
        sync_documents(settings, vs, rm)
    assert len(vs.get()["ids"]) > 0

    cleared = sync_documents(settings, vs, rm, allow_empty=True)
    assert cleared.num_deleted > 0 and len(vs.get()["ids"]) == 0


def test_other_docs_folder_gets_its_own_index(settings, embeddings, tmp_path):
    sync_documents(settings, open_vectorstore(settings, embeddings), open_record_manager(settings))
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    (other_dir / "c.md").write_text("Gamma facts.", encoding="utf-8")
    other = Settings(docs_dir=other_dir, data_dir=settings.data_dir)
    assert other.collection_name != settings.collection_name

    res = sync_documents(other, open_vectorstore(other, embeddings), open_record_manager(other))
    assert res.num_deleted == 0
    assert "handbook.md" in _sources(open_vectorstore(settings, embeddings))  # first folder untouched


def test_long_embedding_model_name_is_a_valid_collection(settings, embeddings):
    model = "hf.co/mixedbread-ai/mxbai-embed-large-v1-gguf:Q8_0-extra-long-tag-name"
    a = Settings(docs_dir=settings.docs_dir, data_dir=settings.data_dir, embed_model=model)
    b = Settings(docs_dir=settings.docs_dir, data_dir=settings.data_dir, embed_model=model + "-2")
    assert a.collection_name != b.collection_name
    assert a.collection_name[-1].isalnum()
    sync_documents(a, open_vectorstore(a, embeddings), open_record_manager(a))  # Chroma accepts the name


def test_chroma_telemetry_is_off(settings, embeddings):
    vs = open_vectorstore(settings, embeddings)
    assert vs._client.get_settings().anonymized_telemetry is False
