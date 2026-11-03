from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

import docqa
import docqa.indexing
from docqa.cli import main


def test_cli_sync_then_ask(settings, monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("DOCQA_DOCS_DIR", str(settings.docs_dir))
    monkeypatch.setenv("DOCQA_DATA_DIR", str(tmp_path / "cli-data"))
    monkeypatch.setattr(docqa.indexing, "make_embeddings", lambda s: DeterministicFakeEmbedding(size=32))
    monkeypatch.setattr("docqa.providers.make_llm", lambda s: FakeListChatModel(responses=["30 days [1]."]))

    assert main(["--env-file", "none", "sync"]) == 0
    out = capsys.readouterr().out
    assert "3 files" in out and "added" in out

    assert main(["--env-file", "none", "sync"]) == 0
    assert "0 added" in capsys.readouterr().out

    assert main(["--env-file", "none", "ask", "Refund window?"]) == 0
    out = capsys.readouterr().out
    assert "30 days [1]." in out and "Sources:" in out


def test_cli_sync_refuses_empty_folder(settings, monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("DOCQA_DOCS_DIR", str(settings.docs_dir))
    monkeypatch.setenv("DOCQA_DATA_DIR", str(tmp_path / "cli-data"))
    monkeypatch.setattr(docqa.indexing, "make_embeddings", lambda s: DeterministicFakeEmbedding(size=32))
    assert main(["--env-file", "none", "sync"]) == 0
    for p in list(settings.docs_dir.rglob("*")):
        if p.is_file():
            p.unlink()
    capsys.readouterr()
    assert main(["--env-file", "none", "sync"]) == 2
    assert "Refusing to delete" in capsys.readouterr().err
    assert main(["--env-file", "none", "sync", "--allow-empty"]) == 0


def test_cli_flags_uncited_answer(settings, monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("DOCQA_DOCS_DIR", str(settings.docs_dir))
    monkeypatch.setenv("DOCQA_DATA_DIR", str(tmp_path / "cli-data"))
    monkeypatch.setattr(docqa.indexing, "make_embeddings", lambda s: DeterministicFakeEmbedding(size=32))
    monkeypatch.setattr("docqa.providers.make_llm", lambda s: FakeListChatModel(responses=["About 30 days."]))
    assert main(["--env-file", "none", "ask", "--sync", "Refund window?"]) == 0
    assert "none cited" in capsys.readouterr().out


def test_ollama_gets_context_window():
    from docqa.config import Settings
    from docqa.providers import make_llm

    assert make_llm(Settings(num_ctx=16384)).num_ctx == 16384
