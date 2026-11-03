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
