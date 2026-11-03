"""The Streamlit app, driven headless with streamlit.testing (fake models)."""

import sys

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import docqa.indexing  # noqa: E402
from docqa.sample import create_sample_workspace  # noqa: E402

APP = str(__import__("pathlib").Path(docqa.indexing.__file__).with_name("app.py"))
GOOD = "SELECT c.name, SUM(o.amount_usd) AS net FROM orders o JOIN customers c USING (customer_id) WHERE o.status = 'C' GROUP BY c.name ORDER BY net DESC"


@pytest.fixture
def app(tmp_path, monkeypatch):
    import streamlit as st

    ws = create_sample_workspace(tmp_path / "ws")
    monkeypatch.setenv("DOCQA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(docqa.indexing, "make_embeddings", lambda s: DeterministicFakeEmbedding(size=32))
    llm = FakeListChatModel(responses=["SQL", "map [1]", GOOD, "Acme Corp leads."])
    monkeypatch.setattr("docqa.providers.make_llm", lambda s: llm)
    monkeypatch.setattr(sys, "argv", ["app.py", "--env-file", "none", "--docs", str(ws["docs"]), "--db", ws["db_url"]])
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    yield at, ws
    st.cache_resource.clear()


def test_sync_then_sql_question_shows_query_and_table(app):
    at, ws = app
    assert not at.exception
    at.button[0].click().run()  # Sync folder
    assert "6 files" in at.sidebar.code[0].value

    at.chat_input[0].set_value("Rank customers by net revenue").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Acme Corp leads." in text and "```sql" in text
    assert at.dataframe[0].value.iloc[0]["name"] == "Acme Corp"
    assert any("Answered from: sql" in c.value for c in at.caption)

    save = [b for b in at.button if b.label == "Save as golden example"]
    save[0].click().run()
    golden = (ws["docs"] / "golden_queries.sql").read_text(encoding="utf-8")
    assert "-- Question: Rank customers by net revenue" in golden


def test_folder_of_databases_routes_each_question(tmp_path, monkeypatch):
    import sqlite3

    import streamlit as st

    for name, script in {
        "shop": "CREATE TABLE orders (id INTEGER, amount REAL); INSERT INTO orders VALUES (1, 9.5), (2, 20);",
        "zoo": "CREATE TABLE animals (id INTEGER, species TEXT); INSERT INTO animals VALUES (1, 'lion');",
    }.items():
        (tmp_path / "dbs" / name).mkdir(parents=True)
        conn = sqlite3.connect(tmp_path / "dbs" / name / f"{name}.sqlite")
        conn.executescript(script)
        conn.close()
    (tmp_path / "docs").mkdir()
    monkeypatch.setenv("DOCQA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DOCQA_DB_ROUTER", "retrieval")
    monkeypatch.setenv("DOCQA_SQL_SCHEMA_LINKING", "0")
    monkeypatch.setattr(docqa.indexing, "make_embeddings", lambda s: DeterministicFakeEmbedding(size=32))
    monkeypatch.setattr("docqa.providers.make_embeddings", lambda s: DeterministicFakeEmbedding(size=32))
    llm = FakeListChatModel(responses=["SELECT species FROM animals", "One lion."])
    monkeypatch.setattr("docqa.providers.make_llm", lambda s: llm)
    monkeypatch.setattr(sys, "argv", ["app.py", "--env-file", "none", "--docs", str(tmp_path / "docs"),
                                      "--db", str(tmp_path / "dbs")])
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception
    at.chat_input[0].set_value("Which animals species live in the zoo?").run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Database: **zoo**" in text and "One lion." in text
    st.cache_resource.clear()
