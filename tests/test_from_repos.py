"""Features adopted from the open-source projects Kiran listed (see README: Open-source projects)."""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

from docqa.agent import Agent
from docqa.config import Settings
from docqa.indexing import open_record_manager, open_vectorstore, sync_documents
from docqa.keyword import KeywordIndex, reciprocal_rank_fusion, tokenize
from docqa.loaders import iter_files, supported_extensions
from docqa.sample import create_sample_workspace
from docqa.sql import SQLAgent, UnsafeSQLError, check_read_only, draft_data_dictionary, open_database


@pytest.fixture
def sample(tmp_path):
    return create_sample_workspace(tmp_path / "ws")


@pytest.fixture
def indexed(sample, tmp_path):
    s = Settings(docs_dir=sample["docs"], data_dir=tmp_path / "data")
    vs = open_vectorstore(s, DeterministicFakeEmbedding(size=32))
    sync_documents(s, vs, open_record_manager(s))
    return vs


# --------------------------------------------------------------------------- hybrid search (RAGFlow, Verba, Continue, Cody)


def test_identifiers_are_split_and_kept_whole():
    toks = tokenize("def validateRequestToken(user_id): calls cust_stat_cd")
    assert {"validaterequesttoken", "validate", "request", "token", "user_id", "user", "cust_stat_cd", "stat"} <= set(toks)
    assert "the" not in tokenize("the")


def test_bm25_ranks_exact_identifier_and_file_name_first():
    docs = [
        Document("refund policy text about tokens of appreciation", metadata={"source": "docs/policy.md"}),
        Document("def calculate_retry_backoff(attempt): ...", metadata={"source": "code/retry.py"}),
        Document("class TokenValidator: def validate_request_token(self)", metadata={"source": "code/auth_service.py"}),
    ]
    idx = KeywordIndex(docs)
    assert idx.search("where is validate_request_token?", 1)[0].metadata["source"] == "code/auth_service.py"
    assert idx.search("what is in auth service", 1)[0].metadata["source"] == "code/auth_service.py"
    assert idx.search("zzz", 3) == []


def test_reciprocal_rank_fusion_rewards_agreement():
    a, b, c = (Document(x, metadata={"source": x}) for x in "abc")
    # b and c appear in both lists, so they beat a, which tops only one
    fused = reciprocal_rank_fusion([[a, b, c], [b, c]], k=3)
    assert [d.page_content for d in fused] == ["b", "c", "a"]


def test_hybrid_search_finds_code_by_identifier_where_vectors_alone_may_not(indexed):
    # fake embeddings are random: only the keyword half can find this reliably
    agent = Agent(FakeListChatModel(responses=[]), indexed, top_k=1)
    hit = agent.search("calculate_retry_backoff", "code", k=1)
    assert hit[0].metadata["source"] == "code/auth_service.py"
    agent.hybrid = False
    assert len(agent.search("calculate_retry_backoff", "code", k=1)) == 1  # still works, just vector-only


def test_keyword_index_is_rebuilt_after_refresh(indexed, sample, tmp_path):
    agent = Agent(FakeListChatModel(responses=[]), indexed)
    agent.search("anything", "doc")
    assert agent._keyword
    agent.refresh()
    assert not agent._keyword


# --------------------------------------------------------------------------- SQL policy (WrenAI)


@pytest.mark.parametrize("sql, dialect", [
    ("SELECT pg_read_file('/etc/passwd')", "postgres"),
    ("SELECT * FROM dblink('host=x', 'select 1') AS t(a int)", "postgres"),
    ("SELECT pg_sleep(600)", "postgres"),
    ("SELECT LOAD_FILE('/etc/passwd')", "mysql"),
    ("SELECT load_extension('evil')", "sqlite"),
    ("SELECT * FROM read_csv('/etc/passwd')", "duckdb"),
    ("SELECT * FROM OPENROWSET('SQLNCLI', 'srv', 'select 1')", "tsql"),
    ("SELECT * FROM orders FOR UPDATE", "postgres"),
])
def test_file_remote_and_locking_reads_are_blocked(sql, dialect):
    with pytest.raises(UnsafeSQLError):
        check_read_only(sql, dialect)


@pytest.mark.parametrize("sql", [
    "SELECT * FROM t WHERE name GLOB 'A*'",
    "SELECT url, sleep_hours FROM t",
    "SELECT strftime('%Y', d), COUNT(*) FROM t GROUP BY 1",
])
def test_ordinary_queries_with_similar_names_still_pass(sql):
    assert check_read_only(sql, "sqlite")


# --------------------------------------------------------------------------- secrets are never indexed (Continue)


def test_secret_files_and_folders_are_skipped(tmp_path):
    d = tmp_path / "repo"
    for rel in ["secrets/db.yaml", "keys/app.txt", "config/credentials.yml", "aws_secret.md", "id_rsa.txt",
                "src/password_reset.py", "src/app.py", "docs/guide.md"]:
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")
    names = [p.relative_to(d).as_posix() for p in iter_files(d, supported_extensions())]
    assert names == ["docs/guide.md", "src/app.py", "src/password_reset.py"]


def test_images_need_docling():
    assert ".png" not in supported_extensions(True, use_docling=False)
    assert ".png" in supported_extensions(True, use_docling=True)


# --------------------------------------------------------------------------- /run (SQLAIAgent-Ollama)


def test_run_your_own_sql_read_only(indexed, sample):
    db = open_database(sample["db_url"])
    agent = Agent(FakeListChatModel(responses=[]), indexed, sql_agent=SQLAgent(FakeListChatModel(responses=[]), db))
    ans = agent.ask("/run SELECT name FROM customers ORDER BY name")
    assert ans.sql.executed and ans.sql.rows[0] == ("Acme Corp",) and "5 rows" in ans.answer
    blocked = agent.ask("/run DELETE FROM customers")
    assert not blocked.sql.ok and "DELETE" in blocked.answer
    assert len(open_database(sample["db_url"]).table_infos()) == 2


# --------------------------------------------------------------------------- draft data dictionary (SmolSQLAgents)


def test_draft_data_dictionary_one_call_per_table(sample):
    db = open_database(sample["db_url"])
    llm = FakeListChatModel(responses=["## customers\nOne row per customer.", "## orders\nOne row per order."])
    text = draft_data_dictionary(llm, db)
    assert text.startswith("# Data dictionary (draft)") and "## customers" in text and "## orders" in text


def test_cli_describe_db_writes_draft_and_will_not_overwrite(sample, monkeypatch, capsys):
    from docqa.cli import main

    monkeypatch.setattr("docqa.providers.make_llm", lambda s: FakeListChatModel(responses=["## t"]))
    args = ["--env-file", "none", "--docs", str(sample["docs"]), "--db", sample["db_url"], "describe-db"]
    assert main(args) == 0
    assert (sample["docs"] / "data_dictionary_draft.md").exists()
    assert main(args) == 2 and "already exists" in capsys.readouterr().err
    assert main(args + ["--force"]) == 0
