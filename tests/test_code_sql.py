"""Code indexing, the SQL specialist and the routing agent (fake models, real SQLite + Chroma)."""


import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

from docqa.agent import Agent, parse_plan
from docqa.config import Settings
from docqa.indexing import open_record_manager, open_vectorstore, sync_documents
from docqa.loaders import iter_files, load_and_split, split_golden_examples
from docqa.sample import create_sample_workspace
from docqa.sql import (
    SQLAgent,
    UnsafeSQLError,
    append_golden_example,
    check_read_only,
    clean_sql,
    dialect_from_url,
    open_database,
    run_query,
)


class RecordingLLM(FakeListChatModel):
    prompts: list = []

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        self.prompts.append(messages)
        return super()._call(messages, stop, run_manager, **kwargs)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self.prompts.append(messages)
        yield from super()._stream(messages, stop, run_manager, **kwargs)


@pytest.fixture
def sample(tmp_path):
    return create_sample_workspace(tmp_path / "ws")


@pytest.fixture
def sample_settings(sample, tmp_path):
    return Settings(docs_dir=sample["docs"], data_dir=tmp_path / "data")


@pytest.fixture
def indexed(sample_settings):
    vs = open_vectorstore(sample_settings, DeterministicFakeEmbedding(size=32))
    sync_documents(sample_settings, vs, open_record_manager(sample_settings))
    return vs


@pytest.fixture
def db(sample):
    return open_database(sample["db_url"])


GOOD = "SELECT c.name, SUM(o.amount_usd) AS net FROM orders o JOIN customers c USING (customer_id) WHERE o.status = 'C' GROUP BY c.name ORDER BY net DESC"
BAD_COLUMN = GOOD.replace("o.amount_usd", "o.amount")


# --------------------------------------------------------------------------- loading code


def test_code_is_split_on_functions_with_line_numbers(tmp_path):
    d = tmp_path / "repo"
    d.mkdir()
    funcs = "\n\n".join(f"def func_{i}(x):\n    total = x\n" + "    total += 1\n" * 8 + "    return total\n" for i in range(12))
    (d / "big.py").write_text(funcs, encoding="utf-8")
    s = Settings(docs_dir=d, data_dir=tmp_path / "data", code_chunk_size=300, code_chunk_overlap=0)
    chunks, _ = load_and_split(s)
    assert len(chunks) > 3
    lines = funcs.splitlines()
    for c in chunks:
        assert c.metadata["category"] == "code" and c.metadata["language"] == "python"
        assert c.page_content.startswith("def func_")  # never cut in the middle of a function
        a, b = map(int, c.metadata["lines"].split("-"))
        assert lines[a - 1] == c.page_content.splitlines()[0]
        assert lines[b - 1] == c.page_content.splitlines()[-1]


@pytest.mark.parametrize("name, language", [
    ("svc.go", "go"), ("App.java", "java"), ("lib.rs", "rust"), ("Program.cs", "csharp"),
    ("index.tsx", "ts"), ("deploy.sh", "sh"), ("config.yaml", "yaml"),
])
def test_many_languages_are_indexed_as_code(tmp_path, name, language):
    d = tmp_path / "repo"
    d.mkdir()
    (d / name).write_text("// sample\nfunc main() {\n  run()\n}\n", encoding="utf-8")
    chunks, report = load_and_split(Settings(docs_dir=d, data_dir=tmp_path / "data"))
    assert report.files == [name]
    assert chunks[0].metadata["category"] == "code" and chunks[0].metadata["language"] == language


def test_dependency_folders_minified_and_huge_files_are_skipped(tmp_path):
    d = tmp_path / "repo"
    (d / "node_modules" / "lib").mkdir(parents=True)
    (d / "src").mkdir()
    (d / "node_modules" / "lib" / "index.js").write_text("function x() {}", encoding="utf-8")
    (d / "src" / "app.min.js").write_text("function x() {}", encoding="utf-8")
    (d / "src" / "app.js").write_text("function app() { return 1 }", encoding="utf-8")
    (d / "src" / "huge.js").write_text("x = 1;\n" * 400, encoding="utf-8")
    s = Settings(docs_dir=d, data_dir=tmp_path / "data", max_file_kb=1)
    assert [p.name for p in iter_files(d, s.extensions)] == ["app.js", "huge.js"]
    chunks, report = load_and_split(s)
    assert {c.metadata["source"] for c in chunks} == {"src/app.js"}
    assert "larger than 1 KB" in report.warnings["src/huge.js"]


def test_golden_examples_are_pulled_out_of_sql_files(sample_settings):
    chunks, _ = load_and_split(sample_settings)
    examples = [c for c in chunks if c.metadata["category"] == "example"]
    assert [c.metadata["question"] for c in examples] == [
        "Month-over-month net revenue growth",
        "Top customer by net revenue in each country",
    ]
    assert "LAG(net_revenue)" in examples[0].page_content
    assert examples[0].page_content.rstrip().endswith("ORDER BY month;")
    ddl = [c for c in chunks if c.metadata["category"] == "sql"]
    assert [c.metadata["source"] for c in ddl] == ["sql/schema.sql"]  # the golden file's header comment is not a chunk


def test_split_golden_examples_keeps_other_lines_in_place():
    text = "CREATE TABLE t (a int);\n-- Question: count rows\nSELECT\n  COUNT(*)\nFROM t;\nCREATE VIEW v AS SELECT 1;"
    examples, rest = split_golden_examples(text)
    assert examples == [("count rows", "SELECT\n  COUNT(*)\nFROM t;", 2, 5)]
    assert rest.splitlines()[0] == "CREATE TABLE t (a int);" and rest.splitlines()[5] == "CREATE VIEW v AS SELECT 1;"


def test_append_golden_example_round_trips(tmp_path):
    path = append_golden_example(tmp_path, "Revenue  by\nmonth", "SELECT 1\nFROM t")
    append_golden_example(tmp_path, "Second", "SELECT 2;")
    examples, _ = split_golden_examples(path.read_text(encoding="utf-8"))
    assert [(q, sql) for q, sql, *_ in examples] == [("Revenue by month", "SELECT 1\nFROM t;"), ("Second", "SELECT 2;")]
    with pytest.raises(ValueError):
        append_golden_example(tmp_path, "", "SELECT 1")


# --------------------------------------------------------------------------- SQL guardrail


@pytest.mark.parametrize("sql", [
    "SELECT * FROM orders",
    "select * from orders;",
    "WITH a AS (SELECT 1 AS x) SELECT x FROM a",
    "SELECT * FROM orders WHERE product_name = 'please update; drop table x'",
    "SELECT name FROM customers UNION SELECT product_name FROM orders",
    "SELECT name, RANK() OVER (ORDER BY name) FROM customers",
])
def test_read_only_queries_pass(sql):
    assert check_read_only(sql, "sqlite")


@pytest.mark.parametrize("sql, why", [
    ("DELETE FROM orders", "DELETE"),
    ("SELECT 1; DROP TABLE orders", "exactly one"),
    ("WITH gone AS (DELETE FROM orders RETURNING *) SELECT * FROM gone", "DELETE"),
    ("PRAGMA writable_schema = 1", "PRAGMA"),
    ("ATTACH DATABASE 'x.db' AS x", "ATTACH|read-only"),
    ("SELECT * INTO backup FROM orders", "INTO"),
    ("UPDATE orders SET status = 'C'", "UPDATE"),
    ("SELEC * FROM orders WHERE", "syntax|read-only"),
])
def test_unsafe_or_invalid_queries_are_rejected(sql, why):
    with pytest.raises(UnsafeSQLError, match=why):
        check_read_only(sql, "postgres" if "INTO" in sql or "RETURNING" in sql else "sqlite")


def test_connection_is_read_only_even_without_the_check(db, sample):
    with pytest.raises(Exception, match="readonly"):
        run_query(db, "DELETE FROM orders", 10)
    assert len(run_query(db, "SELECT * FROM orders", 100).rows) == 10


def test_run_query_caps_rows_and_passes_percent_signs(db):
    r = run_query(db, "SELECT name FROM customers WHERE name LIKE '%Corp%' ORDER BY name", 1)
    assert r.columns == ["name"] and r.rows == [("Acme Corp",)] and r.truncated


def test_clean_sql_and_dialects():
    assert clean_sql("Here you go:\n```sql\nSELECT 1;\n```\nThanks") == "SELECT 1;"
    assert clean_sql("SQL: SELECT 2") == "SELECT 2"
    assert dialect_from_url("postgresql+psycopg://u:p@h/db") == "postgres"
    assert dialect_from_url("mssql+pyodbc://h/db") == "tsql"


def test_missing_sqlite_file_is_a_clear_error(tmp_path):
    from docqa.sql import DatabaseError

    with pytest.raises(DatabaseError, match="not found"):
        open_database(f"sqlite:///{(tmp_path / 'nope.db').as_posix()}")


# --------------------------------------------------------------------------- self-healing SQL


def test_sql_agent_fixes_a_failed_query_with_the_database_error(db):
    llm = RecordingLLM(responses=["- orders join customers", f"```sql\n{BAD_COLUMN}\n```", f"```sql\n{GOOD}\n```"], prompts=[])
    r = SQLAgent(llm, db).run("Rank customers by net revenue", context="[1] Net revenue counts status 'C' only.")
    assert r.ok and r.executed and r.attempts == 2
    assert r.rows[0] == ("Acme Corp", 23200.0)
    assert len(r.failures) == 1 and r.failures[0][0] == BAD_COLUMN
    assert r.failures[0][1].startswith("no such column: o.amount")
    fix_prompt = llm.prompts[2][-1].content
    assert "no such column: o.amount" in fix_prompt and BAD_COLUMN in fix_prompt
    # the retry is told where the real column is (alias o -> orders, closest name amount_usd)
    assert "Table orders (alias o) has these columns" in fix_prompt and "orders.amount_usd" in fix_prompt
    # the live schema, the business rules and the mapping all reached the generator
    system = llm.prompts[1][0].content
    assert "CREATE TABLE orders" in system and "status 'C' only" in system and "orders join customers" in system


def test_sql_agent_rejects_writes_and_retries(db):
    llm = FakeListChatModel(responses=["map", "DELETE FROM orders", GOOD])
    r = SQLAgent(llm, db).run("delete the refunds")
    assert r.ok and r.attempts == 2 and "DELETE" in r.failures[0][1]


def test_sql_agent_gives_up_after_max_attempts(db):
    llm = FakeListChatModel(responses=["map", BAD_COLUMN, BAD_COLUMN, BAD_COLUMN])
    r = SQLAgent(llm, db, max_attempts=3).run("q")
    assert not r.ok and r.attempts == 3 and len(r.failures) == 2 and "no such column" in r.error


def test_sql_without_database_is_parsed_in_the_dialect_but_not_run():
    llm = FakeListChatModel(responses=[
        "map",
        "SELECT TOP 5 name FROM customers",          # T-SQL syntax, invalid in PostgreSQL
        "SELECT name FROM customers LIMIT 5",
    ])
    agent = SQLAgent(llm, None, dialect="postgres")
    r = agent.run("top 5 customers", schema="customers (customer_id, name)")
    assert r.ok and not r.executed and r.attempts == 2 and r.sql.endswith("LIMIT 5")
    assert "syntax error" in r.failures[0][1].lower() or "Expected" in r.failures[0][1]


def test_schema_linking_can_be_turned_off(db):
    llm = FakeListChatModel(responses=[GOOD])
    r = SQLAgent(llm, db, schema_linking=False).run("q")
    assert r.ok and r.mapping == ""


# --------------------------------------------------------------------------- routing agent


def test_parse_plan():
    avail = ["SQL", "CODE", "DOC"]
    assert parse_plan("CODE", avail) == ["CODE"]
    assert parse_plan("Route: code, docs", avail) == ["CODE", "DOC"]
    assert parse_plan("NOSQL? doc", avail) == ["DOC"]  # 'SQL' inside another word is not a route
    assert parse_plan("SQL", ["CODE", "DOC"]) == []    # unavailable routes are ignored
    assert parse_plan("I am not sure", avail) == []


def test_agent_routes_to_sql_and_uses_business_rules_and_golden_examples(indexed, db):
    llm = RecordingLLM(responses=["SQL", "- net revenue: status 'C' [1]", GOOD, "Acme Corp leads."], prompts=[])
    agent = Agent(llm, indexed, sql_agent=SQLAgent(llm, db))
    assert agent.available_routes() == ["SQL", "CODE", "DOC"]
    ans = agent.ask("Rank customers by net revenue")
    assert ans.routes == ("SQL",) and ans.sql.executed and ans.sql.rows[0][0] == "Acme Corp"
    assert ans.answer.startswith("Acme Corp leads.") and "```sql" in ans.answer
    assert not ans.uncited
    # the schema-linking prompt saw golden examples, DDL and the data dictionary, all numbered
    link_system = llm.prompts[1][0].content
    assert "golden SQL example" in link_system and "sql/schema.sql" in link_system and "docs/data_dictionary.md" in link_system
    assert ans.cited_docs and ans.cited_docs[0][0] == 1


def test_agent_code_route_searches_only_code(indexed):
    llm = RecordingLLM(responses=["CODE", "It compares the timestamp with ttl_seconds [1]."], prompts=[])
    ans = Agent(llm, indexed).ask("How does validate_request_token detect expiry?")
    assert ans.routes == ("CODE",)
    assert {d.metadata["category"] for d in ans.docs} == {"code"}
    assert "lines" in ans.sources_text()
    assert "senior software engineer" in llm.prompts[1][0].content


def test_agent_forced_route_skips_the_planner(indexed):
    llm = RecordingLLM(responses=["Enterprise uptime is 99.99% [1]."], prompts=[])
    ans = Agent(llm, indexed).ask("What uptime do Enterprise customers get?", route="doc")
    assert ans.routes == ("DOC",) and len(llm.prompts) == 1
    assert {d.metadata["category"] for d in ans.docs} == {"doc"}


def test_agent_multi_route_runs_each_once_and_combines(indexed):
    llm = RecordingLLM(responses=[
        "CODE, DOC",
        "MAX_RETRIES is 3 [1].",
        "The runbook says retry three times [5].",
        "The code retries 3 times [1], matching the policy [5].",
    ], prompts=[])
    ans = Agent(llm, indexed).ask("Does the client retry as often as the policy says?")
    assert ans.routes == ("CODE", "DOC") and len(llm.prompts) == 4
    combine = llm.prompts[3][0].content
    assert "CODE specialist" in combine and "DOC specialist" in combine
    assert [i for i, _ in ans.cited_docs] == [1, 5]
    assert ans.cited_docs[0][1].metadata["category"] == "code" and ans.cited_docs[1][1].metadata["category"] == "doc"


def test_agent_unclear_plan_searches_everything(indexed):
    llm = FakeListChatModel(responses=["hmm, not sure", "Answer [1]."])
    ans = Agent(llm, indexed).ask("tell me something")
    assert ans.routes == ("ALL",) and ans.docs


def test_agent_sql_follow_up_is_rewritten_with_the_previous_query(indexed, db):
    llm = RecordingLLM(responses=[
        "SQL", "map", GOOD, "Acme Corp leads.",
        "Rank Enterprise customers by net revenue",   # condense
        "SQL", "map", GOOD.replace("WHERE", "WHERE c.tier = 'Enterprise' AND"), "Acme Corp leads Enterprise.",
    ], prompts=[])
    agent = Agent(llm, indexed, sql_agent=SQLAgent(llm, db))
    agent.ask("Rank customers by net revenue")
    ans = agent.ask("Only Enterprise customers")
    assert "SUM(o.amount_usd)" in llm.prompts[4][-1].content  # condense saw the earlier SQL
    assert ans.standalone_question == "Rank Enterprise customers by net revenue"
    assert {r[0] for r in ans.sql.rows} == {"Acme Corp", "Initech", "Umbrella Ltd"}


def test_sql_route_without_any_schema_says_so(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text("# A\n\nOur office opens at 9.", encoding="utf-8")
    s = Settings(docs_dir=d, data_dir=tmp_path / "data")
    vs = open_vectorstore(s, DeterministicFakeEmbedding(size=32))
    sync_documents(s, vs, open_record_manager(s))
    ans = Agent(FakeListChatModel(responses=[]), vs).ask("count orders", route="sql")
    assert "no schema" in ans.answer.lower()


def test_pasted_schema_enables_the_sql_route(indexed):
    agent = Agent(FakeListChatModel(responses=[]), indexed)
    agent.schema = "orders (order_id, amount_usd)"
    assert "SQL" in agent.available_routes()


# --------------------------------------------------------------------------- CLI


def test_cli_sample_then_sql_question(tmp_path, monkeypatch, capsys):
    import docqa.indexing
    from docqa.cli import main

    monkeypatch.chdir(tmp_path)
    assert main(["--env-file", "none", "sample", "ws"]) == 0
    assert "docqa --docs" in capsys.readouterr().out

    monkeypatch.setenv("DOCQA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(docqa.indexing, "make_embeddings", lambda s: DeterministicFakeEmbedding(size=32))
    monkeypatch.setattr("docqa.providers.make_llm", lambda s: FakeListChatModel(responses=["map", GOOD, "Acme Corp leads."]))
    args = ["--env-file", "none", "--docs", "ws/workspace_files", "--db", "sqlite:///ws/my_company.db"]
    assert main(args + ["ask", "--sync", "--route", "sql", "Rank customers by net revenue"]) == 0
    out = capsys.readouterr().out
    assert "Acme Corp leads." in out and "Result (5 rows)" in out and "Answered from: sql" in out


def test_cli_bad_database_url_is_a_clean_error(tmp_path, monkeypatch, capsys):
    from docqa.cli import main

    monkeypatch.setattr("docqa.providers.make_llm", lambda s: FakeListChatModel(responses=[]))
    monkeypatch.setenv("DOCQA_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "docs").mkdir()
    code = main(["--env-file", "none", "--docs", str(tmp_path / "docs"), "--db", "sqlite:///missing.db", "ask", "q"])
    assert code == 2 and "not found" in capsys.readouterr().err


# --------------------------------------------------------------------------- organisation-specific schemas


@pytest.fixture
def legacy_db(tmp_path):
    """An organisation's database with legacy names and many unrelated tables."""
    import sqlite3

    path = tmp_path / "org.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE usr_mst (usr_id INTEGER PRIMARY KEY, tier_cd INTEGER, lt_spend REAL);"
        "CREATE TABLE ord_hdr (ord_id INTEGER PRIMARY KEY, usr_id INTEGER REFERENCES usr_mst(usr_id), net_amt REAL);"
        "CREATE TABLE rgn_lkp (rgn_id INTEGER PRIMARY KEY, rgn_nm TEXT);"
        + "".join(f"CREATE TABLE audit_log_{i} (id INTEGER PRIMARY KEY, msg TEXT);" for i in range(30))
        + "INSERT INTO usr_mst VALUES (1, 5, 12000), (2, 1, 50);"
    )
    conn.close()
    return open_database(f"sqlite:///{path.as_posix()}")


def test_big_database_shows_tables_named_in_the_business_rules_plus_fk_neighbours(legacy_db):
    agent = SQLAgent(FakeListChatModel(responses=[]), legacy_db, max_tables=4)
    rules = "[1] A gold tier user means usr_mst.tier_cd = 5 and lt_spend > 10000."
    chosen = agent.select_tables("How many gold tier users placed orders?", rules)
    assert chosen[0] == "usr_mst" and "ord_hdr" in chosen and len(chosen) == 4
    text = agent.schema_text("", "How many gold tier users placed orders?", rules)
    assert "CREATE TABLE usr_mst" in text and "CREATE TABLE ord_hdr" in text
    assert "29 more tables not shown" in text


def test_small_database_is_shown_whole(db):
    agent = SQLAgent(FakeListChatModel(responses=[]), db)
    assert agent.select_tables("anything") == ["customers", "orders"]


def test_schema_text_respects_the_character_budget(legacy_db):
    agent = SQLAgent(FakeListChatModel(responses=[]), legacy_db, max_tables=33, schema_chars=600)
    text = agent.schema_text("", "users")
    assert 0 < text.count("CREATE TABLE") < 33 and "more tables not shown" in text


def test_table_ranking_uses_embeddings_when_available(legacy_db):
    class KeywordEmbeddings:
        def embed_documents(self, texts):
            return [self.embed_query(t) for t in texts]

        def embed_query(self, text):
            t = text.lower()
            return [float("rgn" in t or "region" in t), float("audit" in t), 0.1]

    agent = SQLAgent(FakeListChatModel(responses=[]), legacy_db, max_tables=2, embeddings=KeywordEmbeddings())
    assert agent.select_tables("Revenue by region")[0] == "rgn_lkp"


def test_golden_examples_are_shown_as_earlier_turns(db):
    llm = RecordingLLM(responses=["map", GOOD], prompts=[])
    SQLAgent(llm, db).run("Rank customers", examples=[("Top product", "SELECT product_name FROM orders")])
    msgs = llm.prompts[1]
    assert [type(m).__name__ for m in msgs] == ["SystemMessage", "HumanMessage", "AIMessage", "HumanMessage"]
    assert msgs[1].content == "Top product" and "SELECT product_name" in msgs[2].content


def test_agent_passes_indexed_golden_examples_as_turns(indexed, db):
    llm = RecordingLLM(responses=["map", GOOD, "Acme leads."], prompts=[])
    Agent(llm, indexed, sql_agent=SQLAgent(llm, db)).ask("Monthly revenue growth", route="sql")
    generate = llm.prompts[1]
    assert any(type(m).__name__ == "AIMessage" and "LAG(net_revenue)" in m.content for m in generate)
