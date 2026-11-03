"""Routing across many SQLite databases, and the Spider / BIRD evaluation (fake models, tiny fake benchmarks)."""

import json
import sqlite3
import zipfile
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

from docqa import benchmark as bm
from docqa.agent import Agent
from docqa.cli import main
from docqa.multidb import CatalogError, DatabaseCatalog, find_databases, match_name
from docqa.sql import SQLAgent, format_sql_answer


def make_db(path: Path, script: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(script)
    conn.commit()
    conn.close()
    return path


CONCERT = """
CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, country TEXT, age INTEGER);
CREATE TABLE concert (concert_id INTEGER PRIMARY KEY, concert_name TEXT, year TEXT);
CREATE TABLE singer_in_concert (concert_id INTEGER REFERENCES concert(concert_id), singer_id INTEGER REFERENCES singer(singer_id));
INSERT INTO singer VALUES (1,'Joe Sharp','Netherlands',52),(2,'Timbaland','United States',32),(3,'Justin Brown','France',29),(4,'Rose White','France',41);
INSERT INTO concert VALUES (1,'Auditions','2014'),(2,'Super bootcamp','2015');
INSERT INTO singer_in_concert VALUES (1,2),(1,3),(2,3);
"""
CARS = """
CREATE TABLE cars_data (id INTEGER PRIMARY KEY, make TEXT, horsepower INTEGER, weight INTEGER, year INTEGER);
INSERT INTO cars_data VALUES (1,'ford',130,3504,1970),(2,'buick',165,3693,1970),(3,'toyota',95,2372,1971);
"""
SCHOOL = """
CREATE TABLE stu (stu_id INTEGER PRIMARY KEY, nm TEXT, gpa_x REAL);
INSERT INTO stu VALUES (1,'Ann',3.5),(2,'Bo',2.9);
"""


@pytest.fixture
def spider(tmp_path):
    root = tmp_path / "spider" / "spider_data"
    db = root / "database"
    make_db(db / "concert_singer" / "concert_singer.sqlite", CONCERT)
    make_db(db / "car_1" / "car_1.sqlite", CARS)
    make_db(db / "school" / "school.sqlite", SCHOOL)
    # BIRD-style column descriptions: the cryptic gpa_x column gets a meaning
    (db / "school" / "database_description").mkdir()
    (db / "school" / "database_description" / "stu.csv").write_text(
        "original_column_name,column_name,column_description,data_format,value_description\n"
        "nm,name,student name,text,\ngpa_x,grade point average,GPA on a 4 point scale,real,\n",
        encoding="utf-8",
    )
    dev = [
        {"db_id": "concert_singer", "question": "How many singers are from France?",
         "query": "SELECT count(*) FROM singer WHERE country = 'France'"},
        {"db_id": "concert_singer", "question": "List singer names ordered by age.",
         "query": "SELECT name FROM singer ORDER BY age"},
        {"db_id": "car_1", "question": "What is the horsepower of the heaviest car?",
         "query": "SELECT horsepower FROM cars_data ORDER BY weight DESC LIMIT 1"},
    ]
    (root / "dev.json").write_text(json.dumps(dev), encoding="utf-8")
    return tmp_path / "spider"


class CountingEmbeddings(DeterministicFakeEmbedding):
    calls: int = 0

    def embed_documents(self, texts):
        type(self).calls += len(texts)
        return super().embed_documents(texts)


# --------------------------------------------------------------------------- catalog


def test_finds_databases_and_builds_cards(spider):
    root = spider / "spider_data" / "database"
    assert sorted(find_databases(root)) == ["car_1", "concert_singer", "school"]
    cat = DatabaseCatalog(root)
    card = cat.cards["concert_singer"].text
    assert "singer_in_concert" in card and "joins singer_id -> singer.singer_id" in card
    assert "France" in card  # example values help route "singers from France"
    assert "GPA on a 4 point scale" in cat.cards["school"].text


def test_missing_or_empty_folder(tmp_path):
    with pytest.raises(CatalogError):
        DatabaseCatalog(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(CatalogError):
        DatabaseCatalog(tmp_path / "empty")


def test_keyword_routing_without_embeddings(spider):
    cat = DatabaseCatalog(spider / "spider_data" / "database")
    assert cat.rank("How many singers are from France?")[0][0] == "concert_singer"
    assert cat.rank("Which car has the most horsepower?")[0][0] == "car_1"
    assert cat.rank("average grade point of students", 1)[0][0] == "school"  # via the description CSV


def test_cards_and_embeddings_are_cached(spider, tmp_path):
    root = spider / "spider_data" / "database"
    CountingEmbeddings.calls = 0
    DatabaseCatalog(root, CountingEmbeddings(size=16), cache_dir=tmp_path / "data", cache_key="fake")
    assert CountingEmbeddings.calls == 3
    cat = DatabaseCatalog(root, CountingEmbeddings(size=16), cache_dir=tmp_path / "data", cache_key="fake")
    assert CountingEmbeddings.calls == 3  # nothing re-embedded
    assert len(cat.rank("singers", 2)) == 2
    make_db(root / "car_1" / "car_1.sqlite", "CREATE TABLE extra (x INTEGER);")  # a changed file
    DatabaseCatalog(root, CountingEmbeddings(size=16), cache_dir=tmp_path / "data", cache_key="fake")
    assert CountingEmbeddings.calls == 4


def test_match_name():
    names = ["car_1", "car_10", "concert_singer"]
    assert match_name("car_1", names) == "car_1"
    assert match_name("The answer is `concert_singer`.", names) == "concert_singer"
    assert match_name("Database: car_10", names) == "car_10"
    assert match_name("no idea", names) is None


# --------------------------------------------------------------------------- SQL agent with a catalog


def test_sql_agent_routes_then_runs_on_that_database(spider):
    cat = DatabaseCatalog(spider / "spider_data" / "database")
    llm = FakeListChatModel(responses=["```sql\nSELECT count(*) FROM singer WHERE country = 'France'\n```"])
    agent = SQLAgent(llm, catalog=cat, db_router="retrieval", schema_linking=False)
    r = agent.run("How many singers are from France?")
    assert r.ok and r.executed and r.database == "concert_singer" and r.rows == [(2,)]
    assert r.candidates[0] == "concert_singer"
    assert "Database: **concert_singer**" in format_sql_answer(r)


def test_llm_router_picks_among_candidates(spider):
    cat = DatabaseCatalog(spider / "spider_data" / "database")
    llm = FakeListChatModel(responses=["car_1", "```sql\nSELECT max(horsepower) FROM cars_data\n```"])
    agent = SQLAgent(llm, catalog=cat, db_router="llm", schema_linking=False)
    r = agent.run("What is the strongest engine?")
    assert r.database == "car_1" and r.rows == [(165,)]


def test_bird_descriptions_reach_the_sql_prompt(spider):
    cat = DatabaseCatalog(spider / "spider_data" / "database")
    agent = SQLAgent(FakeListChatModel(responses=["x"]), catalog=cat, schema_linking=False)
    agent.use_database("school")
    assert "GPA on a 4 point scale" in agent.schema_text(question="best gpa")


def test_wrong_sql_on_routed_database_is_retried(spider):
    cat = DatabaseCatalog(spider / "spider_data" / "database")
    llm = FakeListChatModel(responses=["SELECT nme FROM singer", "SELECT name FROM singer WHERE age > 40"])
    agent = SQLAgent(llm, catalog=cat, db_router="retrieval", schema_linking=False)
    r = agent.run("Which singers are older than 40?")
    assert r.ok and r.attempts == 2 and sorted(r.rows) == [("Joe Sharp",), ("Rose White",)]


def test_agent_run_command_with_database_prefix(spider, tmp_path):
    from docqa.config import Settings
    from docqa.indexing import open_vectorstore

    cat = DatabaseCatalog(spider / "spider_data" / "database")
    llm = FakeListChatModel(responses=["unused"])
    s = Settings(docs_dir=tmp_path / "docs", data_dir=tmp_path / "data")
    vs = open_vectorstore(s, DeterministicFakeEmbedding(size=8))
    agent = Agent(llm, vs, sql_agent=SQLAgent(llm, catalog=cat))
    assert "SQL" in agent.available_routes()
    assert "start with @<database>" in agent.run_sql("SELECT 1").error
    r = agent.run_sql("@car_1 SELECT count(*) FROM cars_data")
    assert r.ok and r.rows == [(3,)] and r.database == "car_1"
    assert agent.run_sql("@nope SELECT 1").error.startswith("Unknown database")
    assert "read-only" in agent.run_sql("@car_1 DELETE FROM cars_data").error


# --------------------------------------------------------------------------- benchmark helpers


def test_locate_and_load_spider(spider):
    split = bm.locate("spider", spider)
    assert split.db_root.name == "database" and len(split.load()) == 3
    assert split.db_path("car_1").name == "car_1.sqlite"
    with pytest.raises(bm.BenchmarkError):
        bm.locate("spider", spider / "missing")


def test_pick_is_reproducible_and_spread(spider):
    ex = bm.locate("spider", spider).load()
    assert [e.id for e in bm.pick(ex, 2, seed=1)] == [e.id for e in bm.pick(ex, 2, seed=1)]
    assert {e.db_id for e in bm.pick(ex, None, db="car_1")} == {"car_1"}


def test_results_match_rules():
    assert bm.results_match([(1,), (2,)], [(2,), (1,)], "spider")
    assert not bm.results_match([(1,), (2,)], [(2,), (1,)], "spider", ordered=True)
    assert not bm.results_match([(1,), (1,)], [(1,)], "spider")  # duplicates count in Spider
    assert bm.results_match([(1,), (1,)], [(1,)], "bird")  # BIRD compares sets
    assert bm.results_match([(0.1 + 0.2,)], [(0.3,)], "bird")
    assert bm.has_top_level_order_by("SELECT a FROM t ORDER BY a")
    assert not bm.has_top_level_order_by("SELECT a FROM (SELECT a FROM t ORDER BY a LIMIT 3)")


def test_execute_times_out(spider):
    path = bm.locate("spider", spider).db_path("car_1")
    endless = "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT count(*) FROM n"
    with pytest.raises(bm.QueryTimeout):
        bm.execute(path, endless, timeout=0.2)


def test_evaluate_scores_routing_and_execution(spider):
    split = bm.locate("spider", spider)
    examples = split.load()
    cat = DatabaseCatalog(split.db_root)
    llm = FakeListChatModel(responses=[
        "SELECT count(*) FROM singer WHERE country = 'France'",        # right
        "SELECT name FROM singer ORDER BY age DESC",                     # wrong order
        "SELECT horsepower FROM cars_data ORDER BY weight DESC LIMIT 1",  # right
    ])
    agent = SQLAgent(llm, catalog=cat, db_router="retrieval", schema_linking=False)
    outs = list(bm.evaluate(agent, split, examples))
    assert [o.correct for o in outs] == [True, False, True]
    text = bm.summarize(outs)
    assert "Execution accuracy: 66.7% (2/3)" in text and "right database 100.0% (3/3)" in text


def test_gold_failure_is_skipped_not_scored(spider):
    split = bm.locate("spider", spider)
    ex = [bm.Example("x", "car_1", "q", "SELECT nope FROM cars_data")]
    agent = SQLAgent(FakeListChatModel(responses=["SELECT 1"]), catalog=DatabaseCatalog(split.db_root))
    outs = list(bm.evaluate(agent, split, ex))
    assert outs[0].gold_error and "Skipped 1" in bm.summarize(outs)


def test_bird_layout_with_nested_zip_and_evidence(tmp_path):
    # dev.zip -> dev_20240627/{dev.json, dev_databases.zip -> dev_databases/<db>/<db>.sqlite}
    work = tmp_path / "build"
    make_db(work / "dev_databases" / "school" / "school.sqlite", SCHOOL)
    inner = tmp_path / "dev_databases.zip"
    with zipfile.ZipFile(inner, "w") as zf:
        zf.write(work / "dev_databases" / "school" / "school.sqlite", "dev_databases/school/school.sqlite")
    dev = [{"question_id": 7, "db_id": "school", "question": "Best student?", "evidence": "best refers to MAX(gpa_x)",
            "SQL": "SELECT nm FROM stu ORDER BY gpa_x DESC LIMIT 1", "difficulty": "simple"}]
    outer = tmp_path / "dev.zip"
    with zipfile.ZipFile(outer, "w") as zf:
        zf.writestr("dev_20240627/dev.json", json.dumps(dev))
        zf.write(inner, "dev_20240627/dev_databases.zip")
    folder = bm.get_benchmark("bird", tmp_path / "benchmarks", zip_path=outer)
    split = bm.locate("bird", folder)
    assert split.load()[0].evidence.startswith("best refers")

    seen = []

    class Spy(FakeListChatModel):
        def _call(self, messages, *a, **k):
            seen.append(str(messages))
            return super()._call(messages, *a, **k)

    agent = SQLAgent(Spy(responses=["SELECT nm FROM stu WHERE gpa_x = (SELECT MAX(gpa_x) FROM stu)"]),
                     catalog=DatabaseCatalog(split.db_root), schema_linking=False)
    outs = list(bm.evaluate(agent, split, split.load(), oracle_db=True))
    assert outs[0].correct and "MAX(gpa_x)" in seen[0]  # the evidence is in the prompt
    assert "simple 100.0% (1/1)" in bm.summarize(outs, routed=False)


def test_zip_slip_is_refused(tmp_path):
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(bm.BenchmarkError, match="outside"):
        bm.get_benchmark("spider", tmp_path / "b", zip_path=bad)
    assert not (tmp_path / "b" / "evil.txt").exists() and not (tmp_path / "evil.txt").exists()


# --------------------------------------------------------------------------- CLI


def _env(monkeypatch, tmp_path, **extra):
    """Settings through monkeypatched variables (a .env file would leak into os.environ for later tests)."""
    for k, v in {"DOCQA_DATA_DIR": str(tmp_path / "data"), **extra}.items():
        monkeypatch.setenv(k, v)
    return tmp_path / "no.env"


def test_cli_eval_route_only(spider, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env = _env(monkeypatch, tmp_path)
    code = main(["--env-file", str(env), "eval", "spider", "--data", str(spider), "--route-only", "--no-embeddings"])
    out = capsys.readouterr().out
    assert code == 0 and "Routing top-1 accuracy: 100.0% (3/3)" in out
    assert list((tmp_path / "data" / "eval").glob("*.jsonl"))


def test_cli_eval_with_model_and_resume(spider, tmp_path, capsys, monkeypatch):
    import docqa.providers

    answers = iter([
        "SELECT count(*) FROM singer WHERE country = 'France'",
        "SELECT name FROM singer ORDER BY age",
        "SELECT horsepower FROM cars_data ORDER BY weight DESC LIMIT 1",
    ])
    monkeypatch.setattr(docqa.providers, "make_llm", lambda s: FakeListChatModel(responses=list(answers)))
    env = _env(monkeypatch, tmp_path, DOCQA_SQL_SCHEMA_LINKING="0")
    args = ["--env-file", str(env), "eval", "spider", "--data", str(spider), "--oracle-db"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "Execution accuracy: 100.0% (3/3)" in out
    results = next((tmp_path / "data" / "eval").glob("*.jsonl"))
    assert main(args + ["--resume", str(results)]) == 0  # nothing left to ask: no model calls needed
    assert "3 questions already done" in capsys.readouterr().out
    assert len(results.read_text(encoding="utf-8").splitlines()) == 3


def test_cli_chat_header_with_db_dir(spider, tmp_path, monkeypatch, capsys):
    import docqa
    import docqa.providers

    monkeypatch.setattr(docqa.providers, "make_llm", lambda s: FakeListChatModel(responses=["x"]))
    monkeypatch.setattr(docqa.providers, "make_embeddings", lambda s: DeterministicFakeEmbedding(size=8))
    monkeypatch.setattr("builtins.input", lambda *_: "/quit")
    env = _env(monkeypatch, tmp_path, DOCQA_DOCS_DIR=str(tmp_path / "docs"))
    (tmp_path / "docs").mkdir()
    assert main(["--env-file", str(env), "--db-dir", str(spider / "spider_data" / "database"), "chat"]) == 0
    assert "3 databases in" in capsys.readouterr().out
