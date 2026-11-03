"""Accuracy features for small models: value lookup, error hints, few-shot bank, voting, Spider scoring."""

import json

from langchain_core.language_models import FakeListChatModel

from docqa import benchmark as bm
from docqa.multidb import DatabaseCatalog
from docqa.sql import SQLAgent, open_database, vote
from docqa.sqlhints import ValueIndex, error_hint, format_value_hints

from test_code_sql import RecordingLLM, sample  # noqa: F401  (fixture)
from test_multidb import make_db, spider  # noqa: F401  (fixture)

CARDS = """
CREATE TABLE cards (id INTEGER PRIMARY KEY, name TEXT, type TEXT, "converted mana cost" REAL, released TEXT);
CREATE TABLE sets (code TEXT PRIMARY KEY, set_name TEXT);
INSERT INTO cards VALUES (1,'Ancestor''s Chosen','Creature — Human Cleric',7,'01:54'),
                         (2,'Angel of Mercy','commander',5,'02:10'),(3,'Aven Cloudchaser','Creature',4,'03:00');
INSERT INTO sets VALUES ('10E','Tenth Edition'),('M14','Magic 2014');
"""


def _cards_db(tmp_path):
    path = make_db(tmp_path / "cards.sqlite", CARDS)
    return open_database(f"sqlite:///{path.as_posix()}")


# --------------------------------------------------------------------------- value lookup


def test_value_index_finds_case_and_zero_padding_variants(tmp_path):
    idx = ValueIndex(_cards_db(tmp_path))
    hits = idx.match("How many Commander cards were released at 1:54 in the Tenth Edition set?")
    found = {(t, c, v) for t, c, v in hits}
    assert ("cards", "type", "commander") in found          # 'Commander' -> stored 'commander'
    assert ("cards", "released", "01:54") in found          # '1:54' -> stored '01:54'
    assert ("sets", "set_name", "Tenth Edition") in found   # multi-word value
    text = format_value_hints(hits)
    assert "cards.type = 'commander'" in text and "exact spellings" in text


def test_value_index_quoted_substring_and_no_noise(tmp_path):
    idx = ValueIndex(_cards_db(tmp_path))
    # a quoted fragment finds the longer stored value
    assert ("cards", "name", "Ancestor's Chosen") in idx.match('What is the type of "Ancestor"?')
    assert idx.match("How many are there in total?") == []


def test_value_hints_reach_the_sql_prompt(tmp_path):
    db = _cards_db(tmp_path)
    llm = RecordingLLM(responses=["```sql\nSELECT count(*) FROM cards WHERE type = 'commander'\n```"], prompts=[])
    r = SQLAgent(llm, db, schema_linking=False).run("How many Commander cards are there?")
    assert r.ok and r.rows == [(1,)]
    assert "cards.type = 'commander'" in llm.prompts[0][0].content


def test_value_lookup_can_be_turned_off(tmp_path):
    db = _cards_db(tmp_path)
    llm = RecordingLLM(responses=["SELECT 1"], prompts=[])
    SQLAgent(llm, db, schema_linking=False, value_lookup=False).run("How many Commander cards are there?")
    assert "commander'" not in llm.prompts[0][0].content


# --------------------------------------------------------------------------- error hints


def test_error_hint_names_the_real_columns(tmp_path):
    db = _cards_db(tmp_path)
    sql = "SELECT T1.name FROM cards AS T1 JOIN sets AS T2 ON T2.code = T1.id WHERE T1.mana_cost > 3"
    hint = error_hint(db, sql, "no such column: T1.mana_cost")
    assert "Table cards (alias T1) has these columns" in hint
    assert '"converted mana cost"' in hint and "must be quoted" in hint


def test_error_hint_for_wrong_table_and_column_in_other_table(tmp_path):
    db = _cards_db(tmp_path)
    assert "Closest names: cards" in error_hint(db, "SELECT * FROM card", "no such table: card")
    hint = error_hint(db, "SELECT T1.set_name FROM cards T1", "no such column: T1.set_name")
    assert "set_name is a column of sets" in hint


def test_retry_gets_the_hint(tmp_path):
    db = _cards_db(tmp_path)
    llm = RecordingLLM(responses=["SELECT nme FROM cards", "SELECT name FROM cards WHERE id = 2"], prompts=[])
    r = SQLAgent(llm, db, schema_linking=False, value_lookup=False).run("Name of card 2?")
    assert r.ok and r.attempts == 2
    fix = llm.prompts[1][-1].content
    assert "no such column: nme" in fix and "cards.name" in fix


# --------------------------------------------------------------------------- voting


def test_vote_keeps_the_majority_result(tmp_path):
    db = _cards_db(tmp_path)
    llm = FakeListChatModel(responses=[
        "SELECT count(*) FROM cards",                         # 3
        "SELECT count(*) FROM cards WHERE type = 'commander'",  # 1
        "SELECT count(id) FROM cards WHERE type='commander'",  # 1
    ])
    r = SQLAgent(llm, db, schema_linking=False, value_lookup=False, sql_candidates=3).run("How many commander cards?")
    assert r.rows == [(1,)] and r.votes == 2 and r.n_candidates == 3


def test_vote_prefers_non_empty_and_ignores_failures():
    from docqa.sql import QueryResult

    empty = {"executed": True, "result": QueryResult(["n"], [], False), "sql": "a"}
    full = {"executed": True, "result": QueryResult(["n"], [(1,)], False), "sql": "b"}
    failed = {"executed": False, "error": "boom", "sql": "c"}
    assert vote([empty, full])["sql"] == "b"
    assert vote([failed, full])["sql"] == "b"
    assert vote([failed])["sql"] == "c"


def test_extra_candidates_get_different_instructions(tmp_path):
    db = _cards_db(tmp_path)
    llm = RecordingLLM(responses=["SELECT 1", "SELECT 1", "SELECT 1"], prompts=[])
    SQLAgent(llm, db, schema_linking=False, value_lookup=False, sql_candidates=3).run("q?")
    systems = [p[0].content for p in llm.prompts]
    assert len(systems) == 3 and len(set(systems)) == 3 and "JOIN" in systems[1]


# --------------------------------------------------------------------------- scoring and few-shot


def test_spider_accepts_columns_in_another_order():
    gold = [("Aruba", 193.0, 1960), ("Angola", 1246700.0, 1975)]
    pred = [("Aruba", 1960, 193.0), ("Angola", 1975, 1246700.0)]
    assert bm.results_match(gold, pred, "spider")
    assert bm.results_match(gold[::-1], pred, "spider")              # unordered
    assert not bm.results_match(gold, pred[::-1], "spider", ordered=True)
    assert not bm.results_match(gold, [("Aruba", 1960, 193.0), ("Angola", 1975, 1.0)], "spider")
    assert not bm.results_match(gold, pred, "bird")                  # BIRD's official check is strict


def test_example_bank_retrieves_similar_training_questions(spider, tmp_path):  # noqa: F811
    root = spider / "spider_data"
    train = [
        {"db_id": "pilots", "question": "How many pilots are from Germany?", "query": "SELECT count(*) FROM pilot WHERE country = 'Germany'"},
        {"db_id": "museum", "question": "List the museum names ordered by opening year.", "query": "SELECT name FROM museum ORDER BY open_year"},
        {"db_id": "concert_singer", "question": "How many singers are from Italy?", "query": "SELECT count(*) FROM singer WHERE country = 'Italy'"},
    ]
    (root / "train_spider.json").write_text(json.dumps(train), encoding="utf-8")
    bank = bm.ExampleBank.from_benchmark("spider", spider)
    assert len(bank) == 3
    shots = bank.similar("How many singers are from France?", k=2, exclude_db="concert_singer")
    assert shots[-1] == ("How many pilots are from Germany?", "SELECT count(*) FROM pilot WHERE country = 'Germany'")
    assert all("Italy" not in q for q, _ in shots)  # same database is excluded


def test_evaluate_passes_examples_to_the_agent(spider):  # noqa: F811
    root = spider / "spider_data"
    (root / "train_spider.json").write_text(json.dumps([
        {"db_id": "pilots", "question": "How many pilots are from Germany?", "query": "SELECT count(*) FROM pilot"},
    ]), encoding="utf-8")
    split = bm.locate("spider", spider)
    bank = bm.ExampleBank.from_benchmark("spider", spider)
    llm = RecordingLLM(responses=["SELECT count(*) FROM singer WHERE country = 'France'"], prompts=[])
    agent = SQLAgent(llm, catalog=DatabaseCatalog(split.db_root), db_router="retrieval", schema_linking=False)
    out = next(bm.evaluate(agent, split, split.load()[:1], bank=bank, shots=2))
    assert out.correct
    texts = [m.content for m in llm.prompts[0]]
    assert "How many pilots are from Germany?" in texts


def test_cli_eval_with_shots_and_candidates(spider, tmp_path, capsys, monkeypatch):  # noqa: F811
    import docqa.providers
    from docqa.cli import main

    (spider / "spider_data" / "train_spider.json").write_text(json.dumps([
        {"db_id": "pilots", "question": "How many pilots are from Germany?", "query": "SELECT count(*) FROM pilot"},
    ]), encoding="utf-8")
    monkeypatch.setattr(docqa.providers, "make_llm", lambda s: FakeListChatModel(
        responses=["SELECT count(*) FROM singer WHERE country = 'France'"]))
    monkeypatch.setenv("DOCQA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DOCQA_SQL_SCHEMA_LINKING", "0")
    code = main(["--env-file", str(tmp_path / "no.env"), "eval", "spider", "--data", str(spider), "--oracle-db",
                 "--limit", "1", "--shots", "2", "--sql-candidates", "2"])
    out = capsys.readouterr().out
    assert code == 0 and "Few-shot: 2 similar solved questions" in out
    summary = next((tmp_path / "data" / "eval").glob("*.summary.txt")).read_text(encoding="utf-8")
    assert '"shots": 2' in summary and '"sql_candidates": 2' in summary
