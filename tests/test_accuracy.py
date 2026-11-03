"""Accuracy features for small models: value lookup, error hints, few-shot bank, voting, Spider scoring."""

import json
import sqlite3

import pytest

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


# --------------------------------------------------------------------------- round 2: identifier repair, routing fit, rules

COUNTRY = """
CREATE TABLE country (Code TEXT PRIMARY KEY, Name TEXT, IndepYear INTEGER, SurfaceArea REAL);
CREATE TABLE city (ID INTEGER PRIMARY KEY, Name TEXT, CountryCode TEXT REFERENCES country(Code));
INSERT INTO country VALUES ('ABW','Aruba',NULL,193.0),('AGO','Angola',1975,1246700.0);
INSERT INTO city VALUES (1,'Luanda','AGO'),(2,'Oranjestad','ABW');
"""


def _country_db(tmp_path):
    return open_database(f"sqlite:///{make_db(tmp_path / 'world.sqlite', COUNTRY).as_posix()}")


def test_auto_fix_rewrites_case_and_underscore_variants(tmp_path):
    from docqa.sqlhints import auto_fix_identifiers

    db = _country_db(tmp_path)
    fixed = auto_fix_identifiers(db, "SELECT name, indep_year, surface_area FROM Country WHERE indep_year > 1950")
    assert fixed and "IndepYear" in fixed and "SurfaceArea" in fixed and "indep_year" not in fixed
    # wrong alias: the column lives in the other table of the query
    fixed = auto_fix_identifiers(db, "SELECT T1.Name FROM city AS T1 JOIN country AS T2 ON T1.CountryCode = T2.Code WHERE T1.SurfaceArea > 1000")
    assert fixed and "T2.SurfaceArea" in fixed
    # nothing to fix, or ambiguous: leave it to the model
    assert auto_fix_identifiers(db, "SELECT Name FROM country") is None
    assert auto_fix_identifiers(db, "SELECT population FROM country") is None


def test_agent_repairs_names_without_a_model_call(tmp_path):
    db = _country_db(tmp_path)
    llm = RecordingLLM(responses=["SELECT name FROM country WHERE indep_year = 1975"], prompts=[])
    r = SQLAgent(llm, db, schema_linking=False, value_lookup=False).run("Which country became independent in 1975?")
    assert r.ok and r.rows == [("Angola",)] and r.attempts == 1 and len(llm.prompts) == 1
    assert "IndepYear" in r.sql


def test_rules_tell_the_model_not_to_embellish(tmp_path):
    db = _country_db(tmp_path)
    llm = RecordingLLM(responses=["SELECT 1"], prompts=[])
    SQLAgent(llm, db, schema_linking=False, value_lookup=False).run("q?")
    system = llm.prompts[0][0].content
    assert "no extra columns" in system and "GROUP BY their key column" in system and "bridge table" in system
    assert "SELECT * FROM that one table" not in system  # only for "all the information" questions


def test_select_star_hint_only_for_all_information_questions(tmp_path):
    db = _country_db(tmp_path)
    llm = RecordingLLM(responses=["SELECT 1", "SELECT 1"], prompts=[])
    agent = SQLAgent(llm, db, schema_linking=False, value_lookup=False)
    agent.run("Show all the information about the countries.")
    agent.run("Give the name of the country with code ABW.")
    assert "SELECT * FROM that one table" in llm.prompts[0][0].content
    assert "SELECT * FROM that one table" not in llm.prompts[1][0].content


FLIGHT_A = """
CREATE TABLE airlines (uid INTEGER PRIMARY KEY, Airline TEXT, Country TEXT);
CREATE TABLE airports (City TEXT, AirportCode TEXT PRIMARY KEY, AirportName TEXT);
CREATE TABLE flights (Airline INTEGER, FlightNo INTEGER, SourceAirport TEXT, DestAirport TEXT);
INSERT INTO airports VALUES ('Aberdeen ','APG','Phillips AAF'),('Abilene ','ABI','Municipal');
INSERT INTO airlines VALUES (1,'United Airlines','USA');
"""
FLIGHT_B = """
CREATE TABLE aircraft (aid INTEGER PRIMARY KEY, name TEXT, distance INTEGER);
CREATE TABLE employee (eid INTEGER PRIMARY KEY, name TEXT, salary INTEGER);
CREATE TABLE flight (flno INTEGER PRIMARY KEY, origin TEXT, destination TEXT, aid INTEGER);
INSERT INTO flight VALUES (1,'Los Angeles','Washington D.C.',1);
"""


def test_rerank_prefers_the_database_that_has_the_value(tmp_path):
    from docqa.sql import rerank_by_fit
    from docqa.sqlhints import ValueIndex

    make_db(tmp_path / "dbs" / "flight_2" / "flight_2.sqlite", FLIGHT_A)
    make_db(tmp_path / "dbs" / "flight_1" / "flight_1.sqlite", FLIGHT_B)
    cat = DatabaseCatalog(tmp_path / "dbs")
    q = "How many flights depart from the airports in Aberdeen?"
    ranked, evidence = rerank_by_fit(cat, q, ["flight_1", "flight_2"], lambda n, db: ValueIndex(db))
    assert ranked[0] == "flight_2"
    assert "Aberdeen" in evidence["flight_2"] and "airport" in evidence["flight_2"]


def test_router_prompt_shows_the_evidence(tmp_path):
    make_db(tmp_path / "dbs" / "flight_2" / "flight_2.sqlite", FLIGHT_A)
    make_db(tmp_path / "dbs" / "flight_1" / "flight_1.sqlite", FLIGHT_B)
    cat = DatabaseCatalog(tmp_path / "dbs")
    llm = RecordingLLM(responses=["flight_2", "SELECT count(*) FROM airports WHERE City = 'Aberdeen '"], prompts=[])
    agent = SQLAgent(llm, catalog=cat, schema_linking=False, route_margin=99)  # always ask the model
    r = agent.run("How many airports are in Aberdeen?")
    assert r.database == "flight_2" and r.rows == [(1,)]
    router_prompt = llm.prompts[0][0].content
    assert "Matches the question:" in router_prompt and "Aberdeen" in router_prompt


def test_clear_fit_winner_skips_the_model(tmp_path):
    make_db(tmp_path / "dbs" / "flight_2" / "flight_2.sqlite", FLIGHT_A)
    make_db(tmp_path / "dbs" / "flight_1" / "flight_1.sqlite", FLIGHT_B)
    cat = DatabaseCatalog(tmp_path / "dbs")
    # Were the model asked, it would pick the wrong database.
    llm = RecordingLLM(responses=["SELECT count(*) FROM airports WHERE City = 'Aberdeen '"], prompts=[])
    agent = SQLAgent(llm, catalog=cat, schema_linking=False)
    r = agent.run("How many airports are in Aberdeen?")
    assert r.database == "flight_2" and r.rows == [(1,)]
    assert len(llm.prompts) == 1 and "Pick" not in llm.prompts[0][0].content


def test_close_call_asks_the_model_among_the_top_two_only(tmp_path):
    for name in ("alpha", "beta", "gamma"):
        make_db(tmp_path / "dbs" / name / f"{name}.sqlite", "CREATE TABLE singer (id INTEGER, name TEXT);")
    cat = DatabaseCatalog(tmp_path / "dbs")
    llm = RecordingLLM(responses=["beta", "SELECT count(*) FROM singer"], prompts=[])
    agent = SQLAgent(llm, catalog=cat, schema_linking=False, db_candidates=3)
    db, candidates = agent.route("How many singers are there?")
    assert len(candidates) == 3 and db in candidates[:2]
    router_prompt = llm.prompts[0][0].content
    assert sum(n in router_prompt for n in ("alpha", "beta", "gamma")) == 2


def test_route_only_reports_fit_margins(tmp_path):
    make_db(tmp_path / "dbs" / "flight_2" / "flight_2.sqlite", FLIGHT_A)
    make_db(tmp_path / "dbs" / "flight_1" / "flight_1.sqlite", FLIGHT_B)
    cat = DatabaseCatalog(tmp_path / "dbs")
    ex = bm.Example("1", "flight_2", "How many airports are in Aberdeen?", "SELECT 1")
    out = list(bm.route_only(cat, [ex], k=2))
    assert out[0].margin > 0.15
    assert "Fit lead >= 0.15: 100.0% (1/1) of questions, #1 right in 100.0% (1/1)" in bm.summarize(out, True)


def test_database_name_in_the_question_counts_for_fit(tmp_path):
    from docqa.sqlhints import schema_fit

    ddl = "CREATE TABLE member (member_id TEXT, first_name TEXT, position TEXT);"
    make_db(tmp_path / "student_club.sqlite", ddl)
    db = open_database(f"sqlite:///{tmp_path / 'student_club.sqlite'}")
    q = "How many members of the Student_Club have the position of Vice President?"
    with_name, evidence = schema_fit(q, db, name="student_club")
    without, _ = schema_fit(q, db, name="card_games")
    assert with_name - without > 0.29 and "database name: student_club" in evidence


def test_route_only_uses_fit(tmp_path):
    make_db(tmp_path / "dbs" / "flight_2" / "flight_2.sqlite", FLIGHT_A)
    make_db(tmp_path / "dbs" / "flight_1" / "flight_1.sqlite", FLIGHT_B)
    cat = DatabaseCatalog(tmp_path / "dbs")
    ex = bm.Example("1", "flight_2", "How many airports are in Aberdeen?", "SELECT 1")
    assert next(bm.route_only(cat, [ex], k=2)).correct


def test_bird_evidence_travels_with_the_question():
    assert bm.with_evidence("Youngest patient?", "youngest refers to MAX(Birthday)") == (
        "Youngest patient?\nHint (definitions to follow exactly): youngest refers to MAX(Birthday)")
    assert bm.with_evidence("q?", "") == "q?"


def _big_db(tmp_path):
    rows = ",".join(f"({i})" for i in range(3000))
    make_db(tmp_path / "big.sqlite", f"CREATE TABLE t (x INTEGER); INSERT INTO t VALUES {rows};")
    return open_database(f"sqlite:///{tmp_path / 'big.sqlite'}")


def test_runaway_query_is_stopped(tmp_path):
    import time

    import pytest

    from docqa.sql import run_query

    db = _big_db(tmp_path)
    start = time.monotonic()
    with pytest.raises(TimeoutError, match="simpler query"):
        run_query(db, "SELECT count(*) FROM t a, t b, t c", 10, timeout=0.5)
    assert time.monotonic() - start < 5
    # the guard is removed: the same pooled connection still runs long-ish queries
    assert run_query(db, "SELECT count(*) FROM t a, t b", 10, timeout=None).rows == [(9_000_000,)]


def test_value_index_leaves_no_guard_behind(tmp_path):
    from docqa.sql import run_query

    import time

    rows = ",".join(f"({i}, 'v{i}')" for i in range(3000))
    make_db(tmp_path / "big.sqlite", f"CREATE TABLE t (x INTEGER, name TEXT); INSERT INTO t VALUES {rows};")
    db = open_database(f"sqlite:///{tmp_path / 'big.sqlite'}")
    assert len(ValueIndex(db, time_budget=0.5)) > 0
    time.sleep(0.6)  # the scan's deadline has passed; a leftover guard would interrupt the next query
    assert run_query(db, "SELECT count(*) FROM t a, t b", 10, timeout=None).rows == [(9_000_000,)]


# --------------------------------------------------------------------------- SQL pattern library


def test_every_pattern_has_five_examples_that_run():
    import sqlite3

    from docqa.sqlpatterns import DEMO_SCHEMA, load_patterns

    patterns = load_patterns()
    assert len(patterns) >= 20 and len({p.id for p in patterns}) == len(patterns)
    conn = sqlite3.connect(":memory:")
    conn.executescript(DEMO_SCHEMA.read_text(encoding="utf-8"))
    for p in patterns:
        assert len(p.examples) == 5, p.id
        assert p.use and p.avoid and p.cues, p.id
        for ex in p.examples:
            rows = conn.execute(ex.sql).fetchall()  # every example is valid SQL on the demo schema
            assert rows, f"{p.id}: {ex.question} returned nothing"


def test_patterns_are_retrieved_by_technique():
    from docqa.sqlpatterns import PatternIndex

    idx = PatternIndex()

    def ids(q):
        return [p.id for p in idx.search(q)]

    assert "negation" in ids("Find the names of stadiums that did not have any concert in 2014.")
    assert "percentage" in ids("What percentage of students are female?")
    assert "window-aggregate" in ids("Show me the running total of sales by month")
    assert "window-rank" in ids("List the top 3 products by revenue in each region")
    assert "lag-lead" in ids("Monthly revenue growth compared to the previous month")
    assert "recursive" in ids("Find all employees under manager John at any level")
    assert "set-ops" in ids("Which students have enrolled in both math and physics?")
    assert "top-n" in ids("What is the name of the youngest singer?")
    assert ids("Give all the information about the airline with abbreviation UAL") == []


def test_patterns_reach_the_sql_prompt_and_can_be_turned_off(tmp_path):
    db = _country_db(tmp_path)
    llm = RecordingLLM(responses=["SELECT 1", "SELECT 1"], prompts=[])
    SQLAgent(llm, db, schema_linking=False, value_lookup=False).run("What percentage of countries are in Europe?")
    system = llm.prompts[0][0].content
    assert "SQL techniques that fit this question" in system and "100.0 *" in system
    assert "never its table or column names" in system
    SQLAgent(llm, db, schema_linking=False, value_lookup=False, sql_patterns=False).run("What percentage of countries are in Europe?")
    assert "SQL techniques" not in llm.prompts[1][0].content


def test_own_pattern_files_are_added(tmp_path):
    (tmp_path / "90_fiscal.md").write_text(
        "# Fiscal years\nid: fiscal\ncues: fiscal, fy\nuse: Our fiscal year starts in April.\navoid: Calendar years.\n\n"
        "## Example\nQ: Revenue in FY2024\n```sql\nSELECT 1;\n```\nWhy: April to March.\n",
        encoding="utf-8",
    )
    db = _country_db(tmp_path)
    llm = RecordingLLM(responses=["SELECT 1"], prompts=[])
    SQLAgent(llm, db, schema_linking=False, value_lookup=False, patterns_dir=str(tmp_path)).run("Revenue in fiscal 2024?")
    assert "Our fiscal year starts in April." in llm.prompts[0][0].content


def test_pattern_examples_follow_the_database_dialect(tmp_path):
    from docqa.sqlpatterns import PatternIndex, format_patterns

    found = PatternIndex().search("What are the three most expensive products?")
    oracle = format_patterns(found, dialect="oracle")
    assert "FETCH FIRST 3 ROWS ONLY" in oracle and "LIMIT 3" not in oracle
    assert "TOP 3" in format_patterns(found, dialect="tsql")
    assert "LIMIT 3" in format_patterns(found)  # SQLite as written


def test_eval_docs_context_comes_from_the_routed_database_folder(tmp_path, embeddings):
    from conftest import make_docx, make_pdf
    from docqa.config import Settings
    from docqa.sqlpatterns import PatternIndex

    docs = tmp_path / "docs"
    (docs / "european_football").mkdir(parents=True)
    (docs / "card_games").mkdir()
    (docs / "sql_manual").mkdir()
    make_pdf(docs / "european_football" / "Country.pdf", ["Country.name: full country name, e.g. Slovakia; code SVK."])
    make_docx(docs / "card_games" / "cards.docx", ["cards.rarity: common, uncommon, rare or mythic."])
    (docs / "sql_manual" / "window.md").write_text(
        "# Window functions\n\nROW_NUMBER() OVER (PARTITION BY ...) numbers rows within each partition.\n", encoding="utf-8")
    s = Settings(docs_dir=docs, data_dir=tmp_path / "data", chunk_size=200, chunk_overlap=40)
    ctx = bm.DocsContext(s, docs, top_k=3, patterns=PatternIndex(), embeddings=embeddings, slim=False)
    ctx.reference_folders = ["sql_manual"]
    text = ctx.for_database("european_football", "Which country has the most matches? Rank the top 2 in each league")
    assert "Slovakia" in text and "Country.pdf" in text
    assert "rarity" not in text  # another database's documents never leak in
    assert "ROW_NUMBER" in text  # the reference page for the ranking technique
    assert ctx.for_database("unknown_db", "anything") == "" or "rarity" not in ctx.for_database("unknown_db", "x")


# --------------------------------------------------------------------------- slim docs + answer checks


def test_slim_docs_keep_only_value_notes_and_manual_comes_on_error(tmp_path, embeddings):
    from docqa.config import Settings
    from docqa.sqlpatterns import PatternIndex

    docs = tmp_path / "docs"
    (docs / "shop").mkdir(parents=True)
    (docs / "sql_manual").mkdir()
    (docs / "shop" / "dictionary.md").write_text(
        "# orders\n\norders.customer_name: the name of the customer who placed the order.\n"
        "orders.status: order status code; 'S' stands for shipped, 'C' means cancelled.\n", encoding="utf-8")
    (docs / "sql_manual" / "dates.md").write_text(
        "# Date functions\n\nstrftime('%Y', date) returns the year of a date as text.\n", encoding="utf-8")
    s = Settings(docs_dir=docs, data_dir=tmp_path / "data", chunk_size=400, chunk_overlap=40)
    ctx = bm.DocsContext(s, docs, top_k=3, patterns=PatternIndex(), embeddings=embeddings)
    ctx.reference_folders = ["sql_manual"]
    text = ctx.for_database("shop", "How many orders were shipped?")
    assert "stands for shipped" in text
    assert "customer_name" not in text  # a plain column description is what distracted the model
    assert "strftime" not in text  # the manual is not shown up front any more
    help_text = ctx.help_for_error("no such function: YEAR", "SELECT YEAR(o.date) FROM orders o")
    assert "strftime" in help_text
    assert ctx.help_for_error("no such column: o.foo", "SELECT o.foo FROM orders o") == ""


def test_result_checks_flag_suspicious_answers():
    from docqa.sqlchecks import result_problems

    assert result_problems("How many?", "SELECT 1", ["n"], [])[0][0] == "empty"
    assert result_problems("Who?", "SELECT x FROM t", ["x"], [(None,), (None,)])[0][0] == "nulls"
    dup = result_problems("List the names", "SELECT a.name FROM a JOIN b ON 1", ["name"], [("x",)] * 6)
    assert dup[0][0] == "dupes"
    assert not result_problems("List the names", "SELECT DISTINCT name FROM a", ["name"], [("x",)] * 6)
    assert result_problems("What percentage of cards are rare?", "SELECT ...", ["p"], [(340.0,)])[0][0] == "range"
    assert not result_problems("What percentage of cards are rare?", "SELECT ...", ["p"], [(34.0,)])
    assert result_problems("Ages", "SELECT ...", ["age"], [(-3,)])[0][0] == "range"
    assert not result_problems("How many members?", "SELECT COUNT(*) FROM m", ["n"], [(12,)])


def test_answer_check_reasks_once_and_keeps_a_correct_empty_answer(tmp_path):
    db_path = tmp_path / "shop.db"
    con = sqlite3.connect(db_path)
    con.executescript("CREATE TABLE orders (id INTEGER, status TEXT); INSERT INTO orders VALUES (1, 'S'), (2, 'C');")
    con.commit(); con.close()
    from docqa.sql import SQLAgent, open_database

    # 1st query is empty (wrong value) -> re-asked -> 2nd query uses the right code
    llm = FakeListChatModel(responses=[
        "```sql\nSELECT id FROM orders WHERE status = 'shipped'\n```",
        "```sql\nSELECT id FROM orders WHERE status = 'S'\n```",
    ])
    agent = SQLAgent(llm, db=open_database(f"sqlite:///{db_path}"), schema_linking=False, value_lookup=False,
                     sql_patterns=False)
    r = agent.run("Which orders were shipped?")
    assert r.rows == [(1,)] and r.attempts == 2 and r.error is None
    assert any("Answer check" in e for _, e in r.failures)

    # empty is really the answer: the model returns the same query, which is accepted (no loop)
    same = "```sql\nSELECT id FROM orders WHERE status = 'X'\n```"
    agent = SQLAgent(FakeListChatModel(responses=[same, same]), db=open_database(f"sqlite:///{db_path}"),
                     schema_linking=False, value_lookup=False, sql_patterns=False)
    r = agent.run("Which orders have status X?")
    assert r.rows == [] and r.attempts == 2 and r.error is None and r.executed

    # the re-ask fails to run every time -> the first query that ran is kept
    agent = SQLAgent(FakeListChatModel(responses=[same, "```sql\nSELECT nope FROM orders\n```"]),
                     db=open_database(f"sqlite:///{db_path}"), schema_linking=False, value_lookup=False,
                     sql_patterns=False, max_attempts=2)
    r = agent.run("Which orders have status X?")
    assert r.executed and r.error is None and "status = 'X'" in r.sql

    # --no-checks: one query, no re-ask
    agent = SQLAgent(FakeListChatModel(responses=[same]), db=open_database(f"sqlite:///{db_path}"),
                     schema_linking=False, value_lookup=False, sql_patterns=False, answer_checks=False)
    assert agent.run("Which orders have status X?").attempts == 1


def test_dry_run_catches_unknown_names_before_running(tmp_path):
    db_path = tmp_path / "s.db"
    con = sqlite3.connect(db_path); con.execute("CREATE TABLE t (a INTEGER)"); con.commit(); con.close()
    from docqa.sql import open_database
    from docqa.sqlchecks import dry_run

    db = open_database(f"sqlite:///{db_path}")
    dry_run(db, "SELECT a FROM t")
    with pytest.raises(Exception, match="no such column"):
        dry_run(db, "SELECT b FROM t")


def test_review_rewrites_once_when_the_model_says_wrong(tmp_path):
    db_path = tmp_path / "r.db"
    con = sqlite3.connect(db_path)
    con.executescript("CREATE TABLE p (id INTEGER, name TEXT, age INTEGER); INSERT INTO p VALUES (1,'a',30),(2,'b',40);")
    con.commit(); con.close()
    from docqa.sql import SQLAgent, open_database
    from docqa.sqlchecks import parse_review

    assert parse_review("OK") == (True, "")
    assert parse_review("WRONG: the question asks only for names")[0] is False
    llm = FakeListChatModel(responses=[
        "```sql\nSELECT id, name FROM p\n```",
        "WRONG - the question asks only for the names",
        "```sql\nSELECT name FROM p\n```",
    ])
    agent = SQLAgent(llm, db=open_database(f"sqlite:///{db_path}"), schema_linking=False, value_lookup=False,
                     sql_patterns=False, review=True)
    r = agent.run("List the names")
    assert r.columns == ["name"] and r.attempts == 2
