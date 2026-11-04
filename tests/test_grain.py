"""Grain plan and its checks: a WHERE that undoes a LEFT JOIN, totals multiplied by a join of two
many-row tables, key paths with two roles. Generic schemas, nothing from any benchmark."""

import sqlite3

from langchain_core.language_models import FakeListChatModel

from docqa.sql import SQLAgent, open_database
from docqa.sqlchecks import fanout_joins, join_paths, optional_side_filters, relations


def _shop(tmp_path, fks=True):
    path = tmp_path / ("shop.db" if fks else "shop_nofk.db")
    ref = (lambda t: f" REFERENCES {t}") if fks else (lambda t: "")
    con = sqlite3.connect(path)
    con.executescript(f"""
        CREATE TABLE people (person_id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE orders (order_id INTEGER PRIMARY KEY, status TEXT,
                             placed_by INTEGER{ref('people(person_id)')}, paid_by INTEGER{ref('people(person_id)')});
        CREATE TABLE lines (line_id INTEGER PRIMARY KEY, order_id INTEGER{ref('orders(order_id)')}, amount INTEGER);
        CREATE TABLE payments (payment_id INTEGER PRIMARY KEY, order_id INTEGER{ref('orders(order_id)')},
                               state TEXT, amount INTEGER);
        INSERT INTO people VALUES (1, 'Ann'), (2, 'Bo');
        INSERT INTO orders VALUES (1, 'Done', 1, 2), (2, 'Done', 2, 2), (3, 'Open', 1, 1);
        INSERT INTO lines VALUES (1, 1, 100), (2, 1, 50), (3, 2, 70), (4, 3, 10);
        INSERT INTO payments VALUES (1, 1, 'ok', 100), (2, 1, 'ok', 50), (3, 2, 'void', 70);
    """)
    con.commit(); con.close()
    return open_database(f"sqlite:///{path}")


def test_optional_side_filter_found_only_in_where():
    bad = ("SELECT o.order_id FROM orders o LEFT JOIN payments p ON p.order_id = o.order_id "
           "WHERE o.status = 'Done' AND p.state = 'ok'")
    assert optional_side_filters(bad, "sqlite") == [("p.state = 'ok'", "payments")]
    on = ("SELECT o.order_id FROM orders o LEFT JOIN payments p ON p.order_id = o.order_id AND p.state = 'ok' "
          "WHERE o.status = 'Done'")
    assert optional_side_filters(on, "sqlite") == []
    anti = "SELECT o.order_id FROM orders o LEFT JOIN payments p ON p.order_id = o.order_id WHERE p.payment_id IS NULL"
    assert optional_side_filters(anti, "sqlite") == []
    cte = ("WITH paid AS (SELECT order_id, SUM(amount) s FROM payments WHERE state = 'ok' GROUP BY order_id) "
           "SELECT o.order_id, COALESCE(pd.s, 0) FROM orders o LEFT JOIN paid pd ON pd.order_id = o.order_id "
           "WHERE o.status = 'Done'")
    assert optional_side_filters(cte, "sqlite") == []
    on_cte = cte + " AND pd.s > 0"
    assert optional_side_filters(on_cte, "sqlite") == [("pd.s > 0", "paid")]
    kept = cte + " AND COALESCE(pd.s, 0) < 100"
    assert optional_side_filters(kept, "sqlite") == []


def test_unqualified_optional_column_resolved_with_the_schema(tmp_path):
    db = _shop(tmp_path)
    sql = ("SELECT orders.order_id FROM orders LEFT JOIN payments ON payments.order_id = orders.order_id "
           "WHERE state = 'ok'")
    assert optional_side_filters(sql, "sqlite", db) == [("state = 'ok'", "payments")]


def test_fanout_of_two_many_row_tables(tmp_path):
    db = _shop(tmp_path)
    both = ("SELECT o.order_id, SUM(l.amount), SUM(p.amount) FROM orders o JOIN lines l ON l.order_id = o.order_id "
            "LEFT JOIN payments p ON p.order_id = o.order_id GROUP BY o.order_id")
    assert fanout_joins(db, both, "sqlite") == [("lines", "payments", "orders")]
    one = "SELECT o.order_id, SUM(l.amount) FROM orders o JOIN lines l ON l.order_id = o.order_id GROUP BY o.order_id"
    assert fanout_joins(db, one, "sqlite") == []
    totals = ("WITH b AS (SELECT order_id, SUM(amount) s FROM lines GROUP BY order_id), "
              "c AS (SELECT order_id, SUM(amount) s FROM payments GROUP BY order_id) "
              "SELECT o.order_id, b.s, COALESCE(c.s, 0) FROM orders o JOIN b ON b.order_id = o.order_id "
              "LEFT JOIN c ON c.order_id = o.order_id")
    assert fanout_joins(db, totals, "sqlite") == []
    distinct = ("SELECT COUNT(DISTINCT o.order_id) FROM orders o JOIN lines l ON l.order_id = o.order_id "
                "JOIN payments p ON p.order_id = o.order_id")
    assert fanout_joins(db, distinct, "sqlite") == []


def test_keys_inferred_by_name_and_roles_named(tmp_path):
    nofk = _shop(tmp_path, fks=False)
    rels = relations(nofk)
    assert ("lines", "order_id", "orders", "order_id") in rels and ("payments", "order_id", "orders", "order_id") in rels
    paths = join_paths(_shop(tmp_path))
    assert "lines.order_id = orders.order_id  (many lines per orders)" in paths
    assert "orders has 2 keys to people (paid_by, placed_by)" in paths


def test_grain_check_resends_a_left_join_undone_by_where(tmp_path):
    db = _shop(tmp_path)
    bad = ("```sql\nSELECT o.order_id, COALESCE(SUM(p.amount), 0) FROM orders o LEFT JOIN payments p "
           "ON p.order_id = o.order_id WHERE o.status = 'Done' AND p.state = 'ok' GROUP BY o.order_id\n```")
    good = ("```sql\nSELECT o.order_id, COALESCE(SUM(p.amount), 0) FROM orders o LEFT JOIN payments p "
            "ON p.order_id = o.order_id AND p.state = 'ok' WHERE o.status = 'Done' GROUP BY o.order_id\n```")
    agent = SQLAgent(FakeListChatModel(responses=["plan", bad, good]), db=db, value_lookup=False,
                     sql_patterns=False, answer_checks=False, grain_plan=True, max_attempts=3)
    r = agent.run("Settled payments per completed order, unpaid orders included")
    assert sorted(r.rows) == [(1, 150), (2, 0)] and r.attempts == 2
    assert "optional side of a LEFT JOIN" in r.failures[0][1]

    off = SQLAgent(FakeListChatModel(responses=["plan", bad]), db=db, value_lookup=False,
                   sql_patterns=False, answer_checks=False, grain_plan=False, max_attempts=3)
    assert off.run("Settled payments per completed order").attempts == 1


def test_grain_check_accepts_the_same_query_back(tmp_path):
    db = _shop(tmp_path)
    q = ("```sql\nSELECT o.order_id FROM orders o LEFT JOIN payments p ON p.order_id = o.order_id "
         "WHERE p.state = 'ok'\n```")
    agent = SQLAgent(FakeListChatModel(responses=["plan", q, q]), db=db, value_lookup=False,
                     sql_patterns=False, answer_checks=False, grain_plan=True, max_attempts=3)
    r = agent.run("Orders with an ok payment")
    assert r.ok and r.attempts == 2 and sorted(r.rows) == [(1,), (1,)]


def test_grain_plan_prompt_has_paths_and_grain(tmp_path):
    db = _shop(tmp_path)
    agent = SQLAgent(FakeListChatModel(responses=[]), db=db, value_lookup=False, sql_patterns=False,
                     grain_plan=True)
    agent.schema_text("", "orders")
    base = agent._base({"question": "q", "schema": "S", "paths": agent._paths_text()})
    text = "".join(m.content for m in __import__("docqa.sql", fromlist=["LINK_PROMPT"]).LINK_PROMPT.format_messages(**base))
    assert "Grain:" in text and "Key paths (from the database)" in text and "2 keys to people" in text
    off = SQLAgent(FakeListChatModel(responses=[]), db=db, value_lookup=False, sql_patterns=False)
    assert off._paths_text() == "" and off._base({"question": "q", "schema": "S"})["grain"] == ""


def test_fanout_needs_the_shared_parent_key_in_the_join(tmp_path):
    db = _shop(tmp_path)
    # lines and payments joined on an unrelated column: not a sibling fan-out through orders
    sql = "SELECT SUM(l.amount) FROM lines l JOIN payments p ON p.amount = l.amount"
    assert fanout_joins(db, sql, "sqlite") == []


def test_connected_run_writes_agent_output_only(tmp_path, monkeypatch, capsys):
    import json

    import docqa.providers
    from docqa import cli

    db = _shop(tmp_path)
    qs = tmp_path / "q.jsonl"
    qs.write_text(json.dumps({"id": 7, "question": "How many orders?", "gold_sql": "SECRET"}) + "\n"
                  + json.dumps({"id": 8, "question": "skip me"}) + "\n", encoding="utf-8")
    ids = tmp_path / "ids.txt"
    ids.write_text("7\n", encoding="utf-8")
    monkeypatch.setattr(docqa.providers, "make_llm", lambda s: FakeListChatModel(
        responses=["plan", "```sql\nSELECT COUNT(*) FROM orders\n```"]))
    out = tmp_path / "out.jsonl"
    path = str(tmp_path / "shop.db")
    assert cli.main(["connected-run", "--db", path, "--questions", str(qs), "--ids", str(ids),
                     "--grain", "--no-patterns", "--out", str(out)]) == 0
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["id"] == "7" and rows[0]["executed"] and "SECRET" not in out.read_text()
    assert rows[0]["pred_sql"].strip().upper().startswith("SELECT COUNT(*)")
