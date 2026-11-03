"""Checks on a query that ran, so a plausible-looking wrong answer is sent back once.

Syntax and safety are already checked before a query runs (``sql.check_read_only``),
and database errors already drive a retry. But on BIRD every wrong answer of the best
run was valid SQL that ran: it returned nothing, a column of NULLs, rows multiplied by
a join, or a percentage of 340. These checks look at the *result* and, when it looks
off, tell the model what is suspicious. Each kind of check fires at most once per
question, and if the second query is no better the first one is kept, so a correct
empty answer is never looped on.

* ``dry_run``: ``EXPLAIN`` the query first (cheap; catches unknown names and bad
  syntax before a heavy query starts).
* ``result_problems``: empty result, all-NULL column, duplicated rows, a percentage
  outside 0-100 or a negative count.
* ``REVIEW_PROMPT``: optional self-review: the model sees the question, the SQL and
  the first rows and answers OK or WRONG with a reason (off by default).
"""

from __future__ import annotations

import re

from langchain_core.prompts import ChatPromptTemplate

_EXPLAIN = {"sqlite": "EXPLAIN QUERY PLAN ", "postgres": "EXPLAIN ", "postgresql": "EXPLAIN ",
            "mysql": "EXPLAIN ", "mariadb": "EXPLAIN ", "duckdb": "EXPLAIN "}


def dry_run(db, sql: str) -> None:
    """``EXPLAIN`` the query on a read-only connection; raises the database's error.

    Databases without a plain ``EXPLAIN`` (Oracle, SQL Server) are skipped: the query
    itself still runs under the same checks.
    """
    prefix = _EXPLAIN.get(db._engine.dialect.name)
    if not prefix:
        return
    with db._engine.connect() as conn:
        try:
            conn.execution_options(no_parameters=True).exec_driver_sql(prefix + sql).fetchall()
        finally:
            conn.rollback()


_PERCENT_Q = re.compile(r"\b(percent(age)?|%|proportion)\b", re.I)
_COUNT_COL = re.compile(r"(^|_|\b)(count|num|number|total|age|cnt)(_|\b|$)", re.I)
_SAME_AGAIN = "If this really is the answer, return the same query unchanged."


def result_problems(question: str, sql: str, columns: list[str], rows: list[tuple]) -> list[tuple[str, str]]:
    """``(kind, message)`` for each thing that looks wrong about a result, most telling first."""
    problems: list[tuple[str, str]] = []
    if not rows:
        return [("empty", "The query ran but returned no rows. Check every filter value against the "
                          "database values in the context (spelling, case, codes, date format) and that "
                          "no join or WHERE condition drops the rows. " + _SAME_AGAIN)]
    for i, name in enumerate(columns):
        if all(r[i] is None for r in rows):
            problems.append(("nulls", f"Column {name!r} is NULL in every row: check it is the right column, "
                                      "that the join keys match and that the aggregate covers matching rows. "
                                      + _SAME_AGAIN))
            break
    distinct = len(set(rows))
    if len(rows) >= 4 and distinct <= len(rows) // 2 and not re.search(r"\b(distinct|group\s+by)\b", sql, re.I):
        problems.append(("dupes", f"The query returned {len(rows)} rows but only {distinct} are different: a join "
                                  "probably multiplies rows. Use DISTINCT or join on the right keys, unless "
                                  "repeated rows are what the question asks for. " + _SAME_AGAIN))
    numbers = [(columns[i], v) for r in rows[:20] for i, v in enumerate(r)
               if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if _PERCENT_Q.search(question) and len(rows) == 1 and numbers:
        bad = [v for _, v in numbers if not -0.0001 <= v <= 100.0001]
        if bad and len(bad) == len(numbers):
            problems.append(("range", f"The question asks for a percentage but the result is {bad[0]}, outside "
                                      "0-100: check the numerator and denominator (multiply by 100 once, cast to "
                                      "REAL before dividing, divide by the right total). " + _SAME_AGAIN))
    negative = [(c, v) for c, v in numbers if v < 0 and _COUNT_COL.search(c or "")]
    if negative:
        problems.append(("range", f"Column {negative[0][0]!r} is {negative[0][1]}, a negative count or age: "
                                  "check the formula and the order of the subtraction. " + _SAME_AGAIN))
    return problems


REVIEW_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You check SQL answers. Given a question, the SQL that was run and the first rows it "
            "returned, decide whether the query answers exactly what was asked: the right rows, "
            "the right columns (no extra and none missing), the right aggregation and formula. "
            "Reply with OK, or with WRONG followed by one sentence saying what to change. "
            "Reply OK unless you are sure something is wrong.",
        ),
        ("human", "Question: {question}\n\nSQL:\n```sql\n{sql}\n```\n\nFirst rows ({n_rows} in total):\n{table}"),
    ]
)


def parse_review(text: str) -> tuple[bool, str]:
    """``(ok, reason)`` from a reviewer reply; anything that is not a clear WRONG counts as OK."""
    t = text.strip()
    m = re.match(r"^\W*wrong\b\W*(.*)", t, re.I | re.S)
    if not m:
        return True, ""
    return False, (m.group(1).strip().splitlines() or ["the reviewer found a problem"])[0][:300]


# ------------------------------------------------------------------ documents, slimmed for SQL

_VALUE_CUE = re.compile(
    r"(value[_ ]description|stands for|\bmeans?\b|\brefers? to\b|\bindicates?\b|\bunits?\b|\bcodes?\b|"
    r"\bwhere\b.*\bis\b|\b\w+\s*[:=]\s*['\"]?\w+['\"]?\s*(?:means|=|:|-)\s*\w|\be\.g\.|for example|"
    r"\bin (?:usd|dollars|euros?|kg|km|cm|seconds|minutes|hours|days|years)\b|"
    r"\b\w{1,12}\s*[:=]\s*[^;:=,]{1,40}[;,]\s*['\"]?\w{1,12}['\"]?\s*[:=]|"  # code lists: F: female; M: male
    r"['\"]+\w[\w ]{0,20}['\"]+\s*[:=])",
    re.I,
)
_HEADER = re.compile(r"original_column_name|^column_name\b", re.I)
_EMPTY = re.compile(r"^\W*value[_ ]description\W*$", re.I)  # a dictionary row with no value notes


def value_notes(texts: list[str], max_chars: int = 700) -> str:
    """Only the lines of the passages that explain values (codes, units, meanings).

    A data dictionary also describes every column, which the schema already shows; a
    small model treats those extra descriptions as a reason to add columns and joins.
    """
    kept: list[str] = []
    seen: set[str] = set()
    used = 0
    for text in texts:
        for line in text.splitlines():
            line = line.strip()
            if len(line) < 8 or _HEADER.search(line) or _EMPTY.match(line) or not _VALUE_CUE.search(line):
                continue
            key = line.lower()
            if key in seen:
                continue
            if used + len(line) > max_chars:
                return "\n".join(kept)
            seen.add(key)
            kept.append(line)
            used += len(line)
    return "\n".join(kept)


_MANUAL_ERROR = re.compile(r"function|date|time|strftime|julianday|syntax|misuse|window|aggregate|cast|"
                           r"datatype|mismatch|percent|over\b|partition", re.I)


def manual_query(error: str, sql: str) -> str:
    """What to look up in an SQL manual for this error, or "" when a manual would not help."""
    if not _MANUAL_ERROR.search(error):
        return ""
    funcs = sorted({f.upper() for f in re.findall(r"\b([A-Za-z_]+)\s*\(", sql)} - {"SELECT", "IN", "EXISTS"})
    return f"{error.splitlines()[0][:200]} {' '.join(funcs[:6])}".strip()
