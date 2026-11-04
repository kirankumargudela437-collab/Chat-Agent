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

_EXPLAIN = {"sqlite": "EXPLAIN QUERY PLAN {sql}", "postgresql": "EXPLAIN {sql}", "mysql": "EXPLAIN {sql}",
            "mariadb": "EXPLAIN {sql}", "duckdb": "EXPLAIN {sql}",
            # Oracle's EXPLAIN PLAN writes to PLAN_TABLE (a read-only user may not have one); an
            # empty wrapped query makes Oracle resolve every name without fetching a row
            "oracle": "SELECT * FROM ({sql}) q WHERE 1 = 0"}


def dry_run(db, sql: str) -> None:
    """Check names and syntax on the database without running the query; raises its error.

    ``EXPLAIN`` where the database has a read-only one; on Oracle the query wrapped in
    ``WHERE 1 = 0``. SQL Server (a CTE cannot be wrapped) is skipped: the query itself
    still runs under the same checks.
    """
    template = _EXPLAIN.get(db._engine.dialect.name)
    if not template:
        return
    with db._engine.connect() as conn:
        try:
            res = conn.execution_options(no_parameters=True).exec_driver_sql(template.format(sql=sql))
            if res.returns_rows:
                res.fetchall()
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


# ------------------------------------------------------------------ mistakes the database accepts


def _parse(sql: str, dialect: str):
    import sqlglot

    try:
        return sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return None


def _alias_map(tree) -> dict[str, str]:
    from sqlglot import exp

    out = {}
    for t in tree.find_all(exp.Table):
        out[(t.alias_or_name or "").lower()] = t.name
        out[t.name.lower()] = t.name
    return out


def self_comparisons(sql: str, dialect: str) -> list[str]:
    """Conditions that compare a column with itself (``ON st.setCode = st.setCode``): always true,
    so the database runs them happily and the join multiplies every row."""
    from sqlglot import exp

    tree = _parse(sql, dialect)
    if tree is None:
        return []
    found = []
    for eq in tree.find_all(exp.EQ):
        a, b = eq.left, eq.right
        if isinstance(a, exp.Column) and isinstance(b, exp.Column) \
                and a.name.lower() == b.name.lower() and (a.table or "").lower() == (b.table or "").lower():
            found.append(eq.sql(dialect=dialect))
    return found


def join_keys(db, tables: list[str]) -> str:
    """How these tables join: foreign keys, else columns with the same name in two tables."""
    md = db._metadata.tables
    lines = []
    for t in tables:
        if t not in md:
            continue
        for fk in md[t].foreign_keys:
            if fk.column.table.name in tables:
                lines.append(f"{t}.{fk.parent.name} = {fk.column.table.name}.{fk.column.name}")
    if not lines:
        for i, a in enumerate(tables):
            for b in tables[i + 1:]:
                if a in md and b in md:
                    shared = {c.name for c in md[a].columns} & {c.name for c in md[b].columns}
                    lines += [f"{a}.{c} = {b}.{c}" for c in sorted(shared)]
    return "; ".join(dict.fromkeys(lines))


def tables_in(sql: str, dialect: str) -> list[str]:
    from sqlglot import exp

    tree = _parse(sql, dialect)
    return list(dict.fromkeys(t.name for t in tree.find_all(exp.Table))) if tree is not None else []


_NOT_CATEGORY = re.compile(r"^\s*[-+]?\d[\d.,:\- /T]*\s*$")  # numbers and dates are not codes


def missing_values(db, sql: str, dialect: str, max_values: int = 40, timeout: float = 5.0) -> list[str]:
    """For ``column = 'text'`` filters whose text never occurs in that column, a line naming the
    values that do (all of them when there are few, else the closest ones).

    'Slovakia' in a column that stores 'SVK' runs fine and returns nothing (or NULL);
    the model can map the name to the code once it sees the codes."""
    import difflib

    from sqlalchemy import text
    from sqlglot import exp

    tree = _parse(sql, dialect)
    if tree is None:
        return []
    aliases = _alias_map(tree)
    tables = [t for t in dict.fromkeys(aliases.values())]
    md = db._metadata.tables
    checks = []
    for node in tree.find_all(exp.EQ, exp.In):
        if isinstance(node, exp.EQ):
            col, lits = node.left, [node.right]
            if isinstance(lits[0], exp.Column) and isinstance(col, exp.Literal):
                col, lits = lits[0], [col]
        else:
            col, lits = node.this, node.expressions
        if not isinstance(col, exp.Column):
            continue
        lits = [lit.this for lit in lits if isinstance(lit, exp.Literal) and lit.is_string
                and not _NOT_CATEGORY.match(lit.this)]
        if not lits:
            continue
        table = aliases.get((col.table or "").lower()) if col.table else None
        if table is None:
            owners = [t for t in tables if t in md and col.name in md[t].columns]
            table = owners[0] if len(owners) == 1 else None
        if table is None or table not in md or col.name not in md[table].columns:
            continue
        checks.append((table, md[table].columns[col.name].name, lits))
    from .sqlhints import _guard
    import time

    q = db._engine.dialect.identifier_preparer.quote
    out = []
    with db._engine.connect() as conn:
        _guard(conn, time.monotonic() + timeout)
        try:
            for table, column, lits in checks:
                t, c = q(table), q(column)
                for lit in lits:
                    hit = conn.execute(text(f"SELECT 1 FROM {t} WHERE {c} = :v LIMIT 1"), {"v": lit}).first()
                    if hit:
                        continue
                    values = [r[0] for r in conn.execute(
                        text(f"SELECT DISTINCT {c} FROM {t} WHERE {c} IS NOT NULL LIMIT 2000")).fetchall()]
                    if not values:
                        continue
                    shown = [str(v) for v in values]
                    if len(shown) > max_values:
                        lower = {s.lower(): s for s in shown}
                        close = difflib.get_close_matches(lit.lower(), list(lower), 8, 0.5)
                        shown = [lower[x] for x in close] + [s for s in shown if lit.lower() in s.lower()][:8]
                        shown = list(dict.fromkeys(shown))
                        if not shown:
                            continue  # many values and none close: a lookup will not help
                        label = "values like it"
                    else:
                        label = "its values"
                    out.append(f"'{lit}' never occurs in {table}.{column}; {label}: "
                               + ", ".join(f"'{s}'" for s in shown[:max_values]))
        except Exception as e:  # never let a lookup break the loop (timeouts, odd types)
            import logging

            logging.getLogger(__name__).debug("value check failed: %s", e)
        finally:
            _guard(conn, None)
            conn.rollback()
    return out


# -- grain: joins that multiply rows, and LEFT JOINs undone by WHERE ---------------------


def relations(db) -> list[tuple[str, str, str, str]]:
    """``(child, child_column, parent, parent_column)`` for every key between two tables.

    Declared foreign keys first; a table that declares none is linked by name: a column
    that is the whole primary key of another table (``order_id`` in ``payments`` when
    ``orders`` has primary key ``order_id``)."""
    md = db._metadata.tables
    out = []
    pk_of = {}
    for name, t in md.items():
        cols = [c.name for c in t.primary_key.columns]
        if len(cols) == 1:
            pk_of.setdefault(cols[0].lower(), []).append(name)
    for name, t in md.items():
        fks = sorted(t.foreign_keys, key=lambda fk: (fk.parent.name, fk.column.table.name))
        for fk in fks:
            out.append((name, fk.parent.name, fk.column.table.name, fk.column.name))
        if fks:
            continue
        own = {c.name.lower() for c in t.primary_key.columns}
        for c in t.columns:
            if c.name.lower() in own and len(own) == 1:
                continue
            for parent in pk_of.get(c.name.lower(), []):
                if parent != name:
                    out.append((name, c.name, parent, c.name))
    return list(dict.fromkeys(out))


def _unique_cols(db, table: str) -> set[str]:
    t = db._metadata.tables.get(table)
    if t is None:
        return set()
    out = set()
    pk = [c.name.lower() for c in t.primary_key.columns]
    if len(pk) == 1:
        out.add(pk[0])
    for con in getattr(t, "constraints", ()):
        cols = [c.name.lower() for c in getattr(con, "columns", ())]
        if type(con).__name__ == "UniqueConstraint" and len(cols) == 1:
            out.add(cols[0])
    for ix in getattr(t, "indexes", ()):
        cols = [c.name.lower() for c in ix.columns]
        if ix.unique and len(cols) == 1:
            out.add(cols[0])
    return out


def join_paths(db, tables: list[str] | None = None) -> str:
    """The key paths between ``tables`` (all tables when None), one line each, and a warning
    where one table has two keys to the same table (two roles: who ordered and who pays):
    the database cannot say which one a question means, so the plan has to."""
    rels = relations(db)
    keep = set(tables) if tables else None
    lines, roles = [], {}
    for child, ccol, parent, pcol in rels:
        if keep is not None and not (child in keep and parent in keep):
            continue
        many = ccol.lower() not in _unique_cols(db, child)
        lines.append(f"{child}.{ccol} = {parent}.{pcol}  ({'many' if many else 'one'} {child} per {parent})")
        roles.setdefault((child, parent), []).append(ccol)
    for (child, parent), cols in roles.items():
        if len(cols) > 1:
            lines.append(f"{child} has {len(cols)} keys to {parent} ({', '.join(cols)}): each is a different "
                         f"role; use the one the question means, and join {parent} once per role it needs.")
    return "\n".join(lines)


def _conjuncts(node):
    from sqlglot import exp

    if isinstance(node, exp.And):
        return _conjuncts(node.left) + _conjuncts(node.right)
    if isinstance(node, exp.Paren):
        return _conjuncts(node.this)
    return [node]


def _keeps_nulls(cond, alias: str) -> bool:
    """True when the condition still lets the unmatched (all-NULL) rows of ``alias`` through:
    ``x IS NULL``, ``... OR x IS NULL``, ``COALESCE(x, ..)``, ``IFNULL``/``NVL``."""
    from sqlglot import exp

    for n in cond.find_all(exp.Is):
        if isinstance(n.expression, exp.Null) or isinstance(n.this, exp.Null):
            return True
    for n in cond.find_all(exp.Coalesce):
        return True
    return any(n.sql().lower().startswith(("ifnull(", "nvl(", "isnull(")) for n in cond.find_all(exp.Anonymous))


def optional_side_filters(sql: str, dialect: str, db=None) -> list[tuple[str, str]]:
    """``(condition, table)`` for each WHERE condition on the optional (right) side of a LEFT
    JOIN. ``WHERE p.status = 'Settled'`` after ``LEFT JOIN payments p`` drops every row that
    had no payment, so the LEFT JOIN quietly becomes an INNER JOIN."""
    from sqlglot import exp

    tree = _parse(sql, dialect)
    if tree is None:
        return []
    md = db._metadata.tables if db is not None else {}
    found = []
    for select in tree.find_all(exp.Select):
        optional: dict[str, str] = {}
        for join in select.args.get("joins") or []:
            if (join.side or "").upper() != "LEFT":
                continue
            src = join.this
            alias = (src.alias_or_name or "").lower()
            if alias:
                optional[alias] = src.name if isinstance(src, exp.Table) else alias
        where = select.args.get("where")
        if not optional or where is None:
            continue
        for cond in _conjuncts(where.this):
            hits = set()
            for col in cond.find_all(exp.Column):
                if col.find_ancestor(exp.Select) is not select:
                    continue  # inside a subquery of the condition
                t = (col.table or "").lower()
                if t in optional:
                    hits.add(t)
                elif not t and md:  # unqualified: the optional table owns it and no other table does
                    owners = [a for a, name in optional.items() if name in md and col.name in md[name].columns]
                    if len(owners) == 1:
                        hits.add(owners[0])
            for alias in hits:
                if not _keeps_nulls(cond, alias):
                    found.append((cond.sql(dialect=dialect), optional[alias]))
    return list(dict.fromkeys(found))



def fanout_joins(db, sql: str, dialect: str) -> list[tuple[str, str, str]]:
    """``(table_a, table_b, parent)`` where a query aggregates (SUM/AVG/COUNT) over a join of two
    tables that each have several rows per the same parent: each row of one is repeated for
    every matching row of the other, so totals are multiplied. The join may sit in the SELECT
    that aggregates or in a CTE that a later SELECT aggregates. Totalling each table per
    parent key in its own CTE first avoids it. ``fanout_shape`` writes that shape out."""
    from sqlglot import exp

    tree = _parse(sql, dialect)
    if tree is None or db is None:
        return []
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    many = _many_keys(db)

    def counts(a) -> bool:
        return not (isinstance(a, exp.Count) and isinstance(a.this, exp.Distinct))

    any_agg = any(counts(a) for a in tree.find_all(exp.Sum, exp.Avg, exp.Count))
    found = []
    for select in tree.find_all(exp.Select):
        own = any(counts(a) and a.find_ancestor(exp.Select) is select
                  for a in select.find_all(exp.Sum, exp.Avg, exp.Count))
        in_cte = isinstance(select.parent, exp.CTE)
        if not (own or (in_cte and any_agg)):
            continue
        sources = []
        frm = select.args.get("from")
        if frm is not None:
            sources.append(frm.this)
        sources += [j.this for j in select.args.get("joins") or []]
        tables = [s.name for s in sources if isinstance(s, exp.Table) and s.name.lower() not in ctes]
        on_cols = set()
        for j in select.args.get("joins") or []:
            if j.args.get("on") is not None:
                on_cols |= {c.name.lower() for c in j.args["on"].find_all(exp.Column)}
            on_cols |= {u.name.lower() for u in j.args.get("using") or []}
        for i, a in enumerate(tables):
            for b in tables[i + 1:]:
                if a == b or (a, b) in many or (b, a) in many:
                    continue  # parent and child: the normal one-to-many join
                for (child, parent) in many:
                    # siblings joined through that parent's key (not two tables that merely share a dimension)
                    if child == a and (b, parent) in many and many[(a, parent)][0].lower() in on_cols \
                            and many[(b, parent)][0].lower() in on_cols:
                        found.append((a, b, parent))
    return list(dict.fromkeys(found))


def _many_keys(db) -> dict[tuple[str, str], tuple[str, str]]:
    """``(child, parent) -> (child_column, parent_column)`` for keys with several child rows per parent."""
    many = {}
    for child, ccol, parent, pcol in relations(db):
        if ccol.lower() not in _unique_cols(db, child):
            many.setdefault((child, parent), (ccol, pcol))
    return many


def fanout_shape(db, a: str, b: str, parent: str) -> str:
    """The query shape that avoids the fan-out, with this schema's table and key names."""
    many = _many_keys(db)
    acol, pcol = many[(a, parent)]
    bcol, _ = many[(b, parent)]
    return (f"WITH {a}_per_{parent} AS (\n"
            f"  SELECT {acol}, SUM(<measure from {a}>) AS {a}_total FROM {a}\n"
            f"  WHERE <conditions on {a} only> GROUP BY {acol}),\n"
            f"{b}_per_{parent} AS (\n"
            f"  SELECT {bcol}, SUM(<measure from {b}>) AS {b}_total FROM {b}\n"
            f"  WHERE <conditions on {b} only> GROUP BY {bcol})\n"
            f"SELECT ... FROM {parent} p\n"
            f"LEFT JOIN {a}_per_{parent} x ON x.{acol} = p.{pcol}\n"
            f"LEFT JOIN {b}_per_{parent} y ON y.{bcol} = p.{pcol}\n"
            f"WHERE <conditions on {parent}>  -- then use COALESCE(x.{a}_total, 0), COALESCE(y.{b}_total, 0)")


def counted_group_keys(sql: str, dialect: str) -> list[str]:
    """``COUNT(DISTINCT x)`` in a SELECT that also groups by ``x``: each group counts exactly 1.
    The model meant to count the groups (orders with a repeated product), which needs the
    grouped query as a subquery."""
    from sqlglot import exp

    tree = _parse(sql, dialect)
    if tree is None:
        return []
    found = []
    for select in tree.find_all(exp.Select):
        group = select.args.get("group")
        if group is None:
            continue
        keys = {g.sql(dialect=dialect).lower() for g in group.expressions if isinstance(g, exp.Column)}
        names = {g.name.lower() for g in group.expressions if isinstance(g, exp.Column)}
        for cnt in select.find_all(exp.Count):
            if cnt.find_ancestor(exp.Select) is not select or not isinstance(cnt.this, exp.Distinct):
                continue
            cols = [c for c in cnt.this.expressions if isinstance(c, exp.Column)]
            if len(cols) == 1:
                c = cols[0]
                if c.sql(dialect=dialect).lower() in keys or (not c.table and c.name.lower() in names):
                    found.append(cnt.sql(dialect=dialect))
    return list(dict.fromkeys(found))
