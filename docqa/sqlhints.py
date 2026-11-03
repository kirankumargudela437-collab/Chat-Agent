"""Grounding a small model in the real database: value lookup and error hints.

Two kinds of miss dominate when a 7B model writes SQL (measured on Spider and
BIRD with qwen2.5-coder:7b):

* **Wrong literal values.** The question says "Commander" and the column holds
  ``'commander'``, or the question says "1:54" and the data says ``'01:54'``. The
  query runs, returns nothing, and is wrong. ``ValueIndex`` reads the distinct
  short text values of every column once per database and finds the ones the
  question (or the business context) mentions, so the prompt can say
  ``card.type = 'Commander'``. This is the "entity and value retrieval" step of
  CHESS (Talaei et al., 2024) and the value lookup Vanna offers, done locally with
  plain string matching instead of an extra index service.
* **Invented columns.** ``no such column: T2.CRP`` tells the model what is wrong
  but not what is right. ``error_hint`` resolves the alias to its table, lists
  that table's real columns and the closest-named columns anywhere in the
  schema, so the retry has something to copy. Execution-guided repair with
  schema feedback is common to DIN-SQL, MAC-SQL's refiner and CHESS's revisor.
"""

from __future__ import annotations

import difflib
import logging
import re
import time
from collections import defaultdict

log = logging.getLogger(__name__)

_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "were", "be", "it",
    "this", "that", "with", "as", "by", "at", "from", "how", "what", "where", "when", "which", "who", "whom",
    "why", "does", "do", "did", "can", "many", "much", "list", "show", "give", "find", "name", "names",
    "all", "each", "every", "number", "count", "total", "average", "their", "its", "his", "her", "they",
    "than", "more", "less", "most", "least", "have", "has", "had", "not", "no", "please", "me", "there",
    "among", "between", "whose", "per", "any", "also", "only", "both", "one", "two", "first", "last",
}


def _norm(v: str) -> str:
    """Lower case, single spaces, leading zeros dropped from numbers ('01:54' -> '1:54')."""
    s = " ".join(str(v).lower().split())
    return re.sub(r"(?<![\d.])0+(\d)", r"\1", s)


def _quote(name: str) -> str:
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) else '"' + name.replace('"', '""') + '"'


class ValueIndex:
    """Distinct short text values per column of one database, for matching against a question."""

    def __init__(self, db, max_distinct: int = 2000, max_len: int = 60, time_budget: float = 20.0):
        self.by_norm: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        self.values: list[tuple[str, str, str, str]] = []  # (norm, table, column, value)
        start = time.monotonic()
        meta = getattr(db, "_metadata", None)
        if meta is None:
            return
        from sqlalchemy import String, Text, func, select

        for table in meta.sorted_tables:
            for col in table.columns:
                if time.monotonic() - start > time_budget:
                    log.info("Value index: time budget reached, later columns skipped")
                    return
                # SQLite declares many text columns loosely; also scan columns with no known type
                text_like = isinstance(col.type, (String, Text)) or type(col.type).__name__ in ("NullType", "TEXT")
                if not text_like or col.primary_key:
                    continue
                try:
                    q = (
                        select(col).where(col.is_not(None)).where(func.length(col) <= max_len)
                        .distinct().limit(max_distinct)
                    )
                    with db._engine.connect() as conn:
                        _guard(conn, start + time_budget)
                        rows = conn.execute(q).fetchall()
                except Exception as e:  # odd column type, permissions, timeout
                    log.debug("Value index skipped %s.%s: %s", table.name, col.name, e)
                    continue
                for (v,) in rows:
                    if not isinstance(v, str) or not v.strip() or v.strip().isdigit():
                        continue
                    n = _norm(v)
                    self.by_norm[n].append((table.name, col.name, v))
                    self.values.append((n, table.name, col.name, v))

    def __len__(self) -> int:
        return len(self.values)

    def match(self, text: str, limit: int = 12) -> list[tuple[str, str, str]]:
        """(table, column, value) whose value the text mentions: exact phrase first, then quoted substrings."""
        phrases = _phrases(text)
        found: list[tuple[str, str, str]] = []
        seen = set()

        def add(hit):
            key = (hit[0], hit[1], hit[2])
            if key not in seen:
                seen.add(key)
                found.append(hit)

        # longest phrases first, so "New York City" beats "York"
        for p in sorted(phrases, key=lambda x: (-len(x.split()), -len(x))):
            for hit in self.by_norm.get(p, ()):
                add(hit)
        # quoted strings in the question/evidence may be part of a longer value
        for q in _quoted(text):
            nq = _norm(q)
            if len(nq) < 3:
                continue
            for n, t, c, v in self.values:
                if nq in n and len(n) <= len(nq) + 25:
                    add((t, c, v))
                    if len(found) >= limit * 2:
                        break
        return found[:limit]


def _guard(conn, deadline: float) -> None:
    """Stop a long SQLite scan at the deadline (no-op on other databases)."""
    try:
        raw = conn.connection.driver_connection
        raw.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 50_000)
    except Exception:
        pass


def _quoted(text: str) -> list[str]:
    return [a or b for a, b in re.findall(r"'([^']{2,60})'|\"([^\"]{2,60})\"", text)]


def _phrases(text: str, max_words: int = 4) -> set[str]:
    words = re.findall(r"[\w][\w.:/&'-]*", text)
    out = set()
    for i in range(len(words)):
        for n in range(1, max_words + 1):
            gram = words[i : i + n]
            if len(gram) < n:
                break
            if n == 1 and (gram[0].lower() in _STOP or len(gram[0]) < 2):
                continue
            if all(g.lower() in _STOP for g in gram):
                continue
            out.add(_norm(" ".join(gram).strip("'.")))
    out |= {_norm(q) for q in _quoted(text)}
    return out


def format_value_hints(hits: list[tuple[str, str, str]]) -> str:
    if not hits:
        return ""
    lines = [f"- {_quote(t)}.{_quote(c)} = '{v}'" for t, c, v in hits]
    return (
        "Values that exist in the database and match words in the question "
        "(use these exact spellings and cases in filters):\n" + "\n".join(lines)
    )


# --------------------------------------------------------------------------- error hints


_NO_COLUMN = re.compile(r"no such column:\s*([^\s,]+)", re.IGNORECASE)
_NO_TABLE = re.compile(r"no such table:\s*([^\s,]+)", re.IGNORECASE)
_AMBIGUOUS = re.compile(r"ambiguous column name:\s*([^\s,]+)", re.IGNORECASE)


def _columns_by_table(db) -> dict[str, list[str]]:
    meta = getattr(db, "_metadata", None)
    if meta is None:
        return {}
    return {t.name: [c.name for c in t.columns] for t in meta.sorted_tables}


def _aliases(sql: str, dialect: str) -> dict[str, str]:
    """alias (lower) -> table name, from the query's FROM/JOIN clauses."""
    out = {}
    try:
        import sqlglot
        from sqlglot import exp

        for t in sqlglot.parse_one(sql, read=dialect).find_all(exp.Table):
            out[(t.alias_or_name or "").lower()] = t.name
            out[t.name.lower()] = t.name
    except Exception:
        for m in re.finditer(r"\b(?:from|join)\s+[\"`\[]?(\w+)[\"`\]]?(?:\s+(?:as\s+)?(\w+))?", sql, re.I):
            out[m.group(1).lower()] = m.group(1)
            if m.group(2):
                out[m.group(2).lower()] = m.group(1)
    return out


def _close(name: str, options: list[str], n: int = 4) -> list[str]:
    low = {o.lower(): o for o in options}
    hits = difflib.get_close_matches(name.lower(), list(low), n=n, cutoff=0.55)
    contains = [o for k, o in low.items() if name.lower() in k or (len(k) > 2 and k in name.lower())]
    out = []
    for o in [low[h] for h in hits] + contains:
        if o not in out:
            out.append(o)
    return out[:n]


def error_hint(db, sql: str, error: str, dialect: str = "sqlite") -> str:
    """Extra lines for the fix prompt that name the real tables/columns behind a schema error."""
    tables = _columns_by_table(db)
    if not tables:
        return ""
    lines = []
    m = _NO_COLUMN.search(error) or _AMBIGUOUS.search(error)
    if m:
        ref = m.group(1).strip("\"'`[]")
        alias, _, col = ref.rpartition(".")
        col = col.strip("\"'`[]")
        aliases = _aliases(sql, dialect)
        if _AMBIGUOUS.search(error):
            owners = [t for t, cols in tables.items() if col.lower() in (c.lower() for c in cols)]
            used = set(aliases.values())
            owners = [t for t in owners if t in used] or owners
            lines.append(f"Column {col} exists in several tables of this query ({', '.join(owners)}): prefix it with the table alias.")
        else:
            table = aliases.get(alias.lower()) if alias else None
            if table and table in tables:
                lines.append(f"Table {table} (alias {alias}) has these columns: {', '.join(_quote(c) for c in tables[table])}.")
            where = []
            for t, cols in tables.items():
                for c in _close(col, cols, 3):
                    where.append(f"{_quote(t)}.{_quote(c)}")
            if where:
                lines.append(f"Columns with a similar name: {', '.join(where[:8])}.")
            owners = [t for t, cols in tables.items() if col.lower() in (c.lower() for c in cols)]
            if owners and (not table or table not in owners):
                lines.append(f"{col} is a column of {', '.join(owners)}: join that table or use its alias.")
    m = _NO_TABLE.search(error)
    if m:
        name = m.group(1).strip("\"'`[]").split(".")[-1]
        close = _close(name, list(tables))
        lines.append(
            f"There is no table {name}. Tables in this database: {', '.join(_quote(t) for t in tables)}."
            + (f" Closest names: {', '.join(close)}." if close else "")
        )
    if lines and any(" " in c or "-" in c for cols in tables.values() for c in cols):
        lines.append('Column names with spaces or dashes must be quoted, e.g. "Customer Name" or `Customer Name`.')
    return "\n".join(lines)


# --------------------------------------------------------------------------- automatic identifier repair


def _key(name: str) -> str:
    """Name with case, underscores, spaces and dashes ignored: indep_year == IndepYear == "Indep Year"."""
    return re.sub(r"[\s_\-]+", "", name).lower()


def auto_fix_identifiers(db, sql: str, dialect: str = "sqlite") -> str | None:
    """Rewrite table/column names that differ from the schema only in case or separators.

    A 7B model that saw ``indep_year`` in an example writes it again even after the
    error hint says ``IndepYear``. When exactly one real name matches once case,
    underscores and spaces are ignored, the fix is mechanical and needs no model
    call. A column qualified with the wrong alias is moved to the one table of the
    query that has it. Returns the repaired SQL, or None when nothing could be fixed.
    """
    tables = _columns_by_table(db)
    if not tables:
        return None
    try:
        import sqlglot
        from sqlglot import exp

        tree = sqlglot.parse_one(sql, read=dialect)
    except Exception:
        return None
    by_key = {_key(t): t for t in tables}
    changed = False
    alias_to_table: dict[str, str] = {}
    alias_spelling: dict[str, str] = {}
    for t in tree.find_all(exp.Table):
        real = t.name if t.name in tables else by_key.get(_key(t.name))
        if real is None:
            continue
        if real != t.name:
            t.set("this", exp.to_identifier(real, quoted=not re.fullmatch(r"[A-Za-z_]\w*", real)))
            changed = True
        alias_to_table[(t.alias or real).lower()] = real
        alias_spelling[(t.alias or real).lower()] = t.alias or real
        alias_to_table[real.lower()] = real
    if not alias_to_table:
        return tree.sql(dialect=dialect) if changed else None
    select_aliases = {a.alias.lower() for a in tree.find_all(exp.Alias) if a.alias}
    in_query = list(dict.fromkeys(alias_to_table.values()))
    for col in tree.find_all(exp.Column):
        name = col.name
        if not name or name == "*" or name.lower() in select_aliases:
            continue
        qual = (col.table or "").lower()
        scope = [alias_to_table[qual]] if qual in alias_to_table else in_query
        if any(name in tables[t] for t in scope):
            continue
        hits = [(t, c) for t in scope for c in tables[t] if _key(c) == _key(name)]
        if not hits and qual in alias_to_table:  # wrong alias: look in the other tables of the query
            hits = [(t, c) for t in in_query for c in tables[t] if _key(c) == _key(name)]
        if len({(t, c) for t, c in hits}) != 1:
            continue
        table, real = hits[0]
        col.set("this", exp.to_identifier(real, quoted=not re.fullmatch(r"[A-Za-z_]\w*", real)))
        if qual and alias_to_table.get(qual) != table:
            alias = next((a for a, t in alias_to_table.items() if t == table and a != table.lower()), table)
            col.set("table", exp.to_identifier(alias_spelling.get(alias, alias)))
        changed = True
    return tree.sql(dialect=dialect) if changed else None


# --------------------------------------------------------------------------- routing evidence


def schema_terms(db) -> set[str]:
    """Stemmed words of every table and column name (``StudentTranscripts`` -> student, transcript)."""
    from .multidb import _terms

    out: set[str] = set()
    for t, cols in _columns_by_table(db).items():
        out.update(_terms(t.replace("_", " ")))
        out.update(_terms(t))
        for c in cols:
            out.update(_terms(c))
            out.update(_terms(c.replace("_", " ")))
    return out


def schema_fit(question: str, db, values: "ValueIndex | None" = None) -> tuple[float, str]:
    """How well a database's names and values cover the question, plus a short evidence line.

    Used to choose among look-alike databases (flight_1 / flight_2 / flight_4):
    the share of the question's content words that are table or column names, and
    the stored values the question mentions.
    """
    from .multidb import _terms

    q = [t for t in dict.fromkeys(_terms(question)) if not t.isdigit()]
    vocab = schema_terms(db)
    named = [t for t in q if t in vocab]
    hits = values.match(question, limit=4) if values is not None else []
    score = (len(named) / len(q) if q else 0.0) + 0.25 * min(len(hits), 2)
    bits = []
    if named:
        bits.append("names: " + ", ".join(named[:8]))
    if hits:
        bits.append("values: " + ", ".join(f"{t}.{c}='{v}'" for t, c, v in hits))
    missing = [t for t in q if t not in vocab]
    if missing:
        bits.append("no match for: " + ", ".join(missing[:6]))
    return score, "; ".join(bits)
