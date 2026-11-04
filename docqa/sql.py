"""Text-to-SQL with your documents as the business layer, and a self-healing loop.

A database schema knows ``cust_stat_cd`` and ``rev_amt``; your data dictionary
knows that "churned revenue" means ``cust_stat_cd = 4`` and no order in 90 days.
Before writing SQL the agent gathers:

1. **Schema**: the live database's tables (``DOCQA_DATABASE_URL``), DDL pasted in
   the app or passed with ``--schema``, and ``.sql`` files from the docs folder.
2. **Business rules**: passages from your PDFs / DOCX / Markdown.
3. **Golden examples**: verified question-to-SQL pairs (``-- Question:`` blocks in
   ``.sql`` files), so the model copies your CTE and window-function style.

Then a LangGraph loop runs:

``link`` (map the question's terms to tables, joins and rules, citing [n])
-> ``generate`` -> ``check`` (parse with sqlglot, read-only statements only)
-> ``execute`` (only with a database, on a read-only connection)
-> back to ``generate`` with the exact error, up to ``DOCQA_SQL_MAX_ATTEMPTS``.

Without a database the query is still parsed in the target dialect, so syntax
errors are fed back and fixed, but it is not run and the answer says so.

Safety is two layers, because a keyword regex is not one (it blocks
``WHERE note = 'please update'`` and lets ``PRAGMA``, ``ATTACH`` or a second
statement through):

* sqlglot must parse exactly one statement whose root is a SELECT / UNION /
  INTERSECT / EXCEPT, with no INSERT, UPDATE, DELETE, MERGE, DDL or SELECT INTO
  anywhere inside it (including in CTEs);
* the connection itself is read-only: SQLite is opened with ``mode=ro``,
  PostgreSQL and MySQL sessions are set to read-only transactions, and every
  query is rolled back. For other databases, use a read-only database user.
"""

from __future__ import annotations

import logging
import re
import time
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TypedDict

import sqlglot
from sqlglot import exp

from .sqlchecks import (REVIEW_PROMPT, counted_group_keys, dry_run, fanout_shape, fanout_sites, join_keys,
                        join_paths, missing_values, optional_side_filters, parse_review, result_problems, self_comparisons, tables_in)
from .trace import set_question, set_step
from .sqlhints import (ValueIndex, auto_fix_identifiers, error_hint, format_value_hints, qualify_ambiguous,
                       schema_fit)
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

log = logging.getLogger(__name__)

# SQLAlchemy backend name -> sqlglot dialect
_BACKEND_TO_DIALECT = {
    "sqlite": "sqlite", "postgresql": "postgres", "postgres": "postgres", "mysql": "mysql",
    "mariadb": "mysql", "mssql": "tsql", "oracle": "oracle", "snowflake": "snowflake",
    "bigquery": "bigquery", "redshift": "redshift", "duckdb": "duckdb", "databricks": "databricks",
    "trino": "trino", "presto": "presto", "clickhouse": "clickhouse",
}

DIALECT_NAMES = {
    "postgres": "PostgreSQL", "mysql": "MySQL", "tsql": "T-SQL (SQL Server)", "oracle": "Oracle",
    "snowflake": "Snowflake", "bigquery": "BigQuery", "sqlite": "SQLite", "redshift": "Redshift",
    "duckdb": "DuckDB", "databricks": "Databricks", "spark": "Spark SQL", "trino": "Trino",
    "presto": "Presto", "clickhouse": "ClickHouse",
}

_ALLOWED_ROOTS = (exp.Select, exp.SetOperation, exp.Subquery)
_FORBIDDEN = tuple(
    getattr(exp, n)
    for n in ("Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "TruncateTable",
              "Command", "Into", "Copy", "Grant", "Pragma", "Attach", "Detach", "Use", "Set", "Transaction",
              "Lock",          # SELECT ... FOR UPDATE takes write locks
              "DDL", "DML")    # base classes: also catches statement types a newer sqlglot adds
    if hasattr(exp, n)
)

# A SELECT can still read files, call other servers or stall the database through
# functions. Blocked by name in any position (idea and most names from WrenAI's policy check).
_BLOCKED_FUNCTIONS = {
    # PostgreSQL: server files, large objects, other servers, sleeping
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_ls_logdir", "pg_ls_waldir", "pg_ls_tmpdir",
    "pg_stat_file", "lo_import", "lo_export", "lo_get", "dblink", "dblink_exec", "dblink_connect",
    "pg_sleep", "pg_sleep_for", "pg_sleep_until", "pg_terminate_backend", "pg_cancel_backend",
    "set_config", "query_to_xml", "query_to_json",
    # MySQL
    "load_file", "sleep", "benchmark", "sys_exec", "sys_eval",
    # SQLite
    "load_extension", "readfile", "writefile", "edit", "fts3_tokenizer",
    # SQL Server
    "openrowset", "opendatasource", "openquery", "xp_cmdshell", "xp_dirtree", "xp_fileexist",
    "sp_execute_external_script", "sp_oacreate",
    # DuckDB / Trino / Spark file and remote readers
    "readcsv", "read_csv", "read_csv_auto", "read_parquet", "read_json", "read_json_auto", "read_ndjson",
    "read_text", "read_blob", "read_xlsx", "parquet_scan", "glob", "postgres_scan", "postgres_query",
    "mysql_scan", "mysql_query", "sqlite_scan", "sqlite_query", "iceberg_scan", "delta_scan", "url",
    # Oracle
    "utl_http.request", "utl_file.fopen", "dbms_pipe.receive_message", "dbms_lock.sleep",
}
_BLOCKED_NORMALISED = {n.replace("_", "") for n in _BLOCKED_FUNCTIONS}


def _function_name(f) -> str:
    name = f.name if isinstance(f, exp.Anonymous) else type(f).key
    return (name or "").lower().replace("_", "")


class UnsafeSQLError(ValueError):
    """The query is not a single read-only statement."""


class DatabaseError(ValueError):
    """The database could not be opened."""


def dialect_from_url(url: str | None) -> str:
    if not url:
        return "postgres"
    backend = url.split(":", 1)[0].split("+", 1)[0].lower()
    return _BACKEND_TO_DIALECT.get(backend, backend)


def dialect_name(dialect: str) -> str:
    return DIALECT_NAMES.get(dialect, dialect)


# --------------------------------------------------------------------------- guardrail


def clean_sql(raw: str) -> str:
    """The SQL inside a model reply: first fenced block, else the text minus a 'SQL:' label."""
    m = re.search(r"```(?:sql)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    text = m.group(1) if m else raw
    return re.sub(r"^\s*(?:sql(?: query)?|corrected sql|query)\s*:\s*", "", text.strip(), flags=re.IGNORECASE).strip()


def check_read_only(sql: str, dialect: str) -> str:
    """Return the single statement, or raise ``UnsafeSQLError`` / a sqlglot parse error."""
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except sqlglot.errors.ParseError as e:
        raise UnsafeSQLError(f"SQL syntax error ({dialect_name(dialect)}): {_first_line(e)}") from e
    if len(statements) != 1:
        raise UnsafeSQLError(f"Expected exactly one SQL statement, got {len(statements)}.")
    tree = statements[0]
    if not isinstance(tree, _ALLOWED_ROOTS):
        raise UnsafeSQLError(f"Only read-only SELECT queries are allowed, not {tree.key.upper()}.")
    bad = next((n for n in tree.walk() if isinstance(n, _FORBIDDEN)), None)
    if bad is not None:
        raise UnsafeSQLError(f"Only read-only SELECT queries are allowed; found {bad.key.upper()} inside the query.")
    for f in tree.find_all(exp.Func):
        if _function_name(f) in _BLOCKED_NORMALISED:
            shown = f.name if isinstance(f, exp.Anonymous) else type(f).key
            raise UnsafeSQLError(
                f"Function {shown.upper()} is not allowed: it can read files, reach other servers or stall the database."
            )
    return sql.strip().rstrip(";").strip()


def _first_line(e: Exception) -> str:
    return str(e).strip().splitlines()[0][:300] if str(e).strip() else type(e).__name__


# --------------------------------------------------------------------------- database


def make_read_only_engine(url: str):
    from sqlalchemy import create_engine, event
    from sqlalchemy.engine import make_url
    from sqlalchemy.pool import NullPool

    u = make_url(url)
    backend = u.get_backend_name()
    if backend == "sqlite":
        if not u.database or u.database == ":memory:":
            raise DatabaseError("An in-memory SQLite database has no tables to query; point DOCQA_DATABASE_URL at a file.")
        path = Path(u.database).expanduser().resolve()
        if not path.is_file():
            raise DatabaseError(f"SQLite database not found: {path}")
        uri = path.as_uri() + "?mode=ro"
        return create_engine(
            "sqlite://", creator=lambda: sqlite3.connect(uri, uri=True, check_same_thread=False), poolclass=NullPool
        )

    engine = create_engine(url, pool_pre_ping=True)

    @event.listens_for(engine, "connect")
    def _read_only(dbapi_conn, _record):  # pragma: no cover - needs a real server
        stmt = {
            "postgresql": "SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY",
            "mysql": "SET SESSION TRANSACTION READ ONLY",
            "mariadb": "SET SESSION TRANSACTION READ ONLY",
        }.get(backend)
        if stmt:
            cur = dbapi_conn.cursor()
            cur.execute(stmt)
            cur.close()

    return engine


class Database:
    """A read-only database plus its schema as ``CREATE TABLE`` text with sample rows.

    (LangChain's ``SQLDatabase`` does the same, but lives in ``langchain-community``,
    which is being sunset.)
    """

    def __init__(self, engine, include_tables: tuple[str, ...] = (), sample_rows: int = 3):
        from sqlalchemy import MetaData, inspect

        self._engine = engine
        self.dialect = engine.dialect.name
        self.sample_rows = sample_rows
        self.notes: dict[str, str] = {}  # table -> column descriptions, appended to its schema text
        names = inspect(engine).get_table_names() + inspect(engine).get_view_names()
        if include_tables:
            missing = set(include_tables) - set(names)
            if missing:
                raise DatabaseError(f"Tables not found in the database: {', '.join(sorted(missing))}")
            names = list(include_tables)
        self.table_names = sorted(n for n in names if not n.startswith("sqlite_"))
        self._metadata = MetaData()
        self._metadata.reflect(bind=engine, only=self.table_names, views=True)

    def table_parts(self) -> dict[str, dict[str, str]]:
        """Per table: ``ddl`` (CREATE TABLE with keys), ``compact`` (column names and keys
        on one line), ``sample`` (a few rows) and ``notes`` (column descriptions). Cached."""
        if getattr(self, "_parts", None) is None:
            from sqlalchemy import select
            from sqlalchemy.schema import CreateTable

            parts = {}
            for table in self._metadata.sorted_tables:
                p = {"ddl": str(CreateTable(table).compile(self._engine)).strip(), "sample": "", "notes": ""}
                keys = [c.name for c in table.primary_key.columns]
                fks = [f"{fk.parent.name} -> {fk.column.table.name}.{fk.column.name}" for fk in table.foreign_keys]
                cols = ", ".join(c.name for c in table.columns)
                p["compact"] = (f"-- {table.name}({cols})" + (f"; primary key {', '.join(keys)}" if keys else "")
                                + (f"; joins {'; '.join(fks)}" if fks else ""))
                p["keys"] = (f"-- {table.name}: primary key {', '.join(keys) or '?'}"
                             + (f"; joins {'; '.join(fks)}" if fks else "") + f"; {len(table.columns)} columns")
                if self.sample_rows:
                    try:
                        with self._engine.connect() as conn:
                            rows = conn.execute(select(table).limit(self.sample_rows)).fetchall()
                        sample = rows_as_text([c.name for c in table.columns], [tuple(r) for r in rows], width=40)
                        p["sample"] = f"/* {len(rows)} sample rows from {table.name}:\n{sample}\n*/"
                    except Exception as e:  # pragma: no cover - permissions on one table
                        log.debug("No sample rows for %s: %s", table.name, e)
                if self.notes.get(table.name):
                    p["notes"] = f"/* Column descriptions for {table.name}:\n{self.notes[table.name]}\n*/"
                parts[table.name] = p
            self._parts = parts
        return self._parts

    def table_infos(self) -> dict[str, str]:
        """``CREATE TABLE`` (with foreign keys) plus a few sample rows, per table. Cached."""
        if getattr(self, "_infos", None) is None:
            self._infos = {n: "\n\n".join(filter(None, [p["ddl"], p["sample"], p["notes"]]))
                           for n, p in self.table_parts().items()}
        return self._infos

    def refresh(self) -> None:
        """Forget the cached schema (after a table was added or changed)."""
        from sqlalchemy import MetaData, inspect

        self._parts = self._infos = None
        self._metadata = MetaData()
        names = inspect(self._engine).get_table_names() + inspect(self._engine).get_view_names()
        self.table_names = sorted(n for n in names if not n.startswith("sqlite_"))
        self._metadata.reflect(bind=self._engine, only=self.table_names, views=True)

    def neighbours(self, name: str) -> set[str]:
        """Tables joined to ``name`` by a foreign key, in either direction."""
        out = set()
        for table in self._metadata.sorted_tables:
            for fk in table.foreign_keys:
                parent = fk.column.table.name
                if table.name == name:
                    out.add(parent)
                elif parent == name:
                    out.add(table.name)
        out.discard(name)
        return out

    def get_table_info(self) -> str:
        return "\n\n".join(self.table_infos().values())


def open_database(url: str, tables: tuple[str, ...] = (), sample_rows: int = 3) -> Database:
    """The database behind ``url`` on a read-only connection."""
    try:
        return Database(make_read_only_engine(url), tables, sample_rows)
    except DatabaseError:
        raise
    except Exception as e:
        safe = re.sub(r"//[^@/]*@", "//***@", url)  # hide the password
        raise DatabaseError(f"Could not open database {safe}: {_first_line(e)}") from e


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    truncated: bool


def _server_timeout(conn, seconds: float) -> None:
    """The same time limit on server databases: Postgres statement_timeout, MySQL
    max_execution_time, Oracle's driver call_timeout. SQLite uses the progress handler."""
    name = conn.dialect.name
    try:
        if name == "postgresql":
            conn.exec_driver_sql(f"SET LOCAL statement_timeout = {int(seconds * 1000)}")
        elif name in ("mysql", "mariadb"):
            conn.exec_driver_sql(f"SET SESSION max_execution_time = {int(seconds * 1000)}")
        elif name == "oracle":
            conn.connection.driver_connection.call_timeout = int(seconds * 1000)
    except Exception as e:  # never block a query over the limit itself
        log.debug("Could not set a query time limit on %s: %s", name, e)


QUERY_TIMEOUT = 30.0  # seconds; a runaway join on a big SQLite file would otherwise hang the agent


def run_query(db, sql: str, max_rows: int, timeout: float | None = QUERY_TIMEOUT) -> QueryResult:
    """Run one checked query, fetch at most ``max_rows`` rows, always roll back.

    On SQLite the query is stopped after ``timeout`` seconds (other databases use
    their own statement timeouts); the error tells the model to write a cheaper query.
    """
    from .sqlhints import _guard

    with db._engine.connect() as conn:
        if timeout:
            _guard(conn, time.monotonic() + timeout)
            _server_timeout(conn, timeout)
        try:
            # no_parameters: '%' and ':name' inside the SQL are passed to the driver untouched
            res = conn.execution_options(no_parameters=True).exec_driver_sql(sql)
            columns = list(res.keys()) if res.returns_rows else []
            rows = [tuple(r) for r in res.fetchmany(max_rows + 1)] if res.returns_rows else []
        except Exception as e:
            msg = str(e).lower()
            if timeout and any(m in msg for m in ("interrupted", "statement timeout", "dpi-1067", "call timeout",
                                                  "max_execution_time", "query execution was interrupted")):
                raise TimeoutError(
                    f"query ran longer than {timeout:g}s and was stopped; write a simpler query "
                    "(filter early, join on keys, avoid cross joins and correlated subqueries)"
                ) from e
            raise
        finally:
            if timeout:
                _guard(conn, None)
            conn.rollback()
    return QueryResult(columns, rows[:max_rows], len(rows) > max_rows)


def rows_as_text(columns: list[str], rows: list[tuple], limit: int = 50, width: int = 60) -> str:
    if not columns:
        return "(no columns)"
    if not rows:
        return " | ".join(columns) + "\n(no rows)"

    def cell(v: Any) -> str:
        s = "NULL" if v is None else str(v)
        return s if len(s) <= width else s[: width - 1] + "…"

    lines = [" | ".join(columns), " | ".join("---" for _ in columns)]
    lines += [" | ".join(cell(v) for v in r) for r in rows[:limit]]
    return "\n".join(lines)


# --------------------------------------------------------------------------- prompts

_RULES = (
    "Rules:\n"
    "- Write ONE read-only {dialect_name} query (SELECT or WITH ... SELECT). Never modify data.\n"
    "- Use only tables and columns that appear in the schema. Do not invent names.\n"
    "- Apply the business rules and definitions from the context exactly (status codes, KPI formulas, exclusions).\n"
    "- Documents explain business terms, codes and values; never add a column, table or join just because a "
    "passage mentions it. Use only what the question asks for.\n"
    "- Prefer readable CTEs over nested subqueries; use window functions (ROW_NUMBER, RANK, LAG, LEAD, "
    "SUM() OVER) where they fit; guard divisions with NULLIF and handle NULLs with COALESCE.\n"
    "- Return exactly the columns the question asks for, in the order it asks; no extra columns.\n"
    "- Do not add filters, conditions or joins the question does not ask for. Put parentheses around mixed AND/OR.\n"
    "- When counting or grouping people or things, GROUP BY their key column (id), not their name: names repeat.\n"
    "- If two tables only connect through a third (a bridge table), join through it using the foreign keys.\n"
    "- For 'not / never / without' questions use NOT IN or EXCEPT, not != on a join.\n"
    "- For the row with the highest / lowest / latest value ('the most recent order'), use {top_one}; "
    "when every tied row should be returned, compare with a subquery (= (SELECT MAX(...) ...)) instead.\n"
    "{grain_rule}"
    "- Follow the style of the golden examples when one is similar, but use the column names of THIS schema.\n"
    "- Unless the question asks for everything, return at most {max_rows} rows.\n"
    "- Output only the SQL in one ```sql block, no explanation."
    "{hint}"
)

ANSWER_SHAPE = (
    "- Answer shape: what ONE result row is about (the entity), the measure it returns, any "
    "grouping, the denominator of a ratio, and how ties are handled. When the question picks one "
    "thing by a property ('the oldest player', 'the most expensive set'), say that it selects that "
    "thing first and then returns its measure: 'the long passing score of the oldest player' is "
    "one player's score, not MAX(long passing) over all players.\n"
)

GRAIN_PLAN = (
    "- Grain: what one counted or summed thing is (one order, one payment, one returned line: "
    "decide it from the question, it is not always the main table) and its key column. For every "
    "other table, say whether it has one row or several rows per that thing; joining a table with "
    "several rows repeats the thing, so plan a CTE that totals that table per key first, and join "
    "the totals. Say which rows must survive even with no match (LEFT JOIN, missing totals as "
    "COALESCE(total, 0)), and where each status, date or type condition belongs: inside the CTE of "
    "the table it filters, or in the LEFT JOIN's ON clause, never in WHERE on the optional side. "
    "Classify, rank or count only after the one-row-per-thing result exists.\n"
    "- Roles: when a table has two keys to the same table (who ordered, who is billed), say which "
    "one the question means.\n"
)

GRAIN_RULE = (
    "- Follow the grain in the mapping: total each table that has several rows per counted thing in "
    "its own CTE, grouped by that thing's key, then join the CTEs. A condition on the optional side "
    "of a LEFT JOIN goes in its CTE or in the ON clause, not in WHERE.\n"
)

LINK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a data architect. Before any SQL is written, map the user's question onto the "
            "database. Using ONLY the schema and the numbered context, list briefly:\n"
            "- Tables: the tables needed and how they join (exact key columns).\n"
            "- Columns and rules: how each business term in the question maps to columns, filters, "
            "status codes or formulas, citing the passage as [n].\n"
            "{shape}"
            "{grain}"
            "- Gaps: anything the question needs that the schema and context do not define.\n"
            "Do not write the SQL. Be concise.{hint}\n\n"
            "Dialect: {dialect_name}\n\nSchema:\n{schema}{paths}\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
).partial(grain="", paths="")  # off unless grain_plan

GENERATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert {dialect_name} engineer.\n" + _RULES + "\n\n"
            "Schema:\n{schema}\n\nBusiness rules, schema notes and golden examples:\n{context}\n\n"
            "Mapping of the question onto the schema:\n{mapping}",
        ),
        # golden examples as earlier turns: small models copy a worked answer more
        # faithfully than an example buried in the instructions (Vanna does the same)
        MessagesPlaceholder("examples", optional=True),
        ("human", "{question}"),
    ]
).partial(grain_rule="")  # off unless grain_plan

FIX_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert {dialect_name} engineer. Your previous query failed. Read the error, "
            "check every table and column against the schema, and return a corrected query.\n" + _RULES + "\n\n"
            "Schema:\n{schema}\n\nBusiness rules, schema notes and golden examples:\n{context}\n\n"
            "Mapping of the question onto the schema:\n{mapping}",
        ),
        ("human", "Question: {question}\n\nFailed SQL:\n```sql\n{sql}\n```\n\nError: {error}\n\nCorrected SQL:"),
    ]
).partial(grain_rule="")  # off unless grain_plan

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the user's question in a few sentences using ONLY the query result below. "
            "Quote the numbers as they appear. If the result is empty, say no rows matched. "
            "Do not repeat the SQL or the full table.",
        ),
        ("human", "Question: {question}\n\nSQL:\n{sql}\n\nResult ({n_rows} rows{more}):\n{table}"),
    ]
)


# --------------------------------------------------------------------------- graph


class SQLState(TypedDict, total=False):
    question: str
    schema: str
    context: str
    examples: list
    mapping: str
    sql: str
    error: Optional[str]
    attempts: int
    failures: list
    result: Optional[QueryResult]
    executed: bool
    hint: str
    votes: int
    checked: list
    fallback: Optional[dict]
    reviewed: bool
    paths: str
    bonus: int


@dataclass
class SQLResult:
    question: str
    sql: str
    dialect: str
    mapping: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    truncated: bool = False
    executed: bool = False
    attempts: int = 0
    error: str | None = None
    failures: list[tuple[str, str]] = field(default_factory=list)  # (sql, error) of earlier attempts
    database: str = ""  # with a database folder: the database the question was routed to
    candidates: list[str] = field(default_factory=list)  # the databases the router considered
    votes: int = 0  # with several candidate queries: how many returned the chosen result
    n_candidates: int = 1
    alternatives: list = field(default_factory=list)  # the SQL of every candidate that ran, in order

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.sql)

    def table_text(self, limit: int = 20) -> str:
        return rows_as_text(self.columns, self.rows, limit)


class SQLAgent:
    """Schema linking + generation + checking (+ execution) with retries, as a LangGraph."""

    def __init__(
        self,
        llm: BaseChatModel,
        db=None,
        dialect: str | None = None,
        max_attempts: int = 3,
        max_rows: int = 50,
        schema_linking: bool = True,
        embeddings=None,
        max_tables: int = 20,
        schema_chars: int = 12000,
        catalog=None,
        db_router: str = "llm",
        db_candidates: int = 5,
        value_lookup: bool = True,
        sql_patterns: bool = True,
        patterns_dir: str | None = None,
        sql_candidates: int = 1,
        route_margin: float = 0.25,
        route_choices: int = 2,
        answer_checks: bool = True,
        review: bool = False,
        independent_plans: bool = False,
        schema_ttl: float = 0.0,
        answer_shape: bool = True,
        grain_plan: bool = False,
        tracer=None,
    ):
        """``catalog`` (a ``multidb.DatabaseCatalog``) routes each question to one of many databases."""
        self.llm = llm
        self.catalog = catalog
        if db_router not in ("llm", "retrieval"):
            raise ValueError("DOCQA_DB_ROUTER must be 'llm' or 'retrieval'")
        self.db_router = db_router
        self.db_candidates = max(1, db_candidates)
        self.route_margin = route_margin  # schema-fit lead that settles routing without the model
        self.route_choices = max(2, route_choices)  # how many of the ranked databases the model chooses from
        self.database_name = ""
        self._vectors_by_db: dict[str, dict[str, list[float]] | None] = {}
        self.embeddings = embeddings  # to pick the relevant tables of a big database
        self.max_tables = max_tables
        self.schema_chars = schema_chars
        self._table_vectors: dict[str, list[float]] | None = None
        self.db = db
        if catalog is not None and db is None:
            dialect = "sqlite"
        self.dialect = dialect or (dialect_from_url(str(db._engine.url)) if db is not None else "postgres")
        if db is not None and db.dialect == "sqlite":
            self.dialect = "sqlite"
        self.value_lookup = value_lookup
        self._patterns = None
        if sql_patterns:  # reference examples of SQL techniques, retrieved per question
            from .sqlpatterns import PatternIndex, load_patterns

            extra = load_patterns(patterns_dir) if patterns_dir else ()
            self._patterns = PatternIndex(load_patterns() + extra)
        self.sql_candidates = max(1, sql_candidates)
        self.answer_checks = answer_checks  # EXPLAIN first; re-ask once when a result looks wrong
        self.review = review  # the model reviews its own query and first rows (one extra call)
        self.independent_plans = independent_plans  # each extra candidate plans the question afresh
        self.answer_shape = answer_shape  # the plan states what one result row is (entity, measure, ...)
        self.grain_plan = grain_plan  # plan the counted thing, multiplying joins and surviving rows; check them
        self.schema_ttl = schema_ttl  # seconds before the live schema is read again (0: once per session)
        self._schema_loaded = time.monotonic()
        self.error_help = None  # (error, sql) -> manual text shown when fixing a failed query
        self.last_schema: dict[str, str] = {}  # table -> what reached the prompt (keys/columns/ddl+sample+notes)
        self._value_indexes: dict[str, object] = {}
        self.max_attempts = max_attempts
        self.max_rows = max_rows
        self.schema_linking = schema_linking
        from .trace import tracer_from_env

        self.tracer = tracer or tracer_from_env(_run_settings(llm, self))
        if self.tracer is not None:  # every prompt, raw reply, token count and stop reason
            llm = llm.with_config(callbacks=[self.tracer])
        self._link = LINK_PROMPT | llm | StrOutputParser()
        self._generate = GENERATE_PROMPT | llm | StrOutputParser()
        self._fix = FIX_PROMPT | llm | StrOutputParser()
        self._summary = SUMMARY_PROMPT | llm | StrOutputParser()
        self._review = REVIEW_PROMPT | llm | StrOutputParser()
        self._graph = self._build_graph()

    # -- schema ---------------------------------------------------------------------

    def live_schema(self) -> str:
        return self.db.get_table_info() if self.db is not None else ""

    def select_tables(self, question: str, context: str = "") -> list[str]:
        """The live tables to show the model, most relevant first.

        A small database is shown whole. In a big one (more than ``max_tables``) a
        table ranks high when the question or the retrieved business rules name it
        (a data dictionary that says "join usr_mst to ord_hdr" pulls both in), then
        by embedding similarity to the question, and tables joined to the top ones by
        a foreign key come along. Vanna retrieves related DDL per question the same
        way; this adds the name and foreign-key signals.
        """
        infos = self.db.table_infos()
        names = list(infos)
        if len(names) <= 1:
            return names
        text = f"{question}\n{context}".lower()
        score = {n: 0.0 for n in names}
        for n in names:
            if re.search(rf"(?<![a-z0-9_]){re.escape(n.lower())}(?![a-z0-9_])", text):
                score[n] += 2.0
        if self.embeddings is not None:
            if self._table_vectors is None:
                self._table_vectors = dict(zip(names, self.embeddings.embed_documents([infos[n] for n in names])))
            q = self.embeddings.embed_query(question)
            for n, v in self._table_vectors.items():
                score[n] += _cosine(q, v)
        else:  # no embeddings: share of the question's words found in the table's DDL
            words = set(re.findall(r"[a-z0-9_]{3,}", question.lower()))
            for n in names:
                if words:
                    score[n] += len(words & set(re.findall(r"[a-z0-9_]{3,}", infos[n].lower()))) / len(words)
        ranked = sorted(names, key=lambda n: -score[n])
        if len(names) <= self.max_tables:
            return ranked  # every table, most relevant first (the order decides what gets detail)
        top = ranked[: max(1, self.max_tables // 2)]
        chosen = list(top)
        for n in top:
            chosen += sorted(self.db.neighbours(n) - set(chosen))
        for n in ranked:
            if len(chosen) >= self.max_tables:
                break
            if n not in chosen:
                chosen.append(n)
        return chosen[: self.max_tables]

    def schema_text(self, pasted: str = "", question: str = "", context: str = "") -> str:
        parts = []
        if self.db is not None and self.schema_ttl and time.monotonic() - self._schema_loaded > self.schema_ttl:
            self.db.refresh()  # pick up tables and columns changed while the chat is open
            self._table_vectors = None
            self._schema_loaded = time.monotonic()
        if self.db is not None:
            text, report = build_schema(self.db.table_parts(), self.select_tables(question, context),
                                        self.schema_chars, total=len(self.db.table_names), question=question)
            self.last_schema = report
            log.info("Schema in prompt: %s", report)
            parts.append(f"-- Live database ({dialect_name(self.dialect)}):\n{text}")
        if pasted.strip():
            parts.append(f"-- Provided by the user:\n{pasted.strip()}")
        return "\n\n".join(parts) or "(no schema given: use the table definitions in the context)"

    # -- many databases -------------------------------------------------------------

    def route(self, question: str) -> tuple[str, list[str]]:
        """``(database, candidates)`` for a question, using the catalog's ranking (and the model)."""
        candidates = [n for n, _ in self.catalog.rank(question, self.db_candidates)]
        if len(candidates) <= 1:
            return candidates[0], candidates
        scores: dict[str, float] = {}
        candidates, evidence = rerank_by_fit(
            self.catalog, question, candidates, self._values_for if self.value_lookup else None, scores
        )
        if self.db_router != "llm":
            return candidates[0], candidates
        # A small model asked to pick among five look-alikes overrides a correct #1 more
        # often than it fixes a wrong one, so a clear schema-fit winner is kept as is and
        # the model only breaks close calls, between the top few.
        if scores and scores[candidates[0]] - scores[candidates[1]] >= self.route_margin:
            log.info("Routing: %s leads on schema fit, no model call", candidates[0])
            return candidates[0], candidates
        shortlist = candidates[: self.route_choices]
        return self.catalog.choose(question, shortlist, self.llm, evidence), candidates

    def _values_for(self, key: str, db):
        if key not in self._value_indexes:
            try:
                self._value_indexes[key] = ValueIndex(db, time_budget=10.0)
            except Exception as e:  # never let the lookup break a question
                log.info("Value index failed for %s: %s", key, e)
                self._value_indexes[key] = None
        return self._value_indexes[key]

    def use_database(self, name: str) -> None:
        """Switch to one database of the catalog (its table embeddings are kept per database)."""
        if name == self.database_name and self.db is not None:
            return
        if self.database_name:
            self._vectors_by_db[self.database_name] = self._table_vectors
        self.db = self.catalog.open(name)
        self.dialect = "sqlite"
        self.database_name = name
        self._table_vectors = self._vectors_by_db.get(name)

    # -- graph nodes ------------------------------------------------------------------

    def _base(self, s: SQLState) -> dict:
        return {
            "dialect_name": dialect_name(self.dialect),
            "top_one": top_one(self.dialect),
            "shape": ANSWER_SHAPE if self.answer_shape else "",
            "grain": GRAIN_PLAN if self.grain_plan else "",
            "grain_rule": GRAIN_RULE if self.grain_plan else "",
            "paths": s.get("paths", ""),
            "max_rows": self.max_rows,
            "schema": s["schema"],
            "context": s.get("context") or "(none)",
            "mapping": s.get("mapping") or "(not done)",
            "question": s["question"],
            "hint": f"\n- {s['hint']}" if s.get("hint") else "",
        }

    def _node_link(self, s: SQLState) -> dict:
        if s.get("mapping"):  # reused from an earlier candidate
            return {"mapping": s["mapping"]}
        if not self.schema_linking:
            return {"mapping": ""}
        set_step("plan", attempt=0)
        return {"mapping": self._link.invoke(self._base(s)).strip()}

    def _node_generate(self, s: SQLState) -> dict:
        attempts = s.get("attempts", 0) + 1
        set_step("repair" if s.get("error") else "generate", attempt=attempts)
        if s.get("error"):
            raw = self._fix.invoke({**self._base(s), "sql": s.get("sql", ""), "error": s["error"]})
        else:
            shots = []
            for q, sql in s.get("examples") or []:
                shots += [HumanMessage(q), AIMessage(f"```sql\n{sql}\n```")]
            raw = self._generate.invoke({**self._base(s), "examples": shots})
        sql = clean_sql(raw)
        if self.tracer is not None:
            self.tracer.event("sql", sql=sql, attempt=attempts)
        return {"sql": sql, "attempts": attempts, "error": None}

    def _node_check(self, s: SQLState) -> dict:
        try:
            sql = check_read_only(s["sql"], self.dialect)
        except UnsafeSQLError as e:
            return self._fail(s, str(e))
        if self.answer_checks:
            same = self_comparisons(sql, self.dialect)
            if same:  # always true: the database accepts it and the join multiplies every row
                keys = join_keys(self.db, tables_in(sql, self.dialect)) if self.db is not None else ""
                return self._fail(s, f"The condition {same[0]} compares a column with itself, so it is always "
                                     "true. Join the two tables on their key columns"
                                  + (f": {keys}." if keys else "."))
        if self.db is None:
            return {"sql": sql, "executed": False}
        try:
            if self.answer_checks:
                dry_run(self.db, sql)  # names and syntax, without running a heavy query
            res = run_query(self.db, sql, self.max_rows)
        except Exception as e:  # the exact database error drives the next attempt
            error = _db_error(e)
            if re.search(r"no such (column|table)", error, re.I):
                fixed = auto_fix_identifiers(self.db, sql, self.dialect)
            elif re.search(r"ambiguous column", error, re.I):
                fixed = qualify_ambiguous(self.db, sql, error, self.dialect)
            else:
                fixed = None
            if fixed and fixed != sql:  # names that differ only in case/underscores: repaired without a model call
                try:
                    res = run_query(self.db, fixed, self.max_rows)
                    log.info("Repaired identifiers: %s -> %s", sql, fixed)
                    return self._checked(s, fixed, res)
                except Exception as e2:  # keep the model's query and its clear error, not the fix's
                    log.info("Identifier repair still fails (%s); keeping the original error", _db_error(e2))
            hint = error_hint(self.db, sql, error, self.dialect)
            if self.error_help is not None:  # a manual page for function / date / syntax errors
                try:
                    manual = self.error_help(error, sql)
                except Exception as e2:  # pragma: no cover - never let a lookup break the loop
                    log.debug("error_help failed: %s", e2)
                    manual = ""
                hint = "\n".join(filter(None, [hint, manual]))
            return self._fail(s, f"{error}\n{hint}" if hint else error)
        return self._checked(s, sql, res)

    def _checked(self, s: SQLState, sql: str, res: QueryResult) -> dict:
        """A query that ran: accept it, or send it back once per kind of suspicious result."""
        ok = {"sql": sql, "result": res, "executed": True}
        prev = (s.get("fallback") or {}).get("sql")
        if prev and _norm_sql(prev) == _norm_sql(sql):  # the model stands by its query: accept it
            return ok
        problem = None
        seen = s.get("checked") or []
        # The grain checks have their own budget: when database errors used up the attempts,
        # a query that runs but multiplies its totals still gets its re-asks.
        bonus = 0
        if self.grain_plan:
            problem = self._grain_problem(sql, seen)
            if problem is not None:  # the rewrite gets its own try plus one repair of a small error
                bonus = max(0, s.get("attempts", 0) + 2 - self._limit(s))
        if problem is None and s.get("attempts", 0) >= self._limit(s):
            return ok
        if problem is None and self.answer_checks:
            if "values" not in seen:  # a text filter on a value the column never holds
                missing = missing_values(self.db, sql, self.dialect)
                if missing:
                    problem = ("values", "A filter value does not exist in the database: " + " | ".join(missing)
                               + ". Use the stored value that means the same thing (a code, an abbreviation, "
                               "another spelling). If none fits, return the same query unchanged.")
            if problem is None:
                problem = next(((k, m) for k, m in result_problems(s["question"], sql, res.columns, res.rows)
                                if k not in seen), None)
        if problem is None and self.review and not s.get("reviewed"):
            set_step("review")
            reply = self._review.invoke({"question": s["question"], "sql": sql, "n_rows": len(res.rows),
                                         "table": rows_as_text(res.columns, res.rows, limit=5)})
            good, reason = parse_review(reply)
            if not good:
                problem = ("review", f"A reviewer says the query does not answer the question: {reason} "
                                     "If you are sure it does, return the same query unchanged.")
        if problem is None:
            return ok
        kind, message = problem
        out = self._fail({**s, "sql": sql}, f"Answer check: {message}")
        out.update(sql=sql, checked=[*(s.get("checked") or []), kind], bonus=s.get("bonus", 0) + bonus,
                   reviewed=bool(s.get("reviewed")) or kind == "review",
                   # the first query that ran is kept if the re-ask does no better; never one that
                   # a structural check showed to be wrong (multiplied totals, a count of 1 per group)
                   fallback=s.get("fallback") or (None if kind in _KNOWN_WRONG else {"sql": sql, "result": res}))
        return out

    def _grain_problem(self, sql: str, seen: list) -> tuple[str, str] | None:
        """A LEFT JOIN undone by WHERE, or totals multiplied by joining two many-row tables."""
        if "optional_side" not in seen:
            hits = optional_side_filters(sql, self.dialect, self.db)
            if hits:
                cond, table = hits[0]
                return ("optional_side", f"The condition {cond} is in WHERE, but {table} is on the optional "
                        f"side of a LEFT JOIN, so every row with no matching {table} row is dropped and the "
                        "LEFT JOIN acts as an inner join. Move the condition into the CTE that builds "
                        f"{table}, or into the ON clause. If dropping those rows is what the question "
                        "asks, return the same query unchanged.")
        if "count_grouped" not in seen:
            hits = counted_group_keys(sql, self.dialect)
            if hits:
                m = re.search(r"DISTINCT\s+(?:\w+\.)?([\w\"]+)", hits[0], re.I)
                key = m.group(1) if m else "the key"
                return ("count_grouped", f"{hits[0]} is counted inside groups that include that same column, so "
                        "every group counts 1 and the query returns one row per group instead of one number. "
                        f"To count the distinct {key} values whose groups pass the condition, keep the "
                        "GROUP BY ... HAVING query as a subquery that returns that column and count it outside: "
                        f"SELECT COUNT(DISTINCT {key}) FROM (SELECT {key}, ... GROUP BY {key}, ... HAVING ...) t. "
                        f"Use COUNT(DISTINCT {key}), not COUNT(*): one {key} can have several groups. "
                        "If one row per group is what the question asks, return the same query unchanged.")
        if seen.count("fanout") < 2 and self.db is not None:  # a 7B often just reshuffles its CTEs once
            sites = fanout_sites(self.db, sql, self.dialect)
            if sites:
                a, b, parent, cte, others = sites[0]
                again = "still " if "fanout" in seen else ""
                where = f"The CTE {cte} {again}joins" if cte else f"This query {again}joins"
                text = (f"{where} {a} and {b} in the same SELECT, and both have several rows per {parent}, so each "
                        f"{a} row is repeated once per matching {b} row and every SUM or COUNT over them is "
                        f"multiplied. Never join {a} and {b} to each other. Total each one per {parent} in its own "
                        f"CTE that reads only that table, then join both totals to {parent}")
                if cte:
                    text += f"; delete {cte} and do not reuse it"
                text += ". "
                if others:
                    text += (f"{', '.join(others)} also has several rows per row of {a} or {b}: total it per "
                             f"{parent} in its own CTE too, joining only to its own parent table inside that CTE. ")
                return ("fanout", text + "Shape:\n" + fanout_shape(self.db, a, b, parent))
        return None

    def _fail(self, s: SQLState, error: str) -> dict:
        log.info("SQL attempt %s failed: %s", s.get("attempts"), error)
        if self.tracer is not None:
            self.tracer.event("check", ok=False, attempt=s.get("attempts"), sql=s.get("sql", ""), error=error)
        return {"error": error, "failures": [*s.get("failures", []), (s.get("sql", ""), error)]}

    def _limit(self, s: SQLState) -> int:
        return self.max_attempts + s.get("bonus", 0)

    def _route(self, s: SQLState) -> str:
        if s.get("error") and s.get("attempts", 0) < self._limit(s):
            return "retry"
        return "done"

    def _build_graph(self):
        from langgraph.graph import END, StateGraph

        g = StateGraph(SQLState)
        g.add_node("link", self._node_link)
        g.add_node("generate", self._node_generate)
        g.add_node("check", self._node_check)
        g.set_entry_point("link")
        g.add_edge("link", "generate")
        g.add_edge("generate", "check")
        g.add_conditional_edges("check", self._route, {"retry": "generate", "done": END})
        return g.compile()

    # -- public -----------------------------------------------------------------------

    def run(
        self,
        question: str,
        context: str = "",
        schema: str = "",
        examples: list[tuple[str, str]] | None = None,
        database: str | None = None,
        context_fn=None,
    ) -> SQLResult:
        """``examples`` are golden (question, sql) pairs, shown to the generator as earlier turns.

        With a catalog, ``database`` names the database to use; otherwise it is routed.
        ``context_fn(database_name, question)`` returns extra context (passages from the
        documents about that database) once the database is known.
        """
        candidates: list[str] = []
        set_question(question)
        if self.catalog is not None:
            if database:
                name = database
            else:
                name, candidates = self.route(question)
            self.use_database(name)
        if context_fn is not None:
            extra = context_fn(self.database_name, question)
            context = f"{context}\n\n{extra}".strip() if extra else context
        context = self._with_value_hints(question, context)
        context = self._with_patterns(question, context)
        base = {
            "question": question,
            "schema": self.schema_text(schema, question, context),
            "context": context,
            "examples": list(examples or []),
            "attempts": 0,
            "failures": [],
        }
        if _ASKS_FOR_ALL.search(question):
            base["hint"] = ALL_INFO_HINT
        base["paths"] = self._paths_text()
        limit = {"recursion_limit": 4 * self.max_attempts + 20}
        set_step("", candidate=0)
        states = [_settle(self._graph.invoke(base, limit))]
        if self.sql_candidates > 1 and self.db is not None:
            shots = list(examples or [])
            for i in range(1, self.sql_candidates):
                hints = [base.get("hint"), CANDIDATE_HINTS[(i - 1) % len(CANDIDATE_HINTS)]]
                if self.independent_plans:  # one misreading must not carry into every candidate
                    hints.append(INDEPENDENT_HINT)
                variant = {
                    **base,
                    "mapping": "" if self.independent_plans else states[0].get("mapping", ""),
                    "hint": " ".join(filter(None, hints)),
                    # different example order/subset per candidate (MCS-SQL)
                    "examples": shots[i % len(shots):] + shots[: i % len(shots)] if shots else [],
                }
                set_step("", candidate=i)
                states.append(_settle(self._graph.invoke(variant, limit)))
        state = vote(states)
        result = self._result(question, state, candidates, len(states))
        if self.tracer is not None:
            self.tracer.event("result", sql=result.sql, ok=result.ok, error=result.error, attempts=result.attempts,
                              executed=result.executed, database=result.database, schema_shown=dict(self.last_schema))
        result.alternatives = [st.get("sql", "") for st in states if st.get("executed") and not st.get("error")]
        return result

    def repair_from(self, question: str, sql: str, schema: str = "", context: str = "",
                    max_repairs: int = 2) -> dict:
        """Run a saved query and, if the database rejects it, send the real error back for up
        to ``max_repairs`` corrections. Nothing else is fed back: no expected rows, no answer key.

        The first attempt is exactly ``sql``, so a replay measures what database feedback alone
        repairs. ``schema`` and ``context`` should be what the original run gave the model; the
        SQL techniques are appended to the context the same way ``run`` does.
        """
        start = time.monotonic()
        set_question(question)
        set_step("replay", attempt=1)
        st: dict = {
            "question": question,
            "schema": f"-- Provided by the user:\n{schema.strip()}" if schema.strip() else self.schema_text("", question),
            "context": self._with_patterns(question, context),
            "examples": [], "mapping": "", "sql": sql, "attempts": 1, "failures": [],
        }
        if _ASKS_FOR_ALL.search(question):
            st["hint"] = ALL_INFO_HINT
        st.update(self._node_check(st))
        first_error = st.get("error")
        auto_fixed = bool(not first_error and _norm_sql(st["sql"]) != _norm_sql(sql))
        while st.get("error") and st["attempts"] < 1 + max_repairs:
            st.update(self._node_generate(st))
            st.update(self._node_check(st))
        st = _settle(st)
        out = {
            "question": question,
            "original_sql": sql,
            "pred_sql": st.get("sql", ""),
            "executed": bool(st.get("executed")) and not st.get("error"),
            "error": st.get("error") or "",
            "first_error": first_error or "",
            "attempts": st["attempts"],
            "repairs": st["attempts"] - 1,
            "auto_fixed": auto_fixed,  # identifiers repaired without a model call
            "errors": [e for _, e in st.get("failures", [])],
            "seconds": round(time.monotonic() - start, 2),
        }
        if self.tracer is not None:
            self.tracer.event("replay", **{k: v for k, v in out.items() if k != "question"})
        return out

    def _paths_text(self) -> str:
        """Key paths between the tables in the prompt, for the grain plan (empty when it is off)."""
        if not self.grain_plan or self.db is None:
            return ""
        try:
            text = join_paths(self.db, list(self.last_schema) or None)
        except Exception as e:  # pragma: no cover - odd metadata must not break a question
            log.debug("join paths failed: %s", e)
            return ""
        return f"\n\nKey paths (from the database):\n{text}" if text else ""

    def _with_patterns(self, question: str, context: str) -> str:
        """Append the SQL techniques that fit the question, with worked examples."""
        if self._patterns is None:
            return context
        from .sqlpatterns import format_patterns

        text = format_patterns(self._patterns.search(question), dialect=self.dialect)
        if not text:
            return context
        return f"{context}\n\n{text}" if context else text

    def _with_value_hints(self, question: str, context: str) -> str:
        """Append the database values the question mentions (CHESS-style value retrieval)."""
        if not self.value_lookup or self.db is None:
            return context
        index = self._values_for(self.database_name or str(self.db._engine.url), self.db)
        if not index:
            return context
        hints = format_value_hints(index.match(f"{question}\n{context}"))
        if not hints:
            return context
        return f"{context}\n\n{hints}" if context else hints

    def _result(self, question: str, state: dict, candidates: list[str], n_candidates: int) -> SQLResult:
        res = state.get("result")
        failures = list(state.get("failures", []))
        error = state.get("error")
        return SQLResult(
            question=question,
            sql=state.get("sql", ""),
            dialect=self.dialect,
            mapping=state.get("mapping", ""),
            columns=res.columns if res else [],
            rows=res.rows if res else [],
            truncated=res.truncated if res else False,
            executed=bool(state.get("executed")),
            attempts=state.get("attempts", 0),
            error=error,
            failures=failures[:-1] if error else failures,
            database=self.database_name if self.catalog is not None else "",
            candidates=candidates,
            votes=state.get("votes", 0),
            n_candidates=n_candidates,
        )

    def summary_inputs(self, r: SQLResult) -> dict:
        return {
            "question": r.question,
            "sql": r.sql,
            "n_rows": len(r.rows),
            "more": ", more were cut off" if r.truncated else "",
            "table": rows_as_text(r.columns, r.rows, self.max_rows),
        }

    def summarize(self, r: SQLResult) -> str:
        set_step("summary")
        return self._summary.invoke(self.summary_inputs(r)).strip()

    def stream_summary(self, r: SQLResult):
        yield from self._summary.stream(self.summary_inputs(r))


def _trim_notes(notes: str, room: int, question: str) -> str:
    """Column notes cut to ``room`` characters, keeping the lines about the question's words first."""
    lines = notes.splitlines()
    head, body = lines[0], [ln for ln in lines[1:] if ln.strip() != "*/"]
    words = set(re.findall(r"[a-z0-9]{3,}", question.lower()))
    body.sort(key=lambda ln: -len(words & set(re.findall(r"[a-z0-9]{3,}", ln.lower()))))
    kept, used = [head], len(head) + 40
    for ln in body:
        if used + len(ln) + 1 > room:
            break
        kept.append(ln)
        used += len(ln) + 1
    return "\n".join(kept + [f"... {len(body) - len(kept) + 1} more column notes not shown */"]) if len(kept) > 1 else ""


_TOP_ONE = {
    "oracle": "ORDER BY ... FETCH FIRST 1 ROWS ONLY (Oracle 12c+; no LIMIT in Oracle)",
    "tsql": "SELECT TOP 1 ... ORDER BY ... (SQL Server has no LIMIT)",
    "teradata": "SELECT TOP 1 ... ORDER BY ...",
}


def top_one(dialect: str) -> str:
    """How this dialect returns only the first row of an ordered result."""
    return _TOP_ONE.get(dialect, "ORDER BY ... LIMIT 1")


def build_schema(parts: dict[str, dict[str, str]], ranked: list[str], budget: int,
                 total: int | None = None, question: str = "") -> tuple[str, dict[str, str]]:
    """The schema text for the prompt, within ``budget`` characters, and what each table got.

    Every chosen table is named with its keys and joins first, so a table the query needs is
    never dropped because an earlier one was wide (BIRD card_games: ``cards`` with its sample
    rows filled the whole budget and ``legalities`` was left out). Then, most relevant first:
    the full CREATE TABLE (else the one-line column list), then sample rows, then column notes.
    """
    level = {n: "keys" for n in ranked}
    used = sum(len(parts[n]["keys"]) + 2 for n in ranked)
    if ranked:  # the most relevant table always gets its full definition
        level[ranked[0]] = "ddl"
        used += len(parts[ranked[0]]["ddl"]) - len(parts[ranked[0]]["keys"])
    for upgrade, field_ in (("columns", "compact"), ("ddl", "ddl")):
        for n in ranked:
            if level[n] == "ddl":
                continue
            cost = len(parts[n][field_]) - len(parts[n]["keys" if level[n] == "keys" else "compact"])
            if used + cost <= budget:
                level[n] = upgrade
                used += cost
    extras: dict[str, list[str]] = {n: [] for n in ranked}
    texts: dict[tuple[str, str], str] = {}
    for field_ in ("notes", "sample"):  # descriptions say what columns mean: before sample rows
        for n in ranked:
            text_ = parts[n][field_]
            if level[n] != "ddl" or not text_:
                continue
            if used + len(text_) + 2 > budget and field_ == "notes":
                text_ = _trim_notes(text_, budget - used - 2, question) if budget - used > 300 else ""
            if text_ and used + len(text_) + 2 <= budget:
                extras[n].append(field_)
                texts[(n, field_)] = text_
                used += len(text_) + 2
    blocks = []
    for n in ranked:
        body = parts[n][{"keys": "keys", "columns": "compact", "ddl": "ddl"}[level[n]]]
        blocks.append("\n\n".join([body] + [texts[(n, f)] for f in extras[n]]))
    text = "\n\n".join(blocks)
    short = [n for n in ranked if level[n] != "ddl"]
    if short:
        text += ("\n\n-- Tables shown as one line (-- name(columns)) have no column types or sample rows "
                 "here; their column names and joins are exact.")
    hidden = (total or len(ranked)) - len(ranked)
    if hidden > 0:
        text += f"\n\n-- {hidden} more tables not shown. Ask about them by name to include them."
    report = {n: "+".join([level[n]] + extras[n]) for n in ranked}
    return text, report


def rerank_by_fit(
    catalog, question: str, candidates: list[str], values_for=None, scores: dict | None = None
) -> tuple[list[str], dict[str, str]]:
    """Reorder routing candidates by how well each database's names and values cover the question.

    Look-alike databases (flight_1 / flight_2 / flight_4) rank close together on
    their cards; checking which one's table and column names and stored values the
    question actually uses separates them. The retrieval rank still counts (a small
    penalty per place), so a clear retrieval winner is not overturned by one word.
    ``values_for(name, db)`` returns a ``ValueIndex`` (cached by the caller) or None.
    ``scores``, if given, is filled with each candidate's ranking score.
    """
    fit, evidence = {}, {}
    for n in candidates:
        try:
            db = catalog.open(n)
            fit[n], evidence[n] = schema_fit(question, db, values_for(n, db) if values_for else None, name=n)
        except Exception as e:  # an unreadable database just gets no evidence
            log.debug("No routing evidence for %s: %s", n, e)
            fit[n], evidence[n] = 0.0, ""
    order = {n: i for i, n in enumerate(candidates)}
    adjusted = {n: fit[n] - 0.05 * order[n] for n in candidates}
    ranked = sorted(candidates, key=lambda n: (-adjusted[n], order[n]))
    if scores is not None:
        scores.update(adjusted)  # the ranking score of each candidate, for the caller's margin check
    return ranked, evidence


# Only for questions that ask for whole rows: as a general rule the 7B model also
# applied it to "give the airline with abbreviation UAL" and returned every column.
_ASKS_FOR_ALL = re.compile(r"\ball\s+(?:the\s+)?(?:information|info|details|data|attributes|columns)\b", re.I)
ALL_INFO_HINT = "The question asks for all the information: SELECT * FROM that one table, with no joins."


# Different instructions per extra candidate, so a temperature-0 model writes
# genuinely different queries (CHASE-SQL uses several generator styles the same way).
INDEPENDENT_HINT = ("Read the question again from the start as if for the first time: decide which entity "
                    "one result row is about before choosing tables.")

CANDIDATE_HINTS = (
    "Write it with explicit JOINs; avoid nested subqueries where a JOIN works.",
    "Write it step by step with CTEs (WITH ...), one CTE per intermediate result.",
    "Before writing, re-check which table each column belongs to and the exact filter values.",
    "Prefer a simple query: no extra columns, no extra joins, only what the question asks for.",
)


def _run_settings(llm, agent) -> dict:
    """What a trace needs to reproduce a run: model, sampling, limits, agent version."""
    import hashlib
    import os

    get = lambda *names: next((getattr(llm, n) for n in names if getattr(llm, n, None) is not None), None)  # noqa: E731
    code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    return {
        "model": get("model", "model_name"),
        "temperature": get("temperature"),
        "seed": get("seed"),
        "num_ctx": get("num_ctx"),
        "num_predict": get("num_predict", "max_tokens"),
        "agent_version": os.getenv("DOCQA_AGENT_VERSION") or f"sql.py {code}",
        "dialect": agent.dialect,
        "schema_chars": agent.schema_chars,
        "max_attempts": agent.max_attempts,
        "schema_linking": agent.schema_linking,
        "patterns": agent._patterns is not None,
        "answer_checks": agent.answer_checks,
        "answer_shape": agent.answer_shape,
        "grain_plan": agent.grain_plan,
    }


def _norm_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip().rstrip(";").lower()


_KNOWN_WRONG = {"fanout", "count_grouped"}


def _settle(state: dict) -> dict:
    """After the loop: if the re-asked query failed, keep the earlier one that ran.

    A re-ask that runs is accepted as the answer (it is the corrected one); the
    earlier query only comes back when every later attempt ended in an error.
    """
    fb = state.get("fallback")
    if fb and state.get("error"):
        failures = list(state.get("failures", []))
        return {**state, "sql": fb["sql"], "result": fb["result"], "executed": True, "error": None,
                "failures": failures}
    return state


def _result_key(res: QueryResult) -> tuple:
    rows = [tuple(round(v, 6) if isinstance(v, float) else v for v in r) for r in res.rows]
    return tuple(sorted(map(repr, rows)))


def vote(states: list[dict]) -> dict:
    """The candidate whose result most candidates agree on (self-consistency, as in MCS-SQL / CHASE-SQL).

    Only queries that ran count; a non-empty result beats an empty one on ties,
    and the earliest candidate wins a remaining tie. With one state it is returned as is.
    """
    if len(states) == 1:
        return states[0]
    ok = [(i, s) for i, s in enumerate(states) if s.get("executed") and not s.get("error") and s.get("result")]
    if not ok:
        return states[0]
    groups: dict[tuple, list[int]] = {}
    for i, s in ok:
        groups.setdefault(_result_key(s["result"]), []).append(i)
    best = max(groups.items(), key=lambda kv: (len(kv[1]), kv[0] != (), -kv[1][0]))
    chosen = dict(states[best[1][0]])
    chosen["votes"] = len(best[1])
    return chosen


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _db_error(e: Exception) -> str:
    orig = getattr(e, "orig", None)  # SQLAlchemy wraps the driver's error
    return _first_line(orig or e)


def format_sql_answer(r: SQLResult, summary: str = "") -> str:
    """The text shown (and remembered in chat history) for a SQL answer."""
    parts = []
    if summary:
        parts.append(summary)
    if r.database:
        others = [c for c in r.candidates if c != r.database]
        parts.append(f"Database: **{r.database}**" + (f" (also considered: {', '.join(others)})" if others else ""))
    if r.mapping:
        parts.append("**How the question maps to your schema**\n" + r.mapping)
    if r.sql:
        parts.append(f"```sql\n{r.sql}\n```")
    if not r.ok:
        parts.append(f"I couldn't get a working query after {r.attempts} attempt(s). Last error: {r.error}")
    elif not r.executed:
        parts.append(
            f"Checked as valid read-only {dialect_name(r.dialect)}, but not run: no database is connected "
            "(set DOCQA_DATABASE_URL to run it)."
        )
    elif r.attempts > 1:
        parts.append(f"(Fixed automatically after {r.attempts - 1} failed attempt(s).)")
    return "\n\n".join(parts)


GOLDEN_FILE = "golden_queries.sql"


def append_golden_example(docs_dir: Path, question: str, sql: str, filename: str = GOLDEN_FILE) -> Path:
    """Save a verified question + query to ``<docs>/golden_queries.sql``; the next sync indexes it.

    Keeping examples as a file in the docs folder (instead of a separate vector store
    filled from the UI) means they survive a rebuilt index, can be reviewed and edited
    like any other file, and are synced with the same add / update / delete rules.
    """
    question = " ".join(question.split())
    if not question or not sql.strip():
        raise ValueError("A golden example needs both a question and a query.")
    body = sql.strip().rstrip(";").rstrip() + ";"  # one statement, so its only ';' ends the example
    path = Path(docs_dir) / filename
    existing = path.read_text(encoding="utf-8") if path.exists() else (
        "-- Golden SQL examples: verified questions and queries the agent copies the style of.\n"
        "-- Format: a '-- Question:' line, then the query, ending with ';'.\n"
    )
    sep = "" if existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
    path.write_text(f"{existing}{sep}-- Question: {question}\n{body}\n", encoding="utf-8")
    return path


DESCRIBE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write a data dictionary for business users. From the table definition and sample rows, "
            "write Markdown in exactly this shape:\n"
            "## <table name>\n"
            "<one sentence: what one row of this table is>\n"
            "- `<column>` (<type>): <what it means; list the codes you can see in the sample rows>\n"
            "Joins: <how it links to other tables, from the foreign keys>\n\n"
            "Mark anything you inferred rather than read with '(guess)'. Do not invent columns. Be brief.",
        ),
        ("human", "{table_info}"),
    ]
)

DRAFT_HEADER = (
    "# Data dictionary (draft)\n\n"
    "Generated by `docqa describe-db` from the live schema and a few sample rows. Lines marked "
    "(guess) were inferred by the model. Correct them and add your business rules (status codes, "
    "KPI formulas, join paths): the agent reads this file like any other document.\n"
)


def draft_data_dictionary(llm: BaseChatModel, db: Database, tables: list[str] | None = None) -> str:
    """One model call per table, describing it in business terms (idea from SmolSQLAgents' documentation agent).

    For organisations whose database has no data dictionary yet: the draft gives the
    SQL route business meaning to retrieve, and a person corrects it once.
    """
    infos = db.table_infos()
    names = tables or list(infos)
    missing = [n for n in names if n not in infos]
    if missing:
        raise DatabaseError(f"Tables not found: {', '.join(missing)}")
    chain = DESCRIBE_PROMPT | llm | StrOutputParser()
    sections = [chain.invoke({"table_info": infos[n]}).strip() for n in names]
    return DRAFT_HEADER + "\n" + "\n\n".join(sections) + "\n"
