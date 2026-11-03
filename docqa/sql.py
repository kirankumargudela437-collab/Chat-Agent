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
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TypedDict

import sqlglot
from sqlglot import exp
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

    def table_infos(self) -> dict[str, str]:
        """``CREATE TABLE`` (with foreign keys) plus a few sample rows, per table. Cached."""
        if getattr(self, "_infos", None) is None:
            from sqlalchemy import select
            from sqlalchemy.schema import CreateTable

            infos = {}
            for table in self._metadata.sorted_tables:
                info = str(CreateTable(table).compile(self._engine)).strip()
                if self.sample_rows:
                    try:
                        with self._engine.connect() as conn:
                            rows = conn.execute(select(table).limit(self.sample_rows)).fetchall()
                        sample = rows_as_text([c.name for c in table.columns], [tuple(r) for r in rows], width=40)
                        info += f"\n\n/* {len(rows)} sample rows from {table.name}:\n{sample}\n*/"
                    except Exception as e:  # pragma: no cover - permissions on one table
                        log.debug("No sample rows for %s: %s", table.name, e)
                if self.notes.get(table.name):
                    info += f"\n\n/* Column descriptions for {table.name}:\n{self.notes[table.name]}\n*/"
                infos[table.name] = info
            self._infos = infos
        return self._infos

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


def run_query(db, sql: str, max_rows: int) -> QueryResult:
    """Run one checked query, fetch at most ``max_rows`` rows, always roll back."""
    with db._engine.connect() as conn:
        try:
            # no_parameters: '%' and ':name' inside the SQL are passed to the driver untouched
            res = conn.execution_options(no_parameters=True).exec_driver_sql(sql)
            columns = list(res.keys()) if res.returns_rows else []
            rows = [tuple(r) for r in res.fetchmany(max_rows + 1)] if res.returns_rows else []
        finally:
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
    "- Prefer readable CTEs over nested subqueries; use window functions (ROW_NUMBER, RANK, LAG, LEAD, "
    "SUM() OVER) where they fit; guard divisions with NULLIF and handle NULLs with COALESCE.\n"
    "- Follow the style of the golden examples when one is similar.\n"
    "- Unless the question asks for everything, return at most {max_rows} rows.\n"
    "- Output only the SQL in one ```sql block, no explanation."
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
            "- Gaps: anything the question needs that the schema and context do not define.\n"
            "Do not write the SQL. Be concise.\n\n"
            "Dialect: {dialect_name}\n\nSchema:\n{schema}\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

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
)

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
)

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
        db_candidates: int = 3,
    ):
        """``catalog`` (a ``multidb.DatabaseCatalog``) routes each question to one of many databases."""
        self.llm = llm
        self.catalog = catalog
        if db_router not in ("llm", "retrieval"):
            raise ValueError("DOCQA_DB_ROUTER must be 'llm' or 'retrieval'")
        self.db_router = db_router
        self.db_candidates = max(1, db_candidates)
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
        self.max_attempts = max_attempts
        self.max_rows = max_rows
        self.schema_linking = schema_linking
        self._link = LINK_PROMPT | llm | StrOutputParser()
        self._generate = GENERATE_PROMPT | llm | StrOutputParser()
        self._fix = FIX_PROMPT | llm | StrOutputParser()
        self._summary = SUMMARY_PROMPT | llm | StrOutputParser()
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
        if len(names) <= self.max_tables:
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
        if self.db is not None:
            infos = self.db.table_infos()
            shown, used = [], 0
            for n in self.select_tables(question, context):
                if shown and used + len(infos[n]) > self.schema_chars:
                    break
                shown.append(infos[n])
                used += len(infos[n])
            text = "\n\n".join(shown)
            hidden = len(infos) - len(shown)
            if hidden:
                text += f"\n\n-- {hidden} more tables not shown. Ask about them by name to include them."
            parts.append(f"-- Live database ({dialect_name(self.dialect)}):\n{text}")
        if pasted.strip():
            parts.append(f"-- Provided by the user:\n{pasted.strip()}")
        return "\n\n".join(parts) or "(no schema given: use the table definitions in the context)"

    # -- many databases -------------------------------------------------------------

    def route(self, question: str) -> tuple[str, list[str]]:
        """``(database, candidates)`` for a question, using the catalog's ranking (and the model)."""
        candidates = [n for n, _ in self.catalog.rank(question, self.db_candidates)]
        if self.db_router == "llm" and len(candidates) > 1:
            return self.catalog.choose(question, candidates, self.llm), candidates
        return candidates[0], candidates

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
            "max_rows": self.max_rows,
            "schema": s["schema"],
            "context": s.get("context") or "(none)",
            "mapping": s.get("mapping") or "(not done)",
            "question": s["question"],
        }

    def _node_link(self, s: SQLState) -> dict:
        if not self.schema_linking:
            return {"mapping": ""}
        return {"mapping": self._link.invoke(self._base(s)).strip()}

    def _node_generate(self, s: SQLState) -> dict:
        attempts = s.get("attempts", 0) + 1
        if s.get("error"):
            raw = self._fix.invoke({**self._base(s), "sql": s.get("sql", ""), "error": s["error"]})
        else:
            shots = []
            for q, sql in s.get("examples") or []:
                shots += [HumanMessage(q), AIMessage(f"```sql\n{sql}\n```")]
            raw = self._generate.invoke({**self._base(s), "examples": shots})
        return {"sql": clean_sql(raw), "attempts": attempts, "error": None}

    def _node_check(self, s: SQLState) -> dict:
        try:
            sql = check_read_only(s["sql"], self.dialect)
        except UnsafeSQLError as e:
            return self._fail(s, str(e))
        if self.db is None:
            return {"sql": sql, "executed": False}
        try:
            res = run_query(self.db, sql, self.max_rows)
        except Exception as e:  # the exact database error drives the next attempt
            return self._fail(s, _db_error(e))
        return {"sql": sql, "result": res, "executed": True}

    def _fail(self, s: SQLState, error: str) -> dict:
        log.info("SQL attempt %s failed: %s", s.get("attempts"), error)
        return {"error": error, "failures": [*s.get("failures", []), (s.get("sql", ""), error)]}

    def _route(self, s: SQLState) -> str:
        if s.get("error") and s.get("attempts", 0) < self.max_attempts:
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
    ) -> SQLResult:
        """``examples`` are golden (question, sql) pairs, shown to the generator as earlier turns.

        With a catalog, ``database`` names the database to use; otherwise it is routed.
        """
        candidates: list[str] = []
        if self.catalog is not None:
            if database:
                name = database
            else:
                name, candidates = self.route(question)
            self.use_database(name)
        state = self._graph.invoke(
            {
                "question": question,
                "schema": self.schema_text(schema, question, context),
                "context": context,
                "examples": list(examples or []),
                "attempts": 0,
                "failures": [],
            },
            {"recursion_limit": 4 * self.max_attempts + 10},
        )
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
        return self._summary.invoke(self.summary_inputs(r)).strip()

    def stream_summary(self, r: SQLResult):
        yield from self._summary.stream(self.summary_inputs(r))


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
