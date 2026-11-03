"""Streamlit web app: ``docqa ui`` (or ``streamlit run docqa/app.py``).

One chat over your documents, code and database. The sidebar picks the files
folder, an optional read-only database, the SQL dialect, pasted table
definitions and the route; it also syncs the folder and saves golden SQL
examples. Each answer shows the route it took, the sources it cited, and for
SQL the query, the result table and any automatic fixes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import streamlit as st

from docqa.agent import Agent
from docqa.config import Settings
from docqa.indexing import EmptySyncError, open_record_manager, open_vectorstore, sync_documents
from docqa.sql import DIALECT_NAMES, SQLAgent, append_golden_example, dialect_from_url, open_database

ROUTES = {"Auto": None, "SQL": "sql", "Code": "code", "Docs": "doc"}
DIALECTS = ["postgres", "mysql", "tsql", "snowflake", "bigquery", "oracle", "sqlite", "redshift", "duckdb", "databricks"]


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env-file", default=".env")
    p.add_argument("--docs")
    p.add_argument("--db")
    return p.parse_known_args(sys.argv[1:])[0]


def _settings(env_file: str, docs_dir: str, db_url: str, dialect: str) -> Settings:
    s = Settings.from_env(env_file)
    s.docs_dir = Path(docs_dir)
    if db_url and _is_folder(db_url):  # a folder of SQLite files: each question goes to the best match
        s.database_dir, s.database_url, s.sql_dialect = Path(db_url), None, "sqlite"
        return s
    s.database_dir = None
    s.database_url = db_url or None
    s.sql_dialect = dialect_from_url(db_url) if db_url else dialect
    return s


def _is_folder(value: str) -> bool:
    return "://" not in value and Path(value).expanduser().is_dir()


@st.cache_resource(show_spinner=False)
def _catalog(env_file: str, folder: str):
    from docqa import open_catalog

    s = Settings.from_env(env_file)
    s.database_dir = Path(folder)
    return open_catalog(s)


@st.cache_resource(show_spinner=False)
def _index(env_file: str, docs_dir: str):
    s = _settings(env_file, docs_dir, "", "postgres")
    return open_vectorstore(s), open_record_manager(s)


@st.cache_resource(show_spinner=False)
def _llm(env_file: str):
    from docqa.providers import make_llm

    return make_llm(Settings.from_env(env_file))


@st.cache_resource(show_spinner=False)
def _database(db_url: str, tables: tuple[str, ...]):
    return open_database(db_url, tables)


def _agent(env_file: str, docs_dir: str, db_url: str, dialect: str, schema: str) -> Agent:
    """One Agent (and chat history) per browser session and configuration."""
    s = _settings(env_file, docs_dir, db_url, dialect)
    key = (env_file, docs_dir, db_url, s.sql_dialect)
    if st.session_state.get("agent_key") != key:
        llm = _llm(env_file)
        vs, _ = _index(env_file, docs_dir)
        catalog = _catalog(env_file, str(s.database_dir)) if s.database_dir is not None else None
        db = _database(db_url, s.sql_tables) if s.database_url else None
        sql = SQLAgent(llm, db, dialect=s.sql_dialect, max_attempts=s.sql_max_attempts,
                       max_rows=s.sql_max_rows, schema_linking=s.sql_schema_linking,
                       max_tables=s.sql_max_tables, schema_chars=s.sql_schema_chars,
                       catalog=catalog, db_router=s.sql_db_router, db_candidates=s.sql_db_candidates)
        st.session_state.agent = Agent(llm, vs, top_k=s.top_k, sql_agent=sql, hybrid=s.hybrid_search)
        st.session_state.agent_key = key
        st.session_state.messages = []
    agent = st.session_state.agent
    agent.schema = schema
    return agent


def _render(msg: dict, i: int, docs_dir: str) -> None:
    st.markdown(msg["content"])
    if msg.get("columns"):
        st.dataframe([dict(zip(msg["columns"], r)) for r in msg["rows"]], use_container_width=True)
        if msg.get("truncated"):
            st.caption(f"First {len(msg['rows'])} rows shown (DOCQA_SQL_MAX_ROWS).")
    meta = [f"Answered from: {msg['route'].lower()}"] if msg.get("route") else []
    if msg.get("sources"):
        meta.append("Sources: " + "; ".join(msg["sources"]))
    elif msg.get("uncited"):
        meta.append("Sources: none cited. This answer may not come from your files.")
    if meta:
        st.caption(" · ".join(meta))
    if msg.get("failures"):
        with st.expander(f"{len(msg['failures'])} failed attempt(s), fixed automatically"):
            for sql, err in msg["failures"]:
                st.code(sql or "(empty)", language="sql")
                st.caption(err)
    if msg.get("golden_sql"):
        if st.button("Save as golden example", key=f"golden-{i}"):
            path = append_golden_example(Path(docs_dir), msg["golden_question"], msg["golden_sql"])
            st.success(f"Saved to {path.name}. Sync the folder to use it.")


def main() -> None:
    st.set_page_config(page_title="docqa", layout="wide")
    args = _args()
    base = Settings.from_env(args.env_file)

    with st.sidebar:
        st.header("docqa")
        st.caption(f"Chat: {base.llm_provider}:{base.llm_model} · Embeddings: {base.embed_provider}:{base.embed_model}")
        docs_dir = st.text_input("Files folder", value=str(args.docs or base.docs_dir))
        db_url = st.text_input(
            "Database URL or folder of SQLite files (optional, opened read-only)",
            value=args.db or base.database_url or (str(base.database_dir) if base.database_dir else ""),
            placeholder="sqlite:///my_company.db, postgresql://user:pass@host/db, or benchmarks/spider/spider_data/database",
            help="A folder of SQLite files (like Spider's database/ folder) routes each question to the best database.",
        )
        default_dialect = base.sql_dialect if base.sql_dialect in DIALECTS else "postgres"
        dialect = st.selectbox(
            "SQL dialect",
            DIALECTS,
            index=DIALECTS.index(dialect_from_url(db_url) if db_url and dialect_from_url(db_url) in DIALECTS else default_dialect),
            format_func=lambda d: DIALECT_NAMES.get(d, d),
            disabled=bool(db_url),
            help="Taken from the database when one is connected.",
        )
        schema = st.text_area(
            "Tables and columns (optional)",
            height=140,
            placeholder="CREATE TABLE ... or\norders (order_id, cust_id, order_dt, net_amt, status_cd)",
        )
        route_label = st.radio("Route", list(ROUTES), horizontal=True)

        try:
            agent = _agent(args.env_file, docs_dir, db_url.strip(), dialect, schema)
        except Exception as e:
            st.error(str(e))
            st.stop()

        uploads = st.file_uploader(
            "Add files (documents, code, .sql, schemas)",
            accept_multiple_files=True,
            help="Saved into an 'uploads' folder inside the files folder, then indexed.",
        )
        allow_empty = st.checkbox("Allow a sync to empty the index", value=False)
        save_uploads = bool(uploads) and st.button("Save uploads and sync", use_container_width=True)
        if save_uploads:
            target = Path(docs_dir) / "uploads"
            target.mkdir(parents=True, exist_ok=True)
            for f in uploads:
                (target / Path(f.name).name).write_bytes(f.getvalue())
        sync_clicked = st.button("Sync folder", use_container_width=True)
        if save_uploads or sync_clicked:
            s = _settings(args.env_file, docs_dir, db_url, dialect)
            vs, rm = _index(args.env_file, docs_dir)
            with st.spinner("Indexing new and changed files..."):
                try:
                    st.session_state.sync_summary = sync_documents(s, vs, rm, allow_empty=allow_empty).summary()
                except (EmptySyncError, FileNotFoundError) as e:
                    st.session_state.sync_summary = f"error: {e}"
                except Exception as e:  # embedding server down, etc.
                    hint = " Is Ollama running, with the embedding model pulled?" if "connect" in str(e).lower() else ""
                    st.session_state.sync_summary = f"error: {e}{hint}"
            agent.refresh()
        if st.session_state.get("sync_summary"):
            st.code(st.session_state.sync_summary, language=None)

        if st.button("Clear chat", use_container_width=True):
            agent.reset()
            st.session_state.messages = []
            st.rerun()

        with st.expander("Add a golden SQL example"):
            gq = st.text_input("Question", key="gq")
            gs = st.text_area("Verified SQL", key="gs", height=120)
            if st.button("Save example"):
                try:
                    path = append_golden_example(Path(docs_dir), gq, gs)
                    st.success(f"Saved to {path.name}. Sync the folder to use it.")
                except (ValueError, OSError) as e:
                    st.error(str(e))

    st.session_state.setdefault("messages", [])
    if not st.session_state.messages:
        st.markdown("#### Ask about your documents, code or data")
        st.caption(
            "Sync the folder first. Business questions are answered from your documents, code questions "
            "from your source files, and data questions become a checked, read-only SQL query. "
            "Pick a route in the sidebar to force one."
        )
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            _render(msg, i, docs_dir)

    question = st.chat_input("Ask about your documents, code or data... (/run SELECT ... runs your own query)")
    if not question:
        return
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.status("Thinking...", expanded=False) as status:
            try:
                ans = agent.ask(question, ROUTES[route_label])
            except Exception as e:
                hint = " Is Ollama running, with the models pulled?" if "connect" in str(e).lower() else ""
                status.update(label="Failed", state="error")
                st.error(f"{e}{hint}")
                st.session_state.messages.pop()
                return
            if ans.standalone_question != question:
                st.write(f"Searched for: {ans.standalone_question}")
            st.write(f"Route: {ans.route}")
            if ans.sql is not None:
                st.write(f"SQL attempts: {ans.sql.attempts}")
            status.update(label=f"Answered from {ans.route.lower()}", state="complete")

        r = ans.sql
        msg = {
            "role": "assistant",
            "content": ans.answer,
            "route": ans.route,
            "sources": [f"[{i}] " + _label(d) for i, d in ans.cited_docs],
            "uncited": ans.uncited,
        }
        if r is not None:
            msg.update(
                columns=r.columns if r.executed else [],
                rows=[list(map(_cell, row)) for row in r.rows],
                truncated=r.truncated,
                failures=r.failures,
            )
            if r.ok:
                msg.update(golden_question=r.question, golden_sql=r.sql)
        st.session_state.messages.append(msg)
        _render(msg, len(st.session_state.messages) - 1, docs_dir)


def _label(doc) -> str:
    from docqa.chain import citation_label

    return citation_label(doc)


def _cell(v):
    return v if isinstance(v, (int, float, str, type(None))) else str(v)


main()
