"""Command line: ``docqa sync | watch | ask | chat | sample | ui``."""

from __future__ import annotations

import argparse
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

from . import build_agent
from .chain import Answer
from .config import Settings
from .indexing import open_record_manager, open_vectorstore, sync_documents
from .providers import ProviderError
from .sql import append_golden_example

CONNECT_HINT = (
    "Could not reach the model provider. For local mode, start Ollama and pull the models:\n"
    "  ollama pull {llm}\n  ollama pull {embed}"
)


def _settings(args: argparse.Namespace) -> Settings:
    s = Settings.from_env(args.env_file)
    if getattr(args, "docs", None):
        s.docs_dir = args.docs
    if getattr(args, "dialect", None):
        s.sql_dialect = args.dialect.lower()
    if getattr(args, "db", None):
        from .sql import dialect_from_url

        s.database_url = args.db
        s.sql_dialect = dialect_from_url(args.db)
    return s


def _schema(args) -> str:
    return Path(args.schema).read_text(encoding="utf-8") if getattr(args, "schema", None) else ""


def cmd_sync(args) -> int:
    s = _settings(args)
    print(f"Syncing {s.docs_dir} ({s.embed_provider}:{s.embed_model}) ...")
    print(sync_documents(s, allow_empty=args.allow_empty).summary())
    return 0


def cmd_watch(args) -> int:
    """Poll the folder and re-sync when any file's size or mtime changes.

    A failed sync (model server restarting, a file vanishing mid-scan) is logged and
    retried on the next tick instead of stopping the watcher.
    """
    s = _settings(args)
    vs, rm = open_vectorstore(s), open_record_manager(s)
    from .loaders import iter_files

    def snapshot():
        snap = set()
        for p in iter_files(s.docs_dir, s.extensions):
            try:
                st = p.stat()
            except FileNotFoundError:  # deleted between listing and stat
                continue
            snap.add((p, st.st_mtime_ns, st.st_size))
        return snap

    last = None
    print(f"Watching {s.docs_dir} every {args.interval}s (Ctrl+C to stop)")
    try:
        while True:
            try:
                snap = snapshot()
                if snap != last:
                    res = sync_documents(s, vs, rm)
                    print(time.strftime("%H:%M:%S"), res.summary(), flush=True)
                    # files that failed to load (locked, still downloading) are retried next tick
                    last = None if res.report.errors else snap
            except KeyboardInterrupt:
                raise
            except Exception as e:  # keep watching; `last` is unchanged so the next tick retries
                print(time.strftime("%H:%M:%S"), f"sync failed, retrying in {args.interval}s: {e}", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


def _print_answer(ans: Answer, show_query: bool) -> None:
    if show_query and ans.standalone_question != ans.question:
        print(f"  (searched for: {ans.standalone_question})")
    r = ans.sql
    if r is not None and r.executed and r.columns:
        more = f", first {len(r.rows)} shown" if r.truncated else ""
        print(f"\nResult ({len(r.rows)} rows{more}):\n" + r.table_text())
    if show_query and r is not None and r.failures:
        for i, (bad, err) in enumerate(r.failures, 1):
            print(f"  (attempt {i} failed: {err})")
    print(f"Answered from: {ans.route.lower()}")
    if ans.cited_docs:
        print("Sources:\n" + ans.sources_text())
    elif ans.uncited:
        print("Sources: none cited. This answer may not come from your documents.")


def cmd_ask(args) -> int:
    s = _settings(args)
    if args.sync:
        print(sync_documents(s).summary(), file=sys.stderr)
    ans = build_agent(s, schema=_schema(args)).ask(args.question, args.route)
    print(ans.answer)
    _print_answer(ans, False)
    return 0


def cmd_chat(args) -> int:
    s = _settings(args)
    if args.sync:
        print(sync_documents(s).summary())
    agent = build_agent(s, schema=_schema(args))
    db = f" and {s.database_url.split('@')[-1]}" if s.database_url else ""
    print(f"Chatting with {s.llm_provider}:{s.llm_model} over {s.docs_dir}{db}.")
    print(
        "Commands: /sql, /code or /doc before a question forces that route; /golden saves the last "
        "query as a golden example; /sync re-index; /reset clear history; /quit exit.\n"
    )
    last_sql = None
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            continue
        if q in ("/quit", "/exit"):
            return 0
        if q == "/reset":
            agent.reset()
            print("History cleared.")
            continue
        if q == "/sync":
            print(sync_documents(s).summary())
            agent.refresh()
            continue
        if q == "/golden":
            if last_sql is None or not last_sql.ok:
                print("No working query to save yet.")
            else:
                path = append_golden_example(s.docs_dir, last_sql.question, last_sql.sql)
                print(f"Saved to {path}. Run /sync to index it.")
            continue
        route = None
        m = re.match(r"^/(sql|code|doc)\s+(.+)$", q, re.IGNORECASE)
        if m:
            route, q = m.group(1), m.group(2)
        print("bot> ", end="", flush=True)
        final = None
        for part in agent.stream(q, route):
            if isinstance(part, Answer):
                final = part
            else:
                print(part, end="", flush=True)
        print()
        if final:
            if final.sql is not None:
                last_sql = final.sql
            _print_answer(final, args.verbose)
        print()


def cmd_sample(args) -> int:
    from .sample import create_sample_workspace

    paths = create_sample_workspace(Path(args.folder))
    print(f"Created a sample workspace in {paths['root']}:")
    for f in sorted(paths["files"]):
        print(f"  {f}")
    print(f"  {paths['db']}  (SQLite database)\n")
    print("Try it:")
    print(f'  docqa --docs "{paths["docs"]}" --db "{paths["db_url"]}" chat --sync')
    print(f'  docqa --docs "{paths["docs"]}" --db "{paths["db_url"]}" ui')
    return 0


def cmd_ui(args) -> int:
    app = Path(__file__).with_name("app.py")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--", "--env-file", args.env_file]
    if args.docs:
        cmd += ["--docs", str(args.docs)]
    if args.db:
        cmd += ["--db", args.db]
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="docqa", description=__doc__)
    p.add_argument("--env-file", default=".env", help="settings file (default: .env)")
    p.add_argument("--docs", type=Path, help="override DOCQA_DOCS_DIR")
    p.add_argument("--db", help="override DOCQA_DATABASE_URL, e.g. sqlite:///my_company.db")
    p.add_argument("--dialect", help="SQL dialect when no database is connected: postgres, mysql, tsql, snowflake, bigquery, oracle, sqlite ...")
    p.add_argument("--schema", help="file with CREATE TABLE statements or a table/column list to write SQL against")
    p.add_argument("-v", "--verbose", action="store_true", help="show rewritten follow-up queries and failed SQL attempts")
    sub = p.add_subparsers(dest="cmd", required=True)

    sy = sub.add_parser("sync", help="index new/changed files, drop deleted ones")
    sy.add_argument("--allow-empty", action="store_true", help="allow a sync of an empty folder to clear the index")
    sy.set_defaults(func=cmd_sync)

    w = sub.add_parser("watch", help="keep the index in sync while files change")
    w.add_argument("--interval", type=float, default=10.0, help="seconds between checks")
    w.set_defaults(func=cmd_watch)

    a = sub.add_parser("ask", help="ask one question")
    a.add_argument("question")
    a.add_argument("--sync", action="store_true", help="sync before answering")
    a.add_argument("--route", default="auto", help="auto (default), sql, code, doc, or a list like code,doc")
    a.set_defaults(func=cmd_ask)

    c = sub.add_parser("chat", help="interactive chat with follow-up questions")
    c.add_argument("--sync", action="store_true", help="sync before starting")
    c.set_defaults(func=cmd_chat)

    sa = sub.add_parser("sample", help="create sample docs, code, SQL and a SQLite database to try the agent")
    sa.add_argument("folder", nargs="?", default="sample_workspace")
    sa.set_defaults(func=cmd_sample)

    u = sub.add_parser("ui", help="open the Streamlit web app (pip install -e \".[ui]\")")
    u.set_defaults(func=cmd_ui)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    try:
        return args.func(args)
    except (ProviderError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # connection errors from Ollama / cloud APIs
        if "connect" in str(e).lower():
            s = Settings.from_env(args.env_file)
            print("error:", e, "\n" + CONNECT_HINT.format(llm=s.llm_model, embed=s.embed_model), file=sys.stderr)
            return 2
        raise
