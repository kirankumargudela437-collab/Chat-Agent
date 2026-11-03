"""Command line: ``docqa sync | watch | ask | chat``."""

from __future__ import annotations

import argparse
import logging
import sys
import time

from . import build_agent
from .chain import Answer
from .config import Settings
from .indexing import open_record_manager, open_vectorstore, sync_documents
from .providers import ProviderError

CONNECT_HINT = (
    "Could not reach the model provider. For local mode, start Ollama and pull the models:\n"
    "  ollama pull {llm}\n  ollama pull {embed}"
)


def _settings(args: argparse.Namespace) -> Settings:
    s = Settings.from_env(args.env_file)
    if getattr(args, "docs", None):
        s.docs_dir = args.docs
    return s


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
    if ans.cited_docs:
        print("Sources:\n" + ans.sources_text())
    elif ans.uncited:
        print("Sources: none cited. This answer may not come from your documents.")


def cmd_ask(args) -> int:
    s = _settings(args)
    if args.sync:
        print(sync_documents(s).summary(), file=sys.stderr)
    ans = build_agent(s).ask(args.question)
    print(ans.answer)
    _print_answer(ans, False)
    return 0


def cmd_chat(args) -> int:
    s = _settings(args)
    if args.sync:
        print(sync_documents(s).summary())
    agent = build_agent(s)
    print(f"Chatting with {s.llm_provider}:{s.llm_model} over {s.docs_dir}.")
    print("Commands: /sync re-index, /reset clear history, /quit exit.\n")
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
            continue
        print("bot> ", end="", flush=True)
        final = None
        for part in agent.stream(q):
            if isinstance(part, Answer):
                final = part
            else:
                print(part, end="", flush=True)
        print()
        if final:
            _print_answer(final, args.verbose)
        print()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="docqa", description=__doc__)
    p.add_argument("--env-file", default=".env", help="settings file (default: .env)")
    p.add_argument("--docs", type=lambda x: __import__("pathlib").Path(x), help="override DOCQA_DOCS_DIR")
    p.add_argument("-v", "--verbose", action="store_true", help="show rewritten follow-up queries")
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
    a.set_defaults(func=cmd_ask)

    c = sub.add_parser("chat", help="interactive chat with follow-up questions")
    c.add_argument("--sync", action="store_true", help="sync before starting")
    c.set_defaults(func=cmd_chat)
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
