"""Command line: ``docqa sync | watch | ask | chat | describe-db | sample | ui | get-benchmark | eval``."""

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
    if getattr(args, "db_dir", None):
        s.database_dir = Path(args.db_dir)
        s.sql_dialect = "sqlite"
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
    if s.database_dir is not None:
        n = len(agent.sql_agent.catalog)
        db = f" and {n} databases in {s.database_dir} (each question goes to the best match)"
    print(f"Chatting with {s.llm_provider}:{s.llm_model} over {s.docs_dir}{db}.")
    print(
        "Commands: /sql, /code or /doc before a question forces that route; /run <SQL> runs your own "
        "query read-only; /golden saves the last query as a golden example; /sync re-index; "
        "/reset clear history; /quit exit.\n"
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


def cmd_describe_db(args) -> int:
    from .providers import make_llm
    from .sql import draft_data_dictionary, open_database

    s = _settings(args)
    if not s.database_url:
        raise ValueError("No database: set DOCQA_DATABASE_URL or pass --db.")
    out = Path(args.out) if args.out else s.docs_dir / "data_dictionary_draft.md"
    if out.exists() and not args.force:
        raise ValueError(f"{out} already exists (you may have edited it). Pass --force to overwrite.")
    db = open_database(s.database_url, s.sql_tables)
    tables = [t.strip() for t in args.tables.split(",") if t.strip()] if args.tables else None
    print(f"Describing {len(tables or db.table_names)} tables with {s.llm_provider}:{s.llm_model} ...")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(draft_data_dictionary(make_llm(s), db, tables), encoding="utf-8")
    print(f"Wrote {out}. Review it, fix the (guess) lines, then run: docqa sync")
    return 0


def cmd_get_benchmark(args) -> int:
    from .benchmark import get_benchmark

    get_benchmark(args.name, Path(args.dest), Path(args.zip) if args.zip else None)
    print(f"Next: docqa eval {args.name} --limit 50")
    return 0


def cmd_eval(args) -> int:
    import json

    from . import open_catalog
    from .benchmark import TRAIN_FILES, evaluate, locate, pick, read_results, route_only, summarize, write_result

    s = _settings(args)
    split = locate(args.name, Path(args.data or f"./benchmarks/{args.name}"), args.questions)
    s.database_dir = split.db_root
    from .benchmark import read_ids

    every = split.load()
    if args.ids:  # a frozen question set: exactly these, in this order
        wanted = read_ids(args.ids)
        by_id = {e.id: e for e in every}
        examples = [by_id[i] for i in wanted if i in by_id]
    else:
        if args.exclude_ids:  # e.g. keep the regression questions out of an unseen set
            skip = set(read_ids(args.exclude_ids))
            every = [e for e in every if e.id not in skip]
        examples = pick(every, args.limit, args.seed, args.only_db)
    if args.save_ids:
        Path(args.save_ids).write_text("\n".join(e.id for e in examples) + "\n", encoding="utf-8")
        print(f"Saved {len(examples)} question ids to {args.save_ids}")
    if not examples:
        raise ValueError("No questions selected.")
    routing = not args.oracle_db
    catalog = open_catalog(s, use_embeddings=routing and not args.no_embeddings)
    k = args.candidates or s.sql_db_candidates
    mode = "route-only" if args.route_only else ("oracle-db" if args.oracle_db else "routed")
    print(
        f"{args.name} {split.questions_file.name}: {len(examples)} questions, {len(catalog)} databases in "
        f"{split.db_root}, mode {mode}" + (f", router {args.router or s.sql_db_router}, top {k}" if routing else "")
    )
    if args.resume:
        out = Path(args.resume)
        done = read_results(out)
        print(f"Resuming {out}: {len(done)} questions already done")
    else:
        out = s.data_dir / "eval" / f"{args.name}-{split.questions_file.stem}-{mode}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        done = {}
    if args.route_only:
        outcomes = route_only(catalog, examples, k, fit=not args.no_fit)
    else:
        from .providers import make_llm
        from .sql import SQLAgent

        print(f"Model: {s.llm_provider}:{s.llm_model}" + ("" if args.no_embeddings or not routing else
                                                           f", embeddings {s.embed_provider}:{s.embed_model}"))
        agent = SQLAgent(
            make_llm(s),
            catalog=catalog,
            max_attempts=s.sql_max_attempts,
            max_rows=1000,  # the benchmark compares full results; don't nudge the model to LIMIT 50
            schema_linking=s.sql_schema_linking and not args.no_linking,
            embeddings=catalog.embeddings,
            max_tables=s.sql_max_tables,
            schema_chars=s.sql_schema_chars,
            db_router=args.router or s.sql_db_router,
            db_candidates=k,
            value_lookup=s.sql_value_lookup and not args.no_values,
            sql_patterns=s.sql_patterns and not args.no_patterns,
            patterns_dir=str(s.sql_patterns_dir) if s.sql_patterns_dir else None,
            sql_candidates=args.sql_candidates or s.sql_candidates,
            route_margin=s.sql_route_margin if args.route_margin is None else args.route_margin,
            route_choices=args.route_choices or s.sql_route_choices,
            answer_checks=s.sql_answer_checks and not args.no_checks,
            review=s.sql_review or args.review,
            independent_plans=s.sql_independent_plans or args.independent,
            answer_shape=s.sql_answer_shape and not args.no_shape,
        )
        bank = None
        if args.shots:
            from .benchmark import ExampleBank

            bank = ExampleBank.from_benchmark(args.name, Path(args.train or split.questions_file.parent))
            if bank is None:
                print(f"No training questions found for --shots (looked for {', '.join(TRAIN_FILES[args.name])}); running zero-shot.")
            else:
                print(f"Few-shot: {args.shots} similar solved questions per question from {len(bank)} training examples")
        docs = None
        if args.docs:
            from .benchmark import DocsContext

            docs = DocsContext(s, Path(args.docs), top_k=s.top_k, patterns=agent._patterns, slim=not args.docs_full)
            docs.reference_folders = [f for f in docs.docs_by_folder if f not in catalog.paths]
            if docs.slim:
                agent.error_help = docs.help_for_error
            print(f"Documents: {args.docs}: {docs.sync.summary().splitlines()[0]}; "
                  f"{len(docs.docs_by_folder) - len(docs.reference_folders)} database folders, "
                  f"reference folders: {', '.join(docs.reference_folders) or 'none'}")
        outcomes = evaluate(agent, split, examples, args.oracle_db, args.timeout, done, bank, args.shots, docs)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with open(out, "a" if args.resume else "w", encoding="utf-8") as fh:
        for i, o in enumerate(outcomes, 1):
            if o.id not in done:
                write_result(fh, o)
            results.append(o)
            mark = "skip" if o.gold_error else ("ok  " if o.correct else "miss")
            where = f"{o.db_id}" + (f" -> {o.routed_db}" if o.routed_db and o.routed_db != o.db_id else "")
            note = f"  ({o.error[:80]})" if o.error else ""
            print(f"[{i}/{len(examples)}] {mark} {where} {o.seconds:.1f}s{note}", flush=True)
    text = summarize(results, args.route_only, routing)
    print("\n" + text)
    summary = out.with_suffix(".summary.txt")
    summary.write_text(
        f"{args.name} {split.questions_file.name} {mode} ({len(results)} questions)\n"
        f"settings: {json.dumps({'llm': f'{s.llm_provider}:{s.llm_model}', 'router': args.router or s.sql_db_router, 'candidates': k, 'linking': s.sql_schema_linking and not args.no_linking, 'embeddings': not args.no_embeddings, 'values': s.sql_value_lookup and not args.no_values, 'patterns': s.sql_patterns and not args.no_patterns, 'docs': args.docs, 'docs_full': args.docs_full, 'checks': s.sql_answer_checks and not args.no_checks, 'review': s.sql_review or args.review, 'independent': s.sql_independent_plans or args.independent, 'shape': s.sql_answer_shape and not args.no_shape, 'shots': args.shots, 'sql_candidates': args.sql_candidates or s.sql_candidates, 'route_margin': s.sql_route_margin if args.route_margin is None else args.route_margin, 'route_choices': args.route_choices or s.sql_route_choices, 'limit': args.limit, 'seed': args.seed})}\n"
        + text + "\n",
        encoding="utf-8",
    )
    print(f"\nPer-question results: {out}\nSummary: {summary}")
    return 0


def cmd_failures(args) -> int:
    from .failures import main

    text = main(args.run, args.out, args.holdout)
    print(text)
    print(f"Report: {args.out}\nReview sheet: {Path(args.out).with_suffix('.csv')}")
    return 0


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
    p.add_argument("--db-dir", help="override DOCQA_DATABASE_DIR: a folder of SQLite files, one picked per question")
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

    dd = sub.add_parser("describe-db", help="draft a data dictionary from the database, for you to correct")
    dd.add_argument("--tables", help="comma-separated tables (default: all)")
    dd.add_argument("--out", help="output file (default: <docs>/data_dictionary_draft.md)")
    dd.add_argument("--force", action="store_true", help="overwrite an existing draft")
    dd.set_defaults(func=cmd_describe_db)

    gb = sub.add_parser("get-benchmark", help="download Spider or BIRD (dev) from the official source into ./benchmarks")
    gb.add_argument("name", choices=["spider", "bird"])
    gb.add_argument("--dest", default="./benchmarks", help="folder to unpack into (default: ./benchmarks)")
    gb.add_argument("--zip", help="use a zip you downloaded yourself instead of downloading")
    gb.set_defaults(func=cmd_get_benchmark)

    ev = sub.add_parser("eval", help="run a benchmark's dev questions and report execution accuracy")
    ev.add_argument("name", choices=["spider", "bird"])
    ev.add_argument("--data", help="unpacked benchmark folder (default: ./benchmarks/<name>)")
    ev.add_argument("--questions", default="dev.json", help="questions file inside it (default: dev.json)")
    ev.add_argument("--limit", type=int, help="a random sample of N questions (reproducible with --seed)")
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--only-db", help="only questions about this database")
    ev.add_argument("--oracle-db", action="store_true", help="give the agent the right database (leaderboard setting)")
    ev.add_argument("--route-only", action="store_true", help="only test database routing; no chat model needed")
    ev.add_argument("--router", choices=["llm", "retrieval"], help="override DOCQA_DB_ROUTER")
    ev.add_argument("--candidates", type=int, help="databases the router considers (default DOCQA_DB_CANDIDATES=5)")
    ev.add_argument("--route-margin", type=float,
                    help="schema-fit lead that skips the model's database pick (default DOCQA_ROUTE_MARGIN=0.25; 99 = always ask)")
    ev.add_argument("--route-choices", type=int, help="the model picks among this many top databases (default DOCQA_ROUTE_CHOICES=2)")
    ev.add_argument("--no-fit", action="store_true", help="route-only: rank by retrieval alone, without schema/value fit")
    ev.add_argument("--no-embeddings", action="store_true", help="route with keyword search only")
    ev.add_argument("--no-linking", action="store_true", help="skip the schema-linking call (faster, usually less accurate)")
    ev.add_argument("--timeout", type=float, default=30.0, help="seconds per query execution")
    ev.add_argument("--resume", help="continue a results .jsonl from an interrupted run")
    ev.add_argument("--shots", type=int, default=0, help="show N similar solved training questions per question (DAIL-SQL style)")
    ev.add_argument("--train", help="folder with the training questions for --shots (default: next to dev.json)")
    ev.add_argument("--no-values", action="store_true", help="don't look up matching database values")
    ev.add_argument("--docs", help="a documents folder (one sub-folder per database, e.g. a data dictionary); "
                    "each question gets the value notes of its database's folder; reference folders (an SQL "
                    "manual) are shown only when a query fails")
    ev.add_argument("--docs-full", action="store_true", help="with --docs: whole passages plus a manual page per "
                    "technique on every question (the round 5b behaviour)")
    ev.add_argument("--no-checks", action="store_true", help="don't EXPLAIN first or re-ask on suspicious results "
                    "(empty, all NULL, duplicated rows, percentage outside 0-100)")
    ev.add_argument("--no-shape", action="store_true", help="the plan does not state the answer shape "
                    "(row entity, measure, denominator, ties); with --no-checks this isolates the schema fix")
    ev.add_argument("--independent", action="store_true", help="with --sql-candidates N: each extra candidate "
                    "plans the question afresh; the results show how often any candidate was right")
    ev.add_argument("--ids", help="a file of question ids (one per line, or a JSON list): run exactly these")
    ev.add_argument("--exclude-ids", help="leave these question ids out before sampling (e.g. the regression set)")
    ev.add_argument("--save-ids", help="write the ids of the selected questions to this file (freeze a set)")
    ev.add_argument("--review", action="store_true", help="the model reviews its own query and first rows and "
                    "rewrites it once if it says WRONG (one more model call per question)")
    ev.add_argument("--no-patterns", action="store_true", help="don't show worked examples of SQL techniques")
    ev.add_argument("--sql-candidates", type=int, help="write N queries and keep the majority result (default DOCQA_SQL_CANDIDATES=1)")
    ev.set_defaults(func=cmd_eval)

    fr = sub.add_parser("failures", help="label SQL failures across runs (from results + DOCQA_TRACE_FILE traces)")
    fr.add_argument("--run", action="append", required=True,
                    help="name=results.jsonl[,trace.jsonl]; repeat for several runs")
    fr.add_argument("--out", default="failure_report.md")
    fr.add_argument("--holdout", help="a file of question ids to leave out of the diagnosis")
    fr.set_defaults(func=cmd_failures)

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
