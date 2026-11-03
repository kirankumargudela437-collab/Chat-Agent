"""Test the SQL agent on public text-to-SQL benchmarks: Spider and BIRD.

``docqa get-benchmark spider|bird`` downloads a benchmark from its official
source into ``./benchmarks`` (git-ignored: the data is not part of this repo).
``docqa eval spider|bird`` asks the agent every question of the dev set and
reports **execution accuracy**: the share of questions whose generated query
returns the same rows as the benchmark's gold query on the same database.

By default the agent must also find the database itself: it routes each question
across *every* database in the benchmark folder (166 for Spider, 11 for BIRD
dev), so the score covers routing and SQL together. ``--oracle-db`` gives it the
right database, which is the setting the published leaderboards use.

How rows are compared:

* **BIRD**: as sets, exactly like BIRD's own ``evaluation.py``.
* **Spider**: as multisets (duplicates count), and in order when the gold query
  has a top-level ``ORDER BY``. This is Spider's classic execution match, not
  the stricter "test-suite" accuracy, which needs extra databases.
* Floats are rounded to 6 places in both, so ``0.30000000000000004`` equals ``0.3``.

Licenses (checked 2026-11-03): Spider 1.0 is CC BY-SA 4.0 (Yale LILY; the
evaluation code in taoyds/spider is Apache-2.0). BIRD is CC BY-SA 4.0 (its
README notes the license was changed to CC BY-SA 4.0). Both allow use and
sharing with attribution; derived datasets must use the same license. Neither
is redistributed here.
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
import time
import urllib.request
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from .multidb import connect_read_only

# --------------------------------------------------------------------------- sources

SOURCES = {
    "spider": {
        "title": "Spider 1.0 (Yale LILY)",
        "page": "https://yale-lily.github.io/spider",
        "license": "CC BY-SA 4.0",
        # the official download on the Spider page is this Google Drive file (spider_data.zip)
        "url": "https://drive.usercontent.google.com/download?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download&confirm=t",
        "size": "about 1 GB",
    },
    "bird": {
        "title": "BIRD dev set (HKU / Alibaba DAMO)",
        "page": "https://bird-bench.github.io/",
        "license": "CC BY-SA 4.0",
        "url": "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip",
        "size": "about 1.3 GB unpacked",
    },
}


class BenchmarkError(ValueError):
    pass


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    root = dest.resolve()
    for m in zf.infolist():
        target = (dest / m.filename).resolve()
        if root != target and root not in target.parents:  # zip-slip: ../ paths
            raise BenchmarkError(f"Refusing to extract {m.filename!r} outside {dest}")
        if m.filename.startswith("__MACOSX/"):
            continue
        zf.extract(m, dest)


def _download(url: str, target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "docqa-benchmark-downloader"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        kind = resp.headers.get("Content-Type", "")
        if "text/html" in kind:
            raise BenchmarkError(
                f"{url} returned a web page instead of a zip (the link may have moved or need a browser)."
            )
        total = int(resp.headers.get("Content-Length") or 0)
        done, last = 0, 0.0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if time.monotonic() - last > 2:
                last = time.monotonic()
                pct = f" ({100 * done / total:.0f}%)" if total else ""
                print(f"  {done / 1e6:,.0f} MB{pct}", flush=True)
    tmp.replace(target)


def get_benchmark(name: str, dest: Path, zip_path: Path | None = None) -> Path:
    """Download (or take ``zip_path``), unpack, unpack nested database zips, return the folder."""
    if name not in SOURCES:
        raise BenchmarkError(f"Unknown benchmark {name!r}; use spider or bird.")
    src = SOURCES[name]
    folder = Path(dest) / name
    folder.mkdir(parents=True, exist_ok=True)
    if zip_path is None:
        zip_path = folder / f"{name}.zip"
        if not zip_path.exists():
            print(f"Downloading {src['title']} ({src['size']}) from {src['url']} ...")
            try:
                _download(src["url"], zip_path)
            except Exception as e:
                raise BenchmarkError(
                    f"Download failed: {e}\nDownload the zip yourself from {src['page']} and run:\n"
                    f"  docqa get-benchmark {name} --zip <downloaded.zip>"
                ) from e
    print(f"Unpacking {zip_path} ...")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            _safe_extract(zf, folder)
    except zipfile.BadZipFile as e:
        raise BenchmarkError(f"{zip_path} is not a zip file. Download it again from {src['page']}.") from e
    for inner in list(folder.rglob("*_databases.zip")):  # BIRD ships dev_databases.zip inside dev.zip
        if not (inner.parent / inner.stem).exists():
            print(f"Unpacking {inner.name} ...")
            with zipfile.ZipFile(inner) as zf:
                _safe_extract(zf, inner.parent)
    split = locate(name, folder)
    print(f"Ready: {split.questions_file} ({len(split.load())} questions), databases in {split.db_root}")
    print(f"License: {src['license']}. Keep the data out of git; see {src['page']}.")
    return folder


# --------------------------------------------------------------------------- data


@dataclass
class Example:
    id: str
    db_id: str
    question: str
    gold_sql: str
    evidence: str = ""
    difficulty: str = ""


@dataclass
class Split:
    benchmark: str
    questions_file: Path
    db_root: Path

    def load(self) -> list[Example]:
        rows = json.loads(self.questions_file.read_text(encoding="utf-8"))
        out = []
        for i, r in enumerate(rows):
            out.append(
                Example(
                    id=str(r.get("question_id", i)),
                    db_id=r["db_id"],
                    question=r["question"],
                    gold_sql=r.get("SQL") or r.get("query") or "",
                    evidence=r.get("evidence", "") or "",
                    difficulty=r.get("difficulty", "") or "",
                )
            )
        return out

    def db_path(self, db_id: str) -> Path:
        for suffix in (".sqlite", ".db", ".sqlite3"):
            p = self.db_root / db_id / f"{db_id}{suffix}"
            if p.exists():
                return p
        raise BenchmarkError(f"Database {db_id} not found under {self.db_root}")


def locate(benchmark: str, data_dir: Path, questions: str = "dev.json") -> Split:
    """Find the questions file and the database folder inside an unpacked benchmark."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise BenchmarkError(f"{data_dir} does not exist. Run: docqa get-benchmark {benchmark}")
    db_dirs = ("dev_databases", "train_databases", "database") if benchmark == "bird" else ("database",)
    candidates = [data_dir / questions] if (data_dir / questions).exists() else sorted(data_dir.rglob(questions))
    for q in candidates:
        if "__MACOSX" in q.parts:
            continue
        for d in db_dirs:
            # BIRD's dev_databases.zip can unpack to dev_databases/dev_databases/
            for root in (q.parent / d / d, q.parent / d):
                if root.is_dir() and any(root.glob("*/*.sqlite")):
                    return Split(benchmark, q, root)
    raise BenchmarkError(
        f"No {questions} next to a database folder ({' or '.join(db_dirs)}) under {data_dir}. "
        f"Run: docqa get-benchmark {benchmark}"
    )


def read_ids(path) -> list[str]:
    """Question ids from a file: a JSON list, or one id per line (# comments allowed)."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return [str(i) for i in json.loads(text)]
    return [ln.split("#", 1)[0].strip() for ln in text.splitlines() if ln.split("#", 1)[0].strip()]


def pick(examples: list[Example], limit: int | None, seed: int = 0, db: str | None = None) -> list[Example]:
    """A reproducible sample. Dev files are sorted by database, so the first N would test one or two."""
    if db:
        examples = [e for e in examples if e.db_id == db]
    if limit and limit < len(examples):
        examples = random.Random(seed).sample(examples, limit)
    return examples


# --------------------------------------------------------------------------- few-shot examples


TRAIN_FILES = {
    "spider": ("train_spider.json", "train_others.json"),
    "bird": ("train.json",),
}


class ExampleBank:
    """Solved training questions, retrieved by similarity to a new question (DAIL-SQL's example selection).

    Small models gain most from seeing a few solved questions shaped like the new
    one. DAIL-SQL (Gao et al., 2023) picks them by question similarity after
    masking domain words, so "How many singers are older than 30?" retrieves
    "How many pilots are younger than 25?". Here: BM25 over the question with
    numbers, quoted values and words that are names in the target schema removed.
    Examples from the target database itself are never used (in Spider and BIRD
    the training databases are disjoint from dev anyway).
    """

    def __init__(self, examples: list[Example]):
        from .multidb import _BM25

        self.examples = [e for e in examples if e.gold_sql]
        self._bm25 = _BM25([_skeleton(e.question) for e in self.examples])

    @classmethod
    def from_benchmark(cls, benchmark: str, data_dir: Path) -> "ExampleBank | None":
        files = []
        for name in TRAIN_FILES.get(benchmark, ()):
            files += [p for p in sorted(Path(data_dir).rglob(name)) if "__MACOSX" not in p.parts]
        if not files:
            return None
        rows = []
        for f in files:
            for i, r in enumerate(json.loads(f.read_text(encoding="utf-8"))):
                rows.append(Example(id=f"{f.stem}-{i}", db_id=r["db_id"], question=r["question"],
                                    gold_sql=r.get("SQL") or r.get("query") or "", evidence=r.get("evidence", "") or ""))
        return cls(rows)

    def __len__(self) -> int:
        return len(self.examples)

    def similar(self, question: str, k: int = 4, exclude_db: str | None = None, schema_words: set[str] = frozenset()) -> list[tuple[str, str]]:
        scores = self._bm25.scores(_skeleton(question, schema_words))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        out, seen = [], set()
        for i in order:
            e = self.examples[i]
            if scores[i] <= 0 or len(out) >= k:
                break
            if e.db_id == exclude_db or e.gold_sql in seen:
                continue
            seen.add(e.gold_sql)
            q = f"{e.question}\n(Hint: {e.evidence})" if e.evidence else e.question
            out.append((q, e.gold_sql.strip().rstrip(";")))
        return out[::-1]  # most similar last, closest to the new question


def _skeleton(question: str, schema_words: set[str] = frozenset()) -> str:
    """The question with values masked: numbers, quoted strings and schema names removed."""
    q = re.sub(r"'[^']*'|\"[^\"]*\"", " value ", question)
    q = re.sub(r"\d+(?:\.\d+)?", " number ", q)
    if schema_words:
        q = " ".join(w for w in q.split() if w.lower().strip("?,.") not in schema_words)
    return q


# --------------------------------------------------------------------------- execution


class QueryTimeout(RuntimeError):
    pass


def execute(path: Path, sql: str, timeout: float = 30.0, max_rows: int = 100_000) -> list[tuple]:
    """Run one query read-only, stopping it after ``timeout`` seconds."""
    conn = connect_read_only(path)
    deadline = time.monotonic() + timeout
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 20_000)
    try:
        return conn.execute(sql).fetchmany(max_rows)
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e):
            raise QueryTimeout(f"query ran longer than {timeout:g}s") from e
        raise
    finally:
        conn.close()


def _norm(row: tuple) -> tuple:
    return tuple(round(v, 6) if isinstance(v, float) else v for v in row)


_TOP_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)


def has_top_level_order_by(sql: str) -> bool:
    try:
        import sqlglot

        tree = sqlglot.parse_one(sql, read="sqlite")
        return tree.args.get("order") is not None
    except Exception:  # unparsable gold: fall back to text
        depth, text = 0, ""
        for ch in sql:
            depth += ch == "("
            depth -= ch == ")"
            text += ch if depth == 0 else " "
        return bool(_TOP_ORDER_BY.search(text))


def results_match(gold: list[tuple], pred: list[tuple], benchmark: str, ordered: bool = False) -> bool:
    """BIRD: row sets, as BIRD's evaluation.py. Spider: rows, with columns in any order.

    Spider's official execution match (``exec_eval.py`` in taoyds/test-suite-sql-eval)
    accepts ``SELECT name, year`` for a gold ``SELECT year, name``: it tries the
    column permutations whose per-column values agree. The first version of this
    function compared columns in order and under-counted.
    """
    g, p = [_norm(r) for r in gold], [_norm(r) for r in pred]
    if benchmark == "bird":
        return set(g) == set(p)

    def same(rows_p):
        return rows_p == g if ordered else Counter(rows_p) == Counter(g)

    if same(p):
        return True
    if not g or not p or len(g) != len(p) or len(g[0]) != len(p[0]) or len(g[0]) > 8:
        return False
    for perm in column_permutations(g, p):
        if same([tuple(r[j] for j in perm) for r in p]):
            return True
    return False


def column_permutations(gold: list[tuple], pred: list[tuple]):
    """Orders of the predicted columns whose values, column by column, match the gold columns'."""
    n = len(gold[0])
    gcols = [Counter(r[i] for r in gold) for i in range(n)]
    pcols = [Counter(r[j] for r in pred) for j in range(n)]
    options = [[j for j in range(n) if pcols[j] == gcols[i]] for i in range(n)]

    def walk(i, used, acc):
        if i == n:
            yield tuple(acc)
            return
        for j in options[i]:
            if j not in used:
                yield from walk(i + 1, used | {j}, acc + [j])

    yield from walk(0, frozenset(), [])


# --------------------------------------------------------------------------- evaluation


@dataclass
class Outcome:
    id: str
    db_id: str
    question: str
    gold_sql: str
    pred_sql: str = ""
    routed_db: str = ""
    candidates: list[str] = field(default_factory=list)
    correct: bool = False
    attempts: int = 0
    error: str = ""
    gold_error: str = ""
    difficulty: str = ""
    seconds: float = 0.0
    margin: float | None = None  # route-only: schema-fit lead of #1 over #2
    alt_correct: list[bool] = field(default_factory=list)  # each SQL candidate that ran: right or not
    schema: dict = field(default_factory=dict)  # table -> what reached the prompt (keys/columns/ddl+notes+sample)


def evaluate(
    agent,
    split: Split,
    examples: Iterable[Example],
    oracle_db: bool = False,
    timeout: float = 30.0,
    done: dict[str, dict] | None = None,
    bank: "ExampleBank | None" = None,
    shots: int = 4,
    docs: "DocsContext | None" = None,
) -> Iterator[Outcome]:
    """Ask ``agent`` (a ``SQLAgent`` with a catalog) every question; yield one ``Outcome`` each."""
    done = done or {}
    for ex in examples:
        if ex.id in done:
            yield Outcome(**done[ex.id])
            continue
        out = Outcome(ex.id, ex.db_id, ex.question, ex.gold_sql, difficulty=ex.difficulty)
        path = split.db_path(ex.db_id)
        try:
            gold = execute(path, ex.gold_sql, timeout)
        except Exception as e:
            out.gold_error = str(e).splitlines()[0][:200]
            yield out
            continue
        start = time.monotonic()
        # BIRD's evidence goes with the question, as in BIRD's own baselines: buried in the
        # context the 7B model ignored it ("youngest = MAX(Birthday)" -> it wrote MIN)
        question = with_evidence(ex.question, ex.evidence)
        try:
            shown = bank.similar(ex.question, shots, exclude_db=ex.db_id) if bank and shots else None
            r = agent.run(question, context="", database=ex.db_id if oracle_db else None, examples=shown,
                          context_fn=docs.for_database if docs else None)
            out.pred_sql, out.attempts, out.routed_db, out.candidates = r.sql, r.attempts, r.database, r.candidates
            out.schema = dict(getattr(agent, "last_schema", {}) or {})
            if len(r.alternatives) > 1 and r.database == ex.db_id:
                order = has_top_level_order_by(ex.gold_sql)
                for alt in r.alternatives:
                    try:
                        out.alt_correct.append(results_match(gold, execute(path, alt, timeout), split.benchmark, order))
                    except Exception:
                        out.alt_correct.append(False)
            if r.error:
                out.error = r.error
            elif r.database == ex.db_id:
                pred = execute(path, r.sql, timeout)
                out.correct = results_match(gold, pred, split.benchmark, has_top_level_order_by(ex.gold_sql))
        except Exception as e:  # model server down, timeout: record and continue
            out.error = f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}"
        out.seconds = round(time.monotonic() - start, 2)
        yield out


class DocsContext:
    """Passages from a documents folder for each benchmark question, as the app would give.

    The folder holds one sub-folder per database (``<docs>/<db_id>/...``: a data dictionary,
    business rules, in any format the app reads) and, optionally, general SQL reference
    material in other folders. It is indexed once with the app's own sync (incremental, so
    a re-run re-embeds nothing). For a question, once the agent has picked its database,
    the passages come from that database's folder only (hybrid vector + keyword search).

    ``slim`` (the default) gives the model only the lines of the top 2 passages that
    explain values (codes, units, meanings), and SQL reference pages only when a query
    fails on a function, date or syntax error (``help_for_error``). Measured on BIRD 30:
    four to six whole passages per question made a 7B model add columns and joins it saw
    in them (66.7% -> 53.3%). ``slim=False`` restores whole passages plus one reference
    page per SQL technique.
    """

    def __init__(self, settings, docs_dir: Path, top_k: int = 4, patterns=None, embeddings=None,
                 slim: bool = True):
        import dataclasses

        from langchain_core.documents import Document

        from .indexing import open_vectorstore, sync_documents
        from .keyword import KeywordIndex

        s = dataclasses.replace(settings, docs_dir=Path(docs_dir), data_dir=Path(settings.data_dir) / "eval-docs")
        self.store = open_vectorstore(s, embeddings)
        self.sync = sync_documents(s, vectorstore=self.store)
        got = self.store.get(include=["documents", "metadatas"])
        by_folder: dict[str, list] = {}
        for text, meta in zip(got["documents"], got["metadatas"]):
            meta = meta or {}
            folder = str(meta.get("source", "")).replace("\\", "/").split("/", 1)[0]
            by_folder.setdefault(folder, []).append(Document(page_content=text, metadata=meta))
        self.docs_by_folder = by_folder
        self._keyword = {f: KeywordIndex(d) for f, d in by_folder.items()}
        self.top_k = top_k
        self.patterns = patterns
        self.reference_folders: list[str] = []  # set by the caller: folders that are not databases
        self.slim = slim

    def _search(self, query: str, folders: list[str], k: int) -> list:
        from .keyword import reciprocal_rank_fusion

        sources = sorted({d.metadata.get("source") for f in folders for d in self.docs_by_folder.get(f, [])})
        if not sources:
            return []
        vector = self.store.similarity_search(query, k=2 * k, filter={"source": {"$in": sources}})
        keyword = [d for f in folders for d in self._keyword[f].search(query, 2 * k)]
        return reciprocal_rank_fusion([vector, keyword], k)

    def for_database(self, database: str, question: str) -> str:
        from .chain import citation_label

        if self.slim:
            from .sqlchecks import value_notes

            notes = value_notes([d.page_content for d in self._search(question, [database], 2)])
            return ("Notes from the documents on what values mean (they explain values; they do not "
                    "ask for extra columns or joins):\n" + notes) if notes else ""
        found = self._search(question, [database], self.top_k)
        if self.patterns is not None and self.reference_folders:
            for p in self.patterns.search(question):
                found += self._search(p.manual, self.reference_folders, 1)
        if not found:
            return ""
        return "Passages from the documents:\n\n" + "\n\n".join(
            f"[{i}] (source: {citation_label(d)})\n{d.page_content}" for i, d in enumerate(found, 1)
        )


    def help_for_error(self, error: str, sql: str, max_chars: int = 900) -> str:
        """One SQL reference page for a function / date / syntax error, for the fix prompt."""
        from .sqlchecks import manual_query

        query = manual_query(error, sql)
        if not query or not self.reference_folders:
            return ""
        found = self._search(query, self.reference_folders, 1)
        if not found:
            return ""
        return "From the SQL manual:\n" + found[0].page_content[:max_chars]


def with_evidence(question: str, evidence: str) -> str:
    """The question plus BIRD's evidence as a hint the model must follow."""
    if not evidence:
        return question
    return f"{question}\nHint (definitions to follow exactly): {evidence}"


def route_only(catalog, examples: Iterable[Example], k: int = 5, fit: bool = True) -> Iterator[Outcome]:
    """Routing without a chat model: is the gold database the top hit / in the top ``k``?

    With ``fit`` the candidates are reordered by schema and value coverage, as the
    agent does before the model picks (``--no-fit`` shows retrieval alone).
    """
    cache: dict = {}

    def values_for(name, db):
        from .sqlhints import ValueIndex

        if name not in cache:
            try:
                cache[name] = ValueIndex(db, time_budget=10.0)
            except Exception:
                cache[name] = None
        return cache[name]

    for ex in examples:
        start = time.monotonic()
        q = with_evidence(ex.question, ex.evidence)
        ranked = [n for n, _ in catalog.rank(q, k)]
        if fit and len(ranked) > 1:
            from .sql import rerank_by_fit

            scores: dict = {}
            ranked, _ = rerank_by_fit(catalog, q, ranked, values_for, scores)
        out = Outcome(ex.id, ex.db_id, ex.question, ex.gold_sql, routed_db=ranked[0] if ranked else "",
                      candidates=ranked, difficulty=ex.difficulty)
        if fit and len(ranked) > 1:
            out.margin = round(scores[ranked[0]] - scores[ranked[1]], 4)
        out.correct = out.routed_db == ex.db_id
        out.seconds = round(time.monotonic() - start, 3)
        yield out


def _pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}% ({a}/{b})" if b else "n/a"


def summarize(outcomes: list[Outcome], route_only_mode: bool = False, routed: bool = True) -> str:
    scored = [o for o in outcomes if not o.gold_error]
    n = len(scored)
    lines = []
    in_top = sum(o.db_id in o.candidates for o in scored)
    right_db = [o for o in scored if o.routed_db == o.db_id]
    if route_only_mode:
        k = max((len(o.candidates) for o in scored), default=0)
        lines.append(f"Routing top-1 accuracy: {_pct(len(right_db), n)}")
        lines.append(f"Gold database in top {k}: {_pct(in_top, n)}")
        if k > 2:
            lines.append(f"Gold database in top 2: {_pct(sum(o.db_id in o.candidates[:2] for o in scored), n)}")
        with_margin = [o for o in scored if o.margin is not None]
        for m in (0.1, 0.15, 0.25) if with_margin else ():
            clear = [o for o in with_margin if o.margin >= m]
            lines.append(
                f"Fit lead >= {m}: {_pct(len(clear), len(with_margin))} of questions, "
                f"#1 right in {_pct(sum(o.routed_db == o.db_id for o in clear), len(clear))}"
            )
    else:
        lines.append(f"Execution accuracy: {_pct(sum(o.correct for o in scored), n)}")
        if routed:
            lines.append(f"Routing: right database {_pct(len(right_db), n)}; gold database among candidates {_pct(in_top, n)}")
            lines.append(f"Execution accuracy when routed right: {_pct(sum(o.correct for o in right_db), len(right_db))}")
        errors = sum(bool(o.error) for o in scored)
        att = [o.attempts for o in scored if o.attempts]
        secs = [o.seconds for o in scored if o.seconds]
        lines.append(
            f"No working query after retries: {errors}; average attempts {sum(att) / len(att) if att else 0:.2f}; "
            f"average {sum(secs) / len(secs) if secs else 0:.1f}s per question"
        )
    levels = sorted({o.difficulty for o in scored if o.difficulty})
    if levels and not route_only_mode:
        lines.append("By difficulty: " + ", ".join(
            f"{lv} {_pct(sum(o.correct for o in scored if o.difficulty == lv), sum(o.difficulty == lv for o in scored))}"
            for lv in levels))
    multi = [o for o in scored if len(o.alt_correct) > 1]
    if multi and not route_only_mode:
        some = [o for o in multi if any(o.alt_correct)]
        lines.append(f"Several candidates: some candidate right {_pct(len(some), len(multi))}; "
                     f"the agent picked a right one {_pct(sum(o.correct for o in some), len(some))} of those times")
    shaped = [o for o in scored if o.schema]
    if shaped and not route_only_mode:
        short = sum(any(not v.startswith("ddl") for v in o.schema.values()) for o in shaped)
        lines.append(f"Schema: some tables shown as one line in {short} of {len(shaped)} questions; "
                     "per-question table detail is in the results file (\"schema\")")
    skipped = len(outcomes) - n
    if skipped:
        lines.append(f"Skipped {skipped} question(s) whose gold query failed or timed out.")
    by_db = Counter(o.db_id for o in scored if not o.correct)
    if by_db and not route_only_mode:
        lines.append("Most misses: " + ", ".join(f"{d} ({c})" for d, c in by_db.most_common(5)))
    return "\n".join(lines)


def read_results(path: Path) -> dict[str, dict]:
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["id"]] = row
    return done


def write_result(fh, o: Outcome) -> None:
    fh.write(json.dumps(asdict(o), ensure_ascii=False) + "\n")
    fh.flush()

