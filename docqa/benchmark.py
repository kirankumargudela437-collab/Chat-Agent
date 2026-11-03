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


def pick(examples: list[Example], limit: int | None, seed: int = 0, db: str | None = None) -> list[Example]:
    """A reproducible sample. Dev files are sorted by database, so the first N would test one or two."""
    if db:
        examples = [e for e in examples if e.db_id == db]
    if limit and limit < len(examples):
        examples = random.Random(seed).sample(examples, limit)
    return examples


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
    g, p = [_norm(r) for r in gold], [_norm(r) for r in pred]
    if benchmark == "bird":
        return set(g) == set(p)
    if ordered:
        return g == p
    return Counter(g) == Counter(p)


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


def evaluate(
    agent,
    split: Split,
    examples: Iterable[Example],
    oracle_db: bool = False,
    timeout: float = 30.0,
    done: dict[str, dict] | None = None,
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
        context = f"[1] (source: benchmark evidence)\n{ex.evidence}" if ex.evidence else ""
        try:
            r = agent.run(ex.question, context=context, database=ex.db_id if oracle_db else None)
            out.pred_sql, out.attempts, out.routed_db, out.candidates = r.sql, r.attempts, r.database, r.candidates
            if r.error:
                out.error = r.error
            elif r.database == ex.db_id:
                pred = execute(path, r.sql, timeout)
                out.correct = results_match(gold, pred, split.benchmark, has_top_level_order_by(ex.gold_sql))
        except Exception as e:  # model server down, timeout: record and continue
            out.error = f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}"
        out.seconds = round(time.monotonic() - start, 2)
        yield out


def route_only(catalog, examples: Iterable[Example], k: int = 3) -> Iterator[Outcome]:
    """Routing without a chat model: is the gold database the top hit / in the top ``k``?"""
    for ex in examples:
        start = time.monotonic()
        ranked = [n for n, _ in catalog.rank(ex.question, k)]
        out = Outcome(ex.id, ex.db_id, ex.question, ex.gold_sql, routed_db=ranked[0] if ranked else "",
                      candidates=ranked, difficulty=ex.difficulty)
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

