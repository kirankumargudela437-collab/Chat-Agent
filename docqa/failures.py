"""Label SQL failures from results + traces, across several runs, so patterns show.

``docqa failures --run name=results.jsonl[,trace.jsonl] ... --out report.md`` reads one or
more runs. A results row needs an id, the question, the reference SQL, the predicted SQL
and whether it passed (field names are matched loosely: ``pred_sql`` / ``predicted_sql`` /
``sql``, ``correct`` / ``passed`` / ``match`` ...). A trace is a ``DOCQA_TRACE_FILE`` written
during the run.

Each failure gets one or more suggested labels, to be confirmed by a reviewer:

* ``loop``: a reply repeats itself (repetition score >= 0.35 over a long reply);
* ``cut_off``: a reply stopped at the output-token limit without repeating;
* ``no_sql`` / ``invalid_sql``: nothing usable, or the database rejected it;
* ``missing_schema``: a table the reference SQL uses was not in the prompt;
* ``wrong_join``: different tables, or an inner join where the reference keeps rows (LEFT JOIN);
* ``wrong_aggregation``: different aggregate functions or GROUP BY;
* ``wrong_filter``: different WHERE / HAVING columns or values;
* ``wrong_columns``: a different number of output columns;
* ``unclear``: none of the above; read the trace.

The report has the pass rate per run and difficulty, a label x run count table, the
pass pattern of every question across runs (always wrong / flaky / always right), and a
CSV (``<out>.csv``) with an empty ``reviewed_labels`` column for the manual pass.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

LABELS = ("loop", "cut_off", "no_sql", "invalid_sql", "missing_schema", "wrong_join", "wrong_aggregation",
          "wrong_filter", "wrong_columns", "unclear")

_KEYS = {
    "id": ("id", "question_id", "qid"),
    "question": ("question", "nl", "text"),
    "gold_sql": ("gold_sql", "reference_sql", "expected_sql", "gold", "query"),
    "pred_sql": ("pred_sql", "predicted_sql", "prediction", "generated_sql", "sql"),
    "correct": ("correct", "passed", "pass", "match", "ok", "is_correct"),
    "error": ("error", "exec_error", "execution_error", "db_error"),
    "difficulty": ("difficulty", "level", "hardness"),
}


def _get(row: dict, name: str, default=None):
    for k in _KEYS[name]:
        if k in row and row[k] is not None:
            return row[k]
    return default


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", str(text)).strip().lower()


@dataclass
class Case:
    run: str
    id: str
    question: str
    gold_sql: str
    pred_sql: str
    correct: bool
    error: str
    difficulty: str
    calls: list = field(default_factory=list)
    labels: list = field(default_factory=list)


def read_jsonl(path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_run(name: str, results: str, trace: str | None = None) -> tuple[list[Case], dict]:
    rows = read_jsonl(results) if str(results).endswith(".jsonl") else json.loads(Path(results).read_text("utf-8"))
    if isinstance(rows, dict):  # {"results": [...]} or {id: row}
        rows = rows.get("results") or [dict(v, id=k) if isinstance(v, dict) else {"id": k} for k, v in rows.items()]
    cases = []
    for r in rows:
        ok = _get(r, "correct", False)
        cases.append(Case(name, str(_get(r, "id", "")), str(_get(r, "question", "")), str(_get(r, "gold_sql", "") or ""),
                          str(_get(r, "pred_sql", "") or ""), bool(ok) and str(ok).lower() not in ("false", "0"),
                          str(_get(r, "error", "") or ""), str(_get(r, "difficulty", "") or "")))
    settings: dict = {}
    if trace:
        by_q: dict[str, list] = defaultdict(list)
        for rec in read_jsonl(trace):
            if rec.get("type") == "run":
                settings = settings or rec
            elif rec.get("type") == "call":
                by_q[_norm(rec.get("question", ""))].append(rec)
        for c in cases:
            key = _norm(c.question)
            c.calls = by_q.get(key) or next((v for k, v in by_q.items() if key and (key in k or k in key)), [])
    return cases, settings


def _parse(sql: str):
    import sqlglot

    try:
        return sqlglot.parse_one(sql, read="sqlite") if sql.strip() else None
    except Exception:
        return None


def _shape(tree) -> dict:
    from sqlglot import exp

    if tree is None:
        return {}
    where_cols, values = set(), set()
    for clause in list(tree.find_all(exp.Where)) + list(tree.find_all(exp.Having)):
        where_cols |= {c.name.lower() for c in clause.find_all(exp.Column)}
        values |= {str(lit.this).lower() for lit in clause.find_all(exp.Literal)}
    sel = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    return {
        "tables": {t.name.lower() for t in tree.find_all(exp.Table)} - _cte_names(tree),
        "left": any((j.args.get("side") or "").upper() in ("LEFT", "RIGHT", "FULL") for j in tree.find_all(exp.Join)),
        "aggs": Counter(type(a).__name__ for a in tree.find_all(exp.AggFunc)),
        "group": {c.name.lower() for g in tree.find_all(exp.Group) for c in g.find_all(exp.Column)},
        "where": where_cols,
        "values": values,
        "n_cols": len(sel.expressions) if sel is not None else 0,
    }


def _cte_names(tree) -> set[str]:
    from sqlglot import exp

    return {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}


def label(case: Case) -> list[str]:
    """Suggested failure labels for one wrong answer (several can apply)."""
    if case.correct:
        return []
    out = []
    long_calls = [c for c in case.calls if (c.get("output_chars") or 0) > 1500]
    if any((c.get("repetition") or 0) >= 0.35 for c in long_calls):
        out.append("loop")
    elif any(str(c.get("stop_reason") or "").lower() in ("length", "max_tokens") for c in case.calls):
        out.append("cut_off")
    if not case.pred_sql.strip():
        return out + ["no_sql"]
    pred, gold = _parse(case.pred_sql), _parse(case.gold_sql)
    if case.error or pred is None:
        out.append("invalid_sql")
    g, p = _shape(gold), _shape(pred)
    if g:
        prompt = " ".join(m.get("content", "") for c in case.calls if c.get("step") in ("generate", "plan")
                          for m in c.get("messages", [])[:1]).lower()
        if prompt and any(not re.search(rf"(?<![a-z0-9_]){re.escape(t)}(?![a-z0-9_])", prompt) for t in g["tables"]):
            out.append("missing_schema")
    if g and p:
        if g["tables"] != p["tables"] or (g["left"] and not p["left"]):
            out.append("wrong_join")
        if g["aggs"] != p["aggs"] or g["group"] != p["group"]:
            out.append("wrong_aggregation")
        if g["where"] != p["where"] or g["values"] != p["values"]:
            out.append("wrong_filter")
        if g["n_cols"] != p["n_cols"]:
            out.append("wrong_columns")
    return out or ["unclear"]


def _pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}% ({a}/{b})" if b else "n/a"


def report(runs: dict[str, list[Case]], settings: dict[str, dict], holdout: set[str] | None = None) -> str:
    holdout = holdout or set()
    lines = ["# SQL failure report", ""]
    if holdout:
        lines.append(f"Held-out questions excluded from the diagnosis: {len(holdout)}.\n")
    lines += ["## Runs", "", "| Run | Model | Temp | Seed | Output limit | Agent | Questions | Correct | Avg s/call |",
              "|---|---|---:|---:|---:|---|---:|---:|---:|"]
    for name, cases in runs.items():
        st = settings.get(name, {})
        calls = [c for k in cases for c in k.calls]
        secs = sum(c.get("seconds") or 0 for c in calls) / len(calls) if calls else 0
        lines.append(f"| {name} | {st.get('model', '?')} | {st.get('temperature', '?')} | {st.get('seed', '-')} | "
                     f"{st.get('num_predict') or '-'} | {st.get('agent_version', '?')} | {len(cases)} | "
                     f"{_pct(sum(c.correct for c in cases), len(cases))} | {secs:.1f} |")
    levels = sorted({c.difficulty for cs in runs.values() for c in cs if c.difficulty})
    if levels:
        lines += ["", "## Correct by difficulty", "", "| Run | " + " | ".join(levels) + " |",
                  "|---|" + "---:|" * len(levels)]
        for name, cases in runs.items():
            lines.append(f"| {name} | " + " | ".join(
                _pct(sum(c.correct for c in cases if c.difficulty == lv), sum(c.difficulty == lv for c in cases))
                for lv in levels) + " |")
    lines += ["", "## Failure labels (suggested; several can apply to one failure)", "",
              "| Label | " + " | ".join(runs) + " |", "|---|" + "---:|" * len(runs)]
    counts = {name: Counter(lb for c in cases for lb in c.labels) for name, cases in runs.items()}
    for lb in LABELS:
        if any(counts[n][lb] for n in runs):
            lines.append(f"| {lb} | " + " | ".join(str(counts[n][lb]) for n in runs) + " |")
    ids = list(dict.fromkeys(c.id for cs in runs.values() for c in cs))
    by = {(c.run, c.id): c for cs in runs.values() for c in cs}
    if len(runs) > 1:
        pattern = {}
        for i in ids:
            marks = [by.get((n, i)) for n in runs]
            got = [m.correct for m in marks if m is not None]
            pattern[i] = "always right" if all(got) else "always wrong" if not any(got) else "flaky"
        tally = Counter(pattern.values())
        lines += ["", "## Across runs", "",
                  f"Always right {tally['always right']}, flaky {tally['flaky']}, always wrong {tally['always wrong']} "
                  f"(of {len(ids)} questions)."]
        stable = Counter(lb for i in ids if pattern[i] == "always wrong"
                         for lb in set().union(*[set(by[(n, i)].labels) for n in runs if (n, i) in by]))
        if stable:
            lines.append("Labels on always-wrong questions: " + ", ".join(f"{k} {v}" for k, v in stable.most_common()))
    loops = [c for cs in runs.values() for c in cs if "loop" in c.labels]
    lines += ["", "## Loops", "", f"{len(loops)} failure(s) show a repeating reply."
              + (" " + ", ".join(f"{c.run}/{c.id}" for c in loops[:20]) if loops else
                 " None: the model finishes and writes a wrong query, so weight-level loop fixes do not apply here.")]
    lines += ["", "## Failures", "", "| Question | Difficulty | " + " | ".join(runs) + " |",
              "|---|---|" + "---|" * len(runs)]
    for i in ids:
        cells = []
        for n in runs:
            c = by.get((n, i))
            cells.append("-" if c is None else "ok" if c.correct else ", ".join(c.labels))
        if any(cell not in ("ok", "-") for cell in cells):
            diff = next((by[(n, i)].difficulty for n in runs if (n, i) in by), "")
            lines.append(f"| {i} | {diff} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_review_csv(runs: dict[str, list[Case]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "id", "difficulty", "correct", "suggested_labels", "reviewed_labels", "notes",
                    "error", "max_repetition", "stop_reasons", "question", "pred_sql", "gold_sql"])
        for cases in runs.values():
            for c in cases:
                w.writerow([c.run, c.id, c.difficulty, c.correct, ";".join(c.labels), "", "", c.error[:300],
                            max((k.get("repetition") or 0 for k in c.calls), default=""),
                            ";".join(sorted({str(k.get("stop_reason")) for k in c.calls})),
                            c.question, c.pred_sql, c.gold_sql])


def main(run_specs: list[str], out: str, holdout_ids: str | None = None) -> str:
    from .benchmark import read_ids

    holdout = set(read_ids(holdout_ids)) if holdout_ids else set()
    runs, settings = {}, {}
    for spec in run_specs:
        name, _, files = spec.partition("=")
        results, _, trace = files.partition(",")
        cases, st = load_run(name, results, trace or None)
        cases = [c for c in cases if c.id not in holdout]
        for c in cases:
            c.labels = label(c)
        runs[name], settings[name] = cases, st
    text = report(runs, settings, holdout)
    Path(out).write_text(text, encoding="utf-8")
    write_review_csv(runs, Path(out).with_suffix(".csv"))
    return text
