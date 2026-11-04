"""A reference library of SQL techniques, each with five worked examples, for retrieval.

Small models make the same reasoning slips again and again: a skipped bridge table,
``!=`` where the question means "never", a running total written as GROUP BY, integer
division in a percentage. Each file in ``sql_patterns/`` teaches one technique on one
small demo schema (``demo_schema.sql``): when to use it, the usual mistake, and five
questions with their SQL and a one-line reason. The examples are checked by the tests
against the demo database, so every query runs.

For a question, the best-matching techniques are retrieved (BM25 over each pattern's
cues, description and example questions) and shown to the model as a reference. The
examples use the demo schema on purpose: the model is told to copy the technique, never
the names, and nothing here depends on the user's own databases.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PATTERN_DIR = Path(__file__).with_name("sql_patterns")
DEMO_SCHEMA = PATTERN_DIR / "demo_schema.sql"


@dataclass
class Example:
    question: str
    sql: str
    why: str = ""


@dataclass
class Pattern:
    id: str
    title: str
    cues: list[str]
    use: str
    avoid: str
    manual: str = ""  # words to search SQL reference documents with (manual pages, cheat sheets)
    examples: list[Example] = field(default_factory=list)

    def text(self, max_examples: int = 2, dialect: str = "sqlite") -> str:
        """The pattern as the model sees it: the rule, the usual mistake and a few examples,
        the SQL translated into ``dialect`` (written once, in SQLite, shown in any dialect)."""
        lines = [f"### {self.title}", f"Use: {self.use}", f"Avoid: {self.avoid}"]
        for ex in self.examples[:max_examples]:
            lines.append(f"Q: {ex.question}\n{translate(ex.sql, dialect)}" + (f"\n-- {ex.why}" if ex.why else ""))
        return "\n".join(lines)


def translate(sql: str, dialect: str) -> str:
    """``sql`` (SQLite) in another dialect: LIMIT -> FETCH FIRST / TOP, strftime -> TO_CHAR, ...
    Falls back to the original when sqlglot cannot translate it."""
    if not dialect or dialect == "sqlite":
        return sql
    try:
        import sqlglot

        return ";\n".join(sqlglot.transpile(sql, read="sqlite", write=dialect)) + ";"
    except Exception:
        return sql


def parse_pattern(text: str) -> Pattern:
    title = re.search(r"^# (.+)$", text, re.M).group(1).strip()

    def field_(name):
        m = re.search(rf"^{name}:\s*(.+)$", text, re.M)
        return m.group(1).strip() if m else ""

    examples = []
    for block in re.split(r"^## Example\s*$", text, flags=re.M)[1:]:
        q = re.search(r"^Q:\s*(.+)$", block, re.M)
        sql = re.search(r"```sql\n(.*?)```", block, re.S)
        why = re.search(r"^Why:\s*(.+)$", block, re.M)
        if q and sql:
            examples.append(Example(q.group(1).strip(), sql.group(1).strip(), why.group(1).strip() if why else ""))
    return Pattern(
        id=field_("id") or title.lower(),
        title=title,
        cues=[c.strip().lower() for c in field_("cues").split(",") if c.strip()],
        use=field_("use"),
        avoid=field_("avoid"),
        manual=field_("manual") or title,
        examples=examples,
    )


@lru_cache(maxsize=4)
def load_patterns(directory: str = str(PATTERN_DIR)) -> tuple[Pattern, ...]:
    return tuple(parse_pattern(p.read_text(encoding="utf-8")) for p in sorted(Path(directory).glob("*.md")))


class PatternIndex:
    """BM25 over each pattern's cue words and title, plus a boost for each multi-word cue
    ("running total", "most recent") found verbatim in the question."""

    def __init__(self, patterns: tuple[Pattern, ...] | list[Pattern] | None = None):
        from .multidb import _BM25

        self.patterns = list(patterns if patterns is not None else load_patterns())
        texts = []
        for p in self.patterns:
            # Only the technique's own one-word cues: the example questions are about the demo
            # shop (customers, orders) and would match topics instead of techniques, and a
            # phrase cue split into words ("moving average") would fire on any "average".
            texts.append(" ".join(c for c in p.cues if " " not in c) or p.id)
        self._bm25 = _BM25(texts)
        # "top ... each" = both parts, in that order, anywhere in the question
        # (regex, weight): a two-part cue is more specific than a phrase, so it weighs more
        self._phrases = [
            [(re.compile(r"\b" + r"\b.*\b".join(re.escape(part.strip()) for part in c.split("...")) + r"\b"),
              6.0 if "..." in c else 2.5)
             for c in p.cues if " " in c]
            for p in self.patterns
        ]

    def search(self, question: str, k: int = 2, min_score: float = 1.0) -> list[Pattern]:
        q = question.lower()
        scores = list(self._bm25.scores(question))
        for i, phrases in enumerate(self._phrases):
            scores[i] += sum(w for rx, w in phrases if rx.search(q))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self.patterns[i] for i in ranked[:k] if scores[i] >= min_score]


def format_patterns(patterns: list[Pattern], max_examples: int = 2, dialect: str = "sqlite") -> str:
    if not patterns:
        return ""
    return (
        "SQL techniques that fit this question. The examples use a demo shop schema: copy the "
        "technique, never its table or column names; use only the schema above.\n\n"
        + "\n\n".join(p.text(max_examples, dialect) for p in patterns)
    )
