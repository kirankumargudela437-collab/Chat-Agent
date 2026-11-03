"""Many SQLite databases behind one agent: pick the right database for each question.

Point ``DOCQA_DATABASE_DIR`` (or ``--db-dir``) at a folder of SQLite files, such
as the ``database/`` folder of Spider (166 databases) or ``dev_databases/`` of
BIRD. For every question the agent:

1. **Ranks the databases** by a short schema card per database (its name, tables,
   columns, foreign keys, a few example values and, for BIRD, the column
   descriptions shipped in ``database_description/*.csv``). Two rankings are
   fused with reciprocal rank fusion, as in ``keyword.py``: BM25 over the cards
   (exact names like ``singer`` or ``concert``) and, when an embedding model is
   configured, cosine similarity to the card's embedding (meaning: "vocalist"
   finds ``singer``).
2. **Picks one** of the top ``DOCQA_DB_CANDIDATES`` (default 3). With
   ``DOCQA_DB_ROUTER=llm`` (the default) the chat model reads the candidate
   cards and names the database; with ``retrieval`` the top-ranked one is used
   and no model call is spent.
3. Hands that database to the normal SQL loop (schema linking, generation,
   checks, read-only execution, retries).

Compared with putting every database's full DDL into a vector store and taking
the top hit, the cards are small enough to embed in one piece, the keyword half
catches table names the embedding misses, and the final choice sees several
candidates side by side. Cards and their embeddings are cached in the data
folder and only rebuilt for database files that changed.

Every database is opened read-only (``mode=ro``) exactly like ``DOCQA_DATABASE_URL``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
import re
import sqlite3
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .keyword import tokenize

log = logging.getLogger(__name__)

SQLITE_SUFFIXES = (".sqlite", ".sqlite3", ".db")
CARD_CHARS = 6000  # cap per database card (embedding input and router prompt)


class CatalogError(ValueError):
    """The database folder is missing, empty, or a database name is unknown."""


@dataclass
class DatabaseCard:
    name: str
    path: Path
    text: str
    tables: list[str]
    notes: dict[str, str]  # table -> column descriptions (BIRD), appended to the SQL schema


# --------------------------------------------------------------------------- reading one database


def _q(name: str) -> str:
    """Quote an identifier for SQLite. (``f"PRAGMA table_info('{t}')"`` breaks on a quote in the name.)"""
    return '"' + name.replace('"', '""') + '"'


def connect_read_only(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")  # some BIRD/Spider rows are not UTF-8
    return conn


def _read_csv(path: Path) -> list[dict]:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):  # BIRD's description files mix encodings
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    return [{(k or "").strip().lower(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(io.StringIO(text))]


def read_descriptions(db_folder: Path) -> dict[str, str]:
    """BIRD's ``database_description/<table>.csv`` files as one note per table."""
    folder = db_folder / "database_description"
    notes: dict[str, str] = {}
    if not folder.is_dir():
        return notes
    for f in sorted(folder.glob("*.csv")):
        try:
            rows = _read_csv(f)
        except Exception as e:  # pragma: no cover - a malformed file should not stop the scan
            log.debug("Skipping %s: %s", f, e)
            continue
        lines = []
        for r in rows:
            col = r.get("original_column_name") or r.get("column_name")
            if not col:
                continue
            bits = [b for b in (r.get("column_description"), r.get("value_description")) if b and b.lower() != "nan"]
            full = r.get("column_name")
            if full and full.lower() != col.lower():
                bits.insert(0, full)
            if bits:
                lines.append(f"- {col}: " + "; ".join(" ".join(b.split()) for b in bits))
        if lines:
            notes[f.stem] = "\n".join(lines)
    return notes


def build_card(name: str, path: Path, sample_values: int = 3) -> DatabaseCard:
    """A compact description of one database for routing."""
    conn = connect_read_only(path)
    try:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
        ]
        words = " ".join(re.split(r"[_\W]+", name)).strip()
        lines = [f"Database: {name}" + (f" ({words})" if words != name else ""), "Tables:"]
        for t in tables:
            cols = conn.execute(f"PRAGMA table_info({_q(t)})").fetchall()  # cid, name, type, notnull, dflt, pk
            parts = []
            for c in cols:
                label = c[1] + (" PK" if c[5] else "")
                ctype = (c[2] or "").upper()
                if sample_values and (not ctype or any(k in ctype for k in ("CHAR", "TEXT", "CLOB"))):
                    try:
                        vals = [
                            v[0]
                            for v in conn.execute(
                                f"SELECT DISTINCT {_q(c[1])} FROM {_q(t)} WHERE {_q(c[1])} IS NOT NULL "
                                f"AND length({_q(c[1])}) BETWEEN 1 AND 30 LIMIT {int(sample_values)}"
                            )
                        ]
                    except sqlite3.Error:
                        vals = []
                    vals = [str(v) for v in vals if isinstance(v, str) and v.strip()]
                    if vals:
                        label += " e.g. " + ", ".join(vals)
                parts.append(label)
            line = f"- {t}: " + "; ".join(parts)
            fks = conn.execute(f"PRAGMA foreign_key_list({_q(t)})").fetchall()  # id, seq, table, from, to, ...
            if fks:
                line += " | joins " + ", ".join(f"{fk[3]} -> {fk[2]}.{fk[4] or '?'}" for fk in fks)
            lines.append(line)
    finally:
        conn.close()
    notes = read_descriptions(path.parent)
    text = "\n".join(lines)
    if notes:
        text += "\nColumn descriptions:\n" + "\n".join(f"{t}:\n{n}" for t, n in notes.items())
    return DatabaseCard(name, path, text[:CARD_CHARS], tables, notes)


def find_databases(root: Path) -> dict[str, Path]:
    """``name -> file`` for every SQLite file under ``root``.

    Spider and BIRD keep ``<name>/<name>.sqlite``; the folder name is used. A loose
    file uses its stem. Two files with the same name get their relative path as name.
    """
    root = Path(root)
    if not root.is_dir():
        raise CatalogError(f"Database folder not found: {root}")
    found: dict[str, list[Path]] = {}
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in SQLITE_SUFFIXES and p.is_file() and not any(part.startswith(".") for part in p.relative_to(root).parts):
            name = p.parent.name if p.stem == p.parent.name else p.stem
            found.setdefault(name, []).append(p)
    out = {}
    for name, paths in found.items():
        if len(paths) == 1:
            out[name] = paths[0]
        else:
            for p in paths:
                out[p.relative_to(root).with_suffix("").as_posix()] = p
    if not out:
        raise CatalogError(f"No SQLite files ({', '.join(SQLITE_SUFFIXES)}) under {root}")
    return out


# --------------------------------------------------------------------------- ranking


def _stem(t: str) -> str:
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def _terms(text: str) -> list[str]:
    return [_stem(t) for t in tokenize(text)]


class _BM25:
    def __init__(self, texts: list[str], k1: float = 1.2, b: float = 0.5):
        self.tfs = [Counter(_terms(t)) for t in texts]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avg = sum(self.lengths) / len(self.lengths) if self.lengths else 1.0
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(texts)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> list[float]:
        terms = set(_terms(query)) & self.idf.keys()
        out = []
        for tf, length in zip(self.tfs, self.lengths):
            s = 0.0
            norm = 1 - self.b + self.b * length / (self.avg or 1)
            for t in terms:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * norm)
            out.append(s)
        return out


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Pick the ONE database that can answer the question. Candidates, each with its tables, "
            "columns and example values:\n\n{cards}\n\n"
            "Reply with the database name only, exactly as written after 'Database:'.",
        ),
        ("human", "{question}"),
    ]
)


class DatabaseCatalog:
    """A folder of SQLite databases with a router that picks one per question."""

    def __init__(
        self,
        root: Path,
        embeddings=None,
        cache_dir: Path | None = None,
        cache_key: str = "",
        max_open: int = 8,
        sample_values: int = 3,
    ):
        self.root = Path(root)
        self.paths = find_databases(self.root)
        self.embeddings = embeddings
        self.sample_values = sample_values
        self._open: OrderedDict[str, object] = OrderedDict()
        self.max_open = max_open
        self._cache_file = None
        if cache_dir is not None:
            key = f"{self.root.resolve().as_posix()}|{cache_key}|v1"
            self._cache_file = Path(cache_dir) / f"db_catalog-{hashlib.sha256(key.encode()).hexdigest()[:12]}.json"
        self.cards: dict[str, DatabaseCard] = {}
        self._vectors: dict[str, list[float]] = {}
        self._load()
        self._bm25 = _BM25([self.cards[n].text for n in self.names])

    @property
    def names(self) -> list[str]:
        return list(self.paths)

    def __len__(self) -> int:
        return len(self.paths)

    def __contains__(self, name: str) -> bool:
        return name in self.paths

    # -- cards and their cache ------------------------------------------------------

    def _load(self) -> None:
        cache = {}
        if self._cache_file and self._cache_file.exists():
            try:
                cache = json.loads(self._cache_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                cache = {}
        changed = False
        to_embed = []
        for name, path in self.paths.items():
            st = path.stat()
            sig = f"{st.st_mtime_ns}:{st.st_size}"
            hit = cache.get(name)
            if hit and hit.get("sig") == sig:
                card = DatabaseCard(name, path, hit["text"], hit["tables"], hit.get("notes", {}))
                if hit.get("vector") is not None:
                    self._vectors[name] = hit["vector"]
            else:
                try:
                    card = build_card(name, path, self.sample_values)
                except sqlite3.Error as e:
                    log.warning("Skipping database %s (%s): %s", name, path, e)
                    continue
                cache[name] = {"sig": sig, "text": card.text, "tables": card.tables, "notes": card.notes}
                changed = True
            self.cards[name] = card
            if self.embeddings is not None and name not in self._vectors:
                to_embed.append(name)
        self.paths = {n: p for n, p in self.paths.items() if n in self.cards}
        if to_embed:
            vectors = self.embeddings.embed_documents([self.cards[n].text for n in to_embed])
            for n, v in zip(to_embed, vectors):
                self._vectors[n] = list(v)
                cache[n]["vector"] = list(v)
            changed = True
        for gone in set(cache) - set(self.paths):
            del cache[gone]
            changed = True
        if changed and self._cache_file:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            self._cache_file.write_text(json.dumps(cache), encoding="utf-8")

    # -- routing --------------------------------------------------------------------

    def rank(self, question: str, k: int | None = None) -> list[tuple[str, float]]:
        """Databases, best first, with a fused score (reciprocal rank fusion of BM25 and embeddings)."""
        names = self.names
        rankings = []
        lex = self._bm25.scores(question)
        rankings.append(sorted(range(len(names)), key=lambda i: -lex[i]))
        if self.embeddings is not None and self._vectors:
            q = self.embeddings.embed_query(question)
            sem = [_cosine(q, self._vectors.get(n, ())) for n in names]
            rankings.append(sorted(range(len(names)), key=lambda i: -sem[i]))
        fused = [0.0] * len(names)
        for r in rankings:
            for pos, i in enumerate(r):
                fused[i] += 1.0 / (60 + pos + 1)
        order = sorted(range(len(names)), key=lambda i: (-fused[i], -lex[i]))
        out = [(names[i], round(fused[i], 5)) for i in order]
        return out[:k] if k else out

    def choose(self, question: str, candidates: list[str], llm: BaseChatModel) -> str:
        """Let the model pick among candidate cards; fall back to the first on a bad reply."""
        if len(candidates) <= 1:
            return candidates[0]
        per = max(800, CARD_CHARS // len(candidates))
        cards = "\n\n".join(self.cards[n].text[:per] for n in candidates)
        reply = (ROUTER_PROMPT | llm | StrOutputParser()).invoke({"cards": cards, "question": question})
        return match_name(reply, candidates) or candidates[0]

    # -- opening --------------------------------------------------------------------

    def open(self, name: str, sample_rows: int = 3):
        """The read-only ``Database`` for ``name``; a few stay open, least recently used are closed."""
        from .sql import open_database

        if name not in self.paths:
            raise CatalogError(f"Unknown database {name!r}. Known: {', '.join(sorted(self.paths)[:20])}...")
        if name in self._open:
            self._open.move_to_end(name)
            return self._open[name]
        db = open_database(f"sqlite:///{self.paths[name].resolve().as_posix()}", sample_rows=sample_rows)
        db.notes = dict(self.cards[name].notes)
        self._open[name] = db
        while len(self._open) > self.max_open:
            _, old = self._open.popitem(last=False)
            old._engine.dispose()
        return db


def match_name(reply: str, names: list[str]) -> str | None:
    """The database named in a model reply (exact first, then the longest name it contains)."""
    text = reply.strip().strip("`'\" .").lower()
    lower = {n.lower(): n for n in names}
    if text in lower:
        return lower[text]
    hits = [n for n in names if re.search(rf"(?<![a-z0-9_]){re.escape(n.lower())}(?![a-z0-9_])", reply.lower())]
    return max(hits, key=len) if hits else None
