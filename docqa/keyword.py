"""Keyword (BM25) search over the indexed chunks, fused with vector search.

Embeddings find passages that *mean* the same thing; they are weak on exact
identifiers like ``validate_request_token``, ``cust_stat_cd`` or an error string.
Every RAG engine Kiran listed pairs vector search with keyword search for that
reason: RAGFlow and Verba (hybrid search with weighted fusion), Continue (SQLite
FTS with BM25, file paths weighted up), Cody (keyword search with symbol names),
Aider (ranks files whose names match identifiers in the question).

Here:

* identifiers are split as well as kept whole, so ``validateRequestToken``,
  ``validate_request_token`` and "validate request token" all match;
* the file path is indexed with the text and counted three times, so a question
  naming a file or module finds it;
* the two ranked lists are merged with reciprocal rank fusion (RRF), which needs
  no score calibration between BM25 and cosine similarity.

The index is built in memory from the chunks already in Chroma, per category,
the first time it is needed after a sync. That is fine for a personal index of
tens of thousands of chunks.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

from langchain_core.documents import Document

_WORD = re.compile(r"[A-Za-z0-9_]+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was", "be", "it", "this",
    "that", "with", "as", "by", "at", "from", "how", "what", "where", "when", "which", "who", "why", "does",
    "do", "did", "can", "i", "we", "you", "our", "my", "me", "us", "there", "their", "any", "all", "about",
}


def tokenize(text: str) -> list[str]:
    out = []
    for word in _WORD.findall(text):
        low = word.lower()
        if low not in _STOP and len(low) > 1:
            out.append(low)
        parts = [p.lower() for piece in word.split("_") for p in _CAMEL.findall(piece)]
        if len(parts) > 1:
            out += [p for p in parts if p not in _STOP and len(p) > 1]
    return out


def doc_key(d: Document) -> tuple:
    m = d.metadata
    return (m.get("source"), m.get("page"), m.get("section"), m.get("lines"), d.page_content[:200])


class KeywordIndex:
    """Okapi BM25 over chunk text plus its file path."""

    def __init__(self, docs: list[Document], k1: float = 1.5, b: float = 0.75, path_weight: int = 3):
        self.docs = docs
        self.k1, self.b = k1, b
        self.tfs: list[Counter] = []
        df: Counter = Counter()
        for d in docs:
            tokens = tokenize(d.page_content) + tokenize(str(d.metadata.get("source", ""))) * path_weight
            tf = Counter(tokens)
            self.tfs.append(tf)
            df.update(tf.keys())
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, k: int) -> list[Document]:
        terms = set(tokenize(query)) & self.idf.keys()
        if not terms:
            return []
        scored = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for t in terms:
                f = tf.get(t)
                if f:
                    norm = 1 - self.b + self.b * self.lengths[i] / (self.avg_len or 1)
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * norm)
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [self.docs[i] for _, i in scored[:k]]


def reciprocal_rank_fusion(rankings: Iterable[list[Document]], k: int, c: int = 60) -> list[Document]:
    scores: dict[tuple, float] = {}
    first: dict[tuple, Document] = {}
    for ranking in rankings:
        for rank, d in enumerate(ranking):
            key = doc_key(d)
            scores[key] = scores.get(key, 0.0) + 1.0 / (c + rank + 1)
            first.setdefault(key, d)
    best = sorted(scores, key=lambda key: -scores[key])[:k]
    return [first[key] for key in best]
