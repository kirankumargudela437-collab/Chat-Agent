"""One chat agent for business documents, source code and SQL.

For each question:

1. **Condense** a follow-up into a standalone question (as in ``chain.py``).
2. **Plan**: pick the specialists the question needs, most important first:
   ``DOC`` (business documents), ``CODE`` (source code), ``SQL`` (write and run a
   query). Usually one, at most three. A planner call is skipped when only one
   specialist has anything to work with, or when the user forces a route
   (``/sql``, ``/code``, ``/doc`` in chat, ``--route`` on the command line).
3. **Run** each planned specialist once, searching only its own kind of chunk
   (``category`` metadata), with a fallback to all chunks when that finds nothing.
4. **Combine**: with more than one specialist, one more call merges their partial
   answers, keeping citations and SQL.

The plan is fixed before anything runs and every specialist runs at most once,
so a question costs at most five model calls plus the SQL retries. A supervisor
that re-decides after every step can keep picking the same worker forever, and
it is slower on small local models. The SQL specialist already reads business
rules from the documents, so "SQL + DOC" is only needed when the user also wants
a written explanation from the documents.
"""

from __future__ import annotations

import re
from typing import Iterator

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .chain import (
    ANSWER_PROMPT,
    CODE_PROMPT,
    COMBINE_PROMPT,
    CONDENSE_PROMPT,
    NOT_FOUND,
    Answer,
    citation_label,
    clean_condensed,
    history_as_text,
)
from .sql import SQLAgent, SQLResult, format_sql_answer

ROUTES = ("SQL", "CODE", "DOC")
_MENTIONS_TABLES = re.compile(r"create\s+table|\btables?\b|\bcolumns?\b", re.IGNORECASE)

_ROUTE_HELP = {
    "SQL": "SQL: write or run a database query: metrics, KPIs, reports, counts, tables, columns, joins, 'write a query'.",
    "CODE": "CODE: the source code: functions, classes, files, bugs, how something is implemented, where it is called.",
    "DOC": "DOC: business documents: policies, processes, specs, definitions, how-to, 'what does the document say'.",
}

PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You route questions to specialists. Available specialists:\n{routes}\n\n"
            "Reply with the names of the specialists the question needs, comma-separated, most "
            "important first. Usually one is enough. The SQL specialist already reads business "
            "definitions from the documents, so do not add DOC just to define terms for a query.\n"
            "Output only the names.\n\n"
            "Examples:\n"
            "Write a query for the top 5 customers by revenue -> SQL\n"
            "Where is the auth token validated? -> CODE\n"
            "What is our refund policy? -> DOC\n"
            "Does the code apply the retry limit that the runbook requires? -> CODE, DOC",
        ),
        ("human", "{question}"),
    ]
)


def _numbered(docs: list[Document], start: int) -> str:
    return "\n\n".join(f"[{i}] (source: {citation_label(d)})\n{d.page_content}" for i, d in enumerate(docs, start))


def parse_plan(text: str, available: list[str], limit: int = 3) -> list[str]:
    found: list[str] = []
    for token in re.findall(r"[A-Za-z]+", text.upper()):
        name = {"DOCS": "DOC", "DOCUMENT": "DOC", "DOCUMENTS": "DOC"}.get(token, token)
        if name in available and name not in found:
            found.append(name)
    return found[:limit]


class Agent:
    """Stateful chat that routes each question to documents, code or SQL."""

    def __init__(
        self,
        llm: BaseChatModel,
        vectorstore,
        top_k: int = 4,
        sql_agent: SQLAgent | None = None,
        max_history_messages: int = 8,
        schema: str = "",
        examples_k: int = 3,
    ):
        self.llm = llm
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.examples_k = examples_k
        self.sql_agent = sql_agent or SQLAgent(llm)
        if self.sql_agent.embeddings is None:  # reuse the index's embeddings to pick tables in big databases
            self.sql_agent.embeddings = getattr(vectorstore, "embeddings", None)
        self.max_history_messages = max_history_messages
        self.schema = schema  # DDL / table list pasted by the user
        self.history: list[BaseMessage] = []
        self._condense = CONDENSE_PROMPT | llm | StrOutputParser()
        self._plan = PLAN_PROMPT | llm | StrOutputParser()
        self._doc = ANSWER_PROMPT | llm | StrOutputParser()
        self._code = CODE_PROMPT | llm | StrOutputParser()
        self._combine = COMBINE_PROMPT | llm | StrOutputParser()
        self._categories: set[str] | None = None

    # -- index state ------------------------------------------------------------------

    def refresh(self) -> None:
        """Re-read which kinds of chunk the index holds (call after a sync)."""
        self._categories = None

    @property
    def categories(self) -> set[str]:
        if self._categories is None:
            self._categories = {
                c for c in ("doc", "code", "sql", "example")
                if self.vectorstore.get(where={"category": c}, limit=1, include=[])["ids"]
            }
        return self._categories

    def available_routes(self) -> list[str]:
        cats = self.categories
        out = []
        if self.sql_agent.db is not None or self.schema.strip() or cats & {"sql", "example"}:
            out.append("SQL")
        if "code" in cats:
            out.append("CODE")
        if "doc" in cats:
            out.append("DOC")
        return out

    def reset(self) -> None:
        self.history.clear()

    # -- steps ------------------------------------------------------------------------

    def _standalone(self, question: str) -> str:
        history = self.history[-self.max_history_messages :]
        if not history:
            return question
        raw = self._condense.invoke({"chat_history": history_as_text(history), "question": question})
        return clean_condensed(raw, question)

    def plan(self, question: str, route: str | None = None) -> list[str]:
        if route and route.upper() != "AUTO":
            forced = [r.strip().upper() for r in route.split(",") if r.strip()]
            bad = [r for r in forced if r not in ROUTES]
            if bad:
                raise ValueError(f"Unknown route {', '.join(bad)}. Use auto, sql, code or doc.")
            return forced
        available = self.available_routes()
        if len(available) <= 1:
            return available or ["DOC"]
        routes = "\n".join(_ROUTE_HELP[r] for r in available)
        return parse_plan(self._plan.invoke({"routes": routes, "question": question}), available) or ["ALL"]

    def search(self, query: str, category: str | None, k: int | None = None) -> list[Document]:
        k = k or self.top_k
        if category and category in self.categories:
            docs = self.vectorstore.similarity_search(query, k=k, filter={"category": category})
            if docs:
                return docs
        if category in ("sql", "example"):
            return []  # no schema or examples indexed: don't pad the SQL prompt with unrelated chunks
        return self.vectorstore.similarity_search(query, k=k)

    def _sql_context(self, question: str) -> list[Document]:
        docs: list[Document] = []
        seen = set()
        for cat, k in (("example", self.examples_k), ("sql", self.top_k), ("doc", self.top_k)):
            if cat not in self.categories:
                continue
            for d in self.vectorstore.similarity_search(question, k=k, filter={"category": cat}):
                key = (d.metadata.get("source"), d.metadata.get("lines"), d.page_content[:80])
                if key not in seen:
                    seen.add(key)
                    docs.append(d)
        return docs

    # -- public -----------------------------------------------------------------------

    def ask(self, question: str, route: str | None = None) -> Answer:
        final = None
        for part in self.stream(question, route):
            if isinstance(part, Answer):
                final = part
        return final

    def stream(self, question: str, route: str | None = None) -> Iterator[str | Answer]:
        """Yields answer text as it arrives, then the final ``Answer``."""
        standalone = self._standalone(question)
        routes = self.plan(standalone, route)
        docs: list[Document] = []
        sql_result: SQLResult | None = None
        parts: list[str] = []

        if len(routes) == 1:
            gen = self._run_route(routes[0], standalone, docs, stream=True)
            chunks = []
            for item in gen:
                if isinstance(item, SQLResult):
                    sql_result = item
                else:
                    chunks.append(item)
                    yield item
            text = "".join(chunks).strip()
        else:
            for r in routes:
                start = len(docs) + 1
                out = []
                for item in self._run_route(r, standalone, docs, stream=False):
                    if isinstance(item, SQLResult):
                        sql_result = item
                    else:
                        out.append(item)
                piece = "".join(out).strip()
                if piece and not piece.startswith(NOT_FOUND):
                    parts.append(f"## {r} specialist (passages [{start}]...)\n{piece}")
            if not parts:
                text = NOT_FOUND
                yield text
            else:
                chunks = []
                for tok in self._combine.stream({"parts": "\n\n".join(parts), "question": standalone}):
                    chunks.append(tok)
                    yield tok
                text = "".join(chunks).strip()

        self.history += [HumanMessage(question), AIMessage(text)]
        yield Answer(text, question, standalone, docs, tuple(routes), sql_result)

    def _run_route(self, route: str, question: str, docs: list[Document], stream: bool):
        """Run one specialist. Appends its passages to ``docs`` (shared citation numbering)."""
        start = len(docs) + 1
        if route == "SQL":
            context = self._sql_context(question)
            docs.extend(context)
            has_schema = (
                self.sql_agent.db is not None
                or self.schema.strip()
                or any(d.metadata.get("category") in ("sql", "example") for d in context)
                or any(_MENTIONS_TABLES.search(d.page_content) for d in context)  # e.g. a data dictionary PDF
            )
            if not has_schema:
                yield (
                    "I have no schema to write SQL against. Connect a database (DOCQA_DATABASE_URL), "
                    "paste the tables, or add .sql files with CREATE TABLE statements to the docs folder."
                )
                return
            examples = [
                (d.metadata["question"], d.page_content.split("SQL:\n", 1)[-1])
                for d in context
                if d.metadata.get("category") == "example" and d.metadata.get("question")
            ]
            result = self.sql_agent.run(question, _numbered(context, start) if context else "", self.schema, examples)
            yield result
            summary_tokens = []
            if result.ok and result.executed:
                for tok in (self.sql_agent.stream_summary(result) if stream else [self.sql_agent.summarize(result)]):
                    summary_tokens.append(tok)
                    yield tok
            rest = format_sql_answer(result)
            yield ("\n\n" + rest) if summary_tokens else rest
            return

        category = {"CODE": "code", "DOC": "doc"}.get(route)
        found = self.search(question, category, k=self.top_k + (1 if route == "CODE" else 0))
        if not found:
            yield NOT_FOUND
            return
        docs.extend(found)
        chain = self._code if route == "CODE" else self._doc
        inputs = {"context": _numbered(found, start), "question": question}
        if stream:
            yield from chain.stream(inputs)
        else:
            yield chain.invoke(inputs)
