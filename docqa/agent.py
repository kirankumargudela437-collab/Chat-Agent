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
   answers, keeping citations. SQL never goes through that call: the query the SQL
   specialist wrote is shown unchanged after any cited explanation.

The plan is fixed before anything runs and every specialist runs at most once,
so a question costs at most six model calls plus the SQL retries. A supervisor
that re-decides after every step can keep picking the same worker forever, and
it is slower on small local models. The SQL specialist already reads business
rules from the documents, so "SQL + DOC" is only needed when the user also wants
a written explanation from the documents.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import replace
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
    is_not_found,
)
from .keyword import KeywordIndex, doc_key, reciprocal_rank_fusion
from .sql import SQLAgent, SQLResult, format_sql_answer

ROUTES = ("SQL", "CODE", "DOC")
_MENTIONS_TABLES = re.compile(r"create\s+table|\btables?\b|\bcolumns?\b", re.IGNORECASE)
_ASKS_FOR_QUERY = re.compile(r"\b(?:sql|quer(?:y|ies))\b", re.IGNORECASE)
_USES_RULES = re.compile(r"\b(?:cite|citations?|polic(?:y|ies)|rules?|terms|handbook|sources?)\b", re.IGNORECASE)
_SQL_TEXT = re.compile(r"\bselect\b.*\bfrom\b", re.IGNORECASE | re.DOTALL)
_FENCE = re.compile(r"```.*?(?:```|$)", re.DOTALL)
_CITED_SENTENCE = re.compile(r"([^\[\]\n]+?)\s*\[(\d+)\]")

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


RULE_CHECK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You check whether written business rules are complete enough to write a SQL query. "
            "You never write SQL.\n"
            "A request can depend on a business term, threshold, rate, formula or status. If the "
            "passages or the schema define every one the request needs, reply exactly: READY\n"
            "If any is missing, reply: ASK: followed by one question that asks the user for it.\n\n"
            "Examples:\n"
            "Passages: Members with 10 or more visits earn a free visit.\n"
            "Request: Write SQL for members earning a free visit.\n"
            "Reply: READY\n\n"
            "Passages: The librarian reviews late returns.\n"
            "Request: Write SQL for the late-return charge per member.\n"
            "Reply: ASK: How is the late-return charge calculated, and what counts as late?",
        ),
        ("human", "Schema:\n{schema}\n\nPassages:\n{context}\n\nRequest: {question}\nReply:"),
    ]
)

RULE_NOTES_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You copy rule sentences from numbered passages. You never write SQL.\n"
            "For the request below, copy each sentence from the passages that defines a rule, threshold, "
            "amount, code or date the request depends on. Put one sentence per line and end each line "
            "with its passage number in brackets.\n\n"
            "Example\n"
            "Passages:\n"
            "[1] Members with 10 or more visits earn a free visit.\n"
            "[2] Kind L means a long visit. The gym opens at six.\n"
            "Request: Write SQL listing members earning a free visit from long visits.\n"
            "Reply:\n"
            "Members with 10 or more visits earn a free visit. [1]\n"
            "Kind L means a long visit. [2]\n\n"
            "Passages:\n{context}",
        ),
        ("human", "Request: {question}\nReply:"),
    ]
)


def _numbered(docs: list[Document], start: int) -> str:
    return "\n\n".join(f"[{i}] (source: {citation_label(d)})\n{d.page_content}" for i, d in enumerate(docs, start))


def _supported(sentence: str, passage: str, share: float = 0.7) -> bool:
    words = re.findall(r"\w+", sentence.lower())
    found = set(re.findall(r"\w+", passage.lower()))
    return bool(words) and sum(w in found for w in words) >= share * len(words)


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
        hybrid: bool = True,
    ):
        self.llm = llm
        self.hybrid = hybrid  # add keyword (BM25) search to vector search
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.examples_k = examples_k
        self.sql_agent = sql_agent or SQLAgent(llm)
        if self.sql_agent.embeddings is None:  # reuse the index's embeddings to pick tables in big databases
            self.sql_agent.embeddings = getattr(vectorstore, "embeddings", None)
        if self.sql_agent.error_help is None:
            self.sql_agent.error_help = self._manual_help
        self.max_history_messages = max_history_messages
        self.schema = schema  # DDL / table list pasted by the user
        self.history: list[BaseMessage] = []
        self._condense = CONDENSE_PROMPT | llm | StrOutputParser()
        self._plan = PLAN_PROMPT | llm | StrOutputParser()
        self._doc = ANSWER_PROMPT | llm | StrOutputParser()
        self._code = CODE_PROMPT | llm | StrOutputParser()
        self._combine = COMBINE_PROMPT | llm | StrOutputParser()
        self._rule_check = RULE_CHECK_PROMPT | llm | StrOutputParser()
        self._rule_notes = RULE_NOTES_PROMPT | llm | StrOutputParser()
        self._categories: set[str] | None = None
        self._keyword: dict[str | None, KeywordIndex] = {}

    # -- index state ------------------------------------------------------------------

    def refresh(self) -> None:
        """Re-read which kinds of chunk the index holds (call after a sync)."""
        self._categories = None
        self._keyword.clear()

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
        if self.sql_agent.db is not None or self.sql_agent.catalog is not None or self.schema.strip() or cats & {"sql", "example"}:
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
        planned = parse_plan(self._plan.invoke({"routes": routes, "question": question}), available) or ["ALL"]
        if {"SQL", "DOC"} <= set(planned) and not self._connected() and not _ASKS_FOR_QUERY.search(question):
            planned.remove("SQL")  # with no database a query can't answer this; only the documents can
        return planned

    def _connected(self) -> bool:
        return self.sql_agent.db is not None or self.sql_agent.catalog is not None

    def _keyword_index(self, category: str | None) -> KeywordIndex:
        if category not in self._keyword:
            got = self.vectorstore.get(
                where={"category": category} if category else None, include=["documents", "metadatas"]
            )
            docs = [Document(page_content=t, metadata=m or {}) for t, m in zip(got["documents"], got["metadatas"])]
            self._keyword[category] = KeywordIndex(docs)
        return self._keyword[category]

    def _find(self, query: str, category: str | None, k: int) -> list[Document]:
        """Vector search, plus keyword search fused in when ``hybrid`` is on."""
        flt = {"filter": {"category": category}} if category else {}
        if not self.hybrid:
            return self.vectorstore.similarity_search(query, k=k, **flt)
        vector = self.vectorstore.similarity_search(query, k=2 * k, **flt)
        keyword = self._keyword_index(category).search(query, 2 * k)
        return reciprocal_rank_fusion([vector, keyword], k)

    def search(self, query: str, category: str | None, k: int | None = None) -> list[Document]:
        k = k or self.top_k
        if category and category in self.categories:
            docs = self._find(query, category, k)
            if docs:
                return docs
        if category in ("sql", "example"):
            return []  # no schema or examples indexed: don't pad the SQL prompt with unrelated chunks
        return self._find(query, None, k)

    def _sql_context(self, question: str, scoped: bool = False) -> list[Document]:
        """Examples, schema files and business documents for the SQL route. With ``scoped``
        (several databases with their own document folders) the business documents are left
        out here and fetched for the chosen database after routing (``_database_docs``)."""
        docs: list[Document] = []
        seen = set()
        for cat, k in (("example", self.examples_k), ("sql", self.top_k), ("doc", self.top_k)):
            if cat == "doc" and scoped:
                continue
            if cat not in self.categories:
                continue
            for d in self._find(question, cat, k):
                key = doc_key(d)
                if key not in seen:
                    seen.add(key)
                    docs.append(d)
        return docs

    def _has_database_folders(self) -> bool:
        """Do the documents have a folder per connected database (``<docs>/<database>/...``)?"""
        catalog = self.sql_agent.catalog
        if catalog is None or "doc" not in self.categories:
            return False
        if getattr(self, "_db_folders", None) is None:
            got = self.vectorstore.get(where={"category": "doc"}, include=["metadatas"])
            folders = {str((m or {}).get("source", "")).replace("\\", "/").split("/", 1)[0].lower()
                       for m in got.get("metadatas") or []}
            self._db_folders = folders & {str(n).lower() for n in catalog.paths}
        return bool(self._db_folders)

    def _database_docs(self, database: str, question: str, k: int = 2) -> str:
        """Passages from the chosen database's own folder, as in ``docqa eval --docs``: picked
        after routing, so another database's dictionary never reaches the prompt."""
        if not database or database.lower() not in (self._db_folders or set()):
            return ""
        prefix = database.lower() + "/"
        pool = self._find(question, "doc", 8 * k)
        found = [d for d in pool if str(d.metadata.get("source", "")).replace("\\", "/").lower().startswith(prefix)][:k]
        if not found:
            return ""
        return f"Passages from the documents about {database}:\n\n" + "\n\n".join(
            f"(source: {citation_label(d)})\n{d.page_content}" for d in found)

    def _manual_help(self, error: str, sql: str, max_chars: int = 900) -> str:
        """A page from an SQL manual in the documents, shown only when a query failed on a
        function, date or syntax error (on every question it distracted a small model)."""
        from .sqlchecks import manual_query

        query = manual_query(error, sql)
        if not query or "doc" not in self.categories:
            return ""
        found = self._find(query, "doc", 1)
        return f"From the documents ({citation_label(found[0])}):\n{found[0].page_content[:max_chars]}" if found else ""

    def _missing_rule(self, question: str, passages: str) -> str:
        """The question to ask when the documents don't define what a query needs, else empty.

        A small model can ask for a rule that is right there, so its question is put to the
        documents first; it only reaches the user if they can't answer it.
        """
        reply = self._rule_check.invoke({"schema": self.schema.strip(), "context": passages, "question": question})
        m = re.match(r"\s*ASK\s*:\s*(.+)", reply, re.IGNORECASE)
        if not m:
            return ""
        ask = m.group(1).strip().splitlines()[0].strip()
        return ask if is_not_found(self._doc.invoke({"context": passages, "question": ask})) else ""

    def _notes_for(self, question: str, passages: str, docs: list[Document]) -> str:
        """Rule sentences with citations, kept apart from the query. A sentence stays only if
        the passage it cites really contains its words, so a citation can't be made up."""
        reply = _FENCE.sub("", self._rule_notes.invoke({"context": passages, "question": question}))
        kept: list[str] = []
        for sentence, number in _CITED_SENTENCE.findall(reply):
            i = int(number)
            line = f"{sentence.strip()} [{i}]"
            if 1 <= i <= len(docs) and line not in kept and _supported(sentence, docs[i - 1].page_content):
                kept.append(line)
        return "\n".join(kept)

    def _check_offline(self, result: SQLResult) -> SQLResult:
        """Without a database, a SQLite query can still be compiled against the supplied schema."""
        if self._connected() or not result.ok or result.dialect != "sqlite" or not self.schema.strip():
            return result
        db = sqlite3.connect(":memory:")
        try:
            db.executescript(self.schema)
        except sqlite3.Error:  # a table list rather than DDL: nothing to check against
            db.close()
            return result
        try:
            db.execute(f"EXPLAIN {result.sql}")
        except sqlite3.Error as e:
            return replace(result, error=str(e))
        finally:
            db.close()
        return result

    def run_sql(self, sql: str) -> SQLResult:
        """Run the user's own SQL (``/run SELECT ...``) through the same read-only checks."""
        from .sql import check_read_only, run_query

        agent = self.sql_agent
        r = SQLResult(question=f"/run {sql}", sql=sql.strip(), dialect=agent.dialect, attempts=1)
        if agent.catalog is not None:  # many databases: "/run @name SELECT ..." or the last one used
            m = re.match(r"^\s*@(\S+)\s+(.+)$", sql, re.DOTALL)
            if m:
                name, sql = m.group(1), m.group(2)
                r.sql = sql.strip()
                if name not in agent.catalog:
                    r.error = f"Unknown database {name!r}."
                    return r
                agent.use_database(name)
            elif not agent.database_name:
                r.error = "Several databases are connected: start with @<database>, e.g. /run @concert_singer SELECT ..."
                return r
            r.database = agent.database_name
        try:
            r.sql = check_read_only(sql, agent.dialect)
            if agent.db is None:
                return r
            res = run_query(agent.db, r.sql, agent.max_rows)
        except Exception as e:
            r.error = str(getattr(e, "orig", None) or e).strip().splitlines()[0]
            return r
        r.columns, r.rows, r.truncated, r.executed = res.columns, res.rows, res.truncated, True
        return r

    # -- public -----------------------------------------------------------------------

    def ask(self, question: str, route: str | None = None) -> Answer:
        final = None
        for part in self.stream(question, route):
            if isinstance(part, Answer):
                final = part
        return final

    def stream(self, question: str, route: str | None = None) -> Iterator[str | Answer]:
        """Yields answer text as it arrives, then the final ``Answer``."""
        m = re.match(r"^\s*/run\s+(.+)$", question, re.DOTALL | re.IGNORECASE)
        if m:  # the user's own query: no model involved
            r = self.run_sql(m.group(1))
            if r.ok and r.executed:
                more = ", more not shown" if r.truncated else ""
                text = f"```sql\n{r.sql}\n```\n\n{len(r.rows)} rows{more}."
            else:
                text = format_sql_answer(r)
            yield text
            yield Answer(text, question, question, [], ("SQL",), r)
            return
        standalone = self._standalone(question)
        routes = self.plan(standalone, route)
        docs: list[Document] = []
        sql_result: SQLResult | None = None
        parts: list[str] = []

        with_rules = "SQL" in routes and "DOC" in routes
        if with_rules:  # the SQL specialist reads the documents and cites them itself
            routes = [r for r in routes if r != "DOC"]

        if len(routes) == 1:
            gen = self._run_route(routes[0], standalone, docs, stream=True, with_rules=with_rules)
            chunks = []
            for item in gen:
                if isinstance(item, SQLResult):
                    sql_result = item
                else:
                    chunks.append(item)
                    yield item
            text = "".join(chunks).strip()
        else:
            sql_text = ""
            for r in routes:
                start = len(docs) + 1
                out = []
                for item in self._run_route(r, standalone, docs, stream=False, with_rules=with_rules):
                    if isinstance(item, SQLResult):
                        sql_result = item
                    else:
                        out.append(item)
                piece = "".join(out).strip()
                if r == "SQL":
                    sql_text = piece  # shown as written, so combining can't rewrite the query
                elif piece and not piece.startswith(NOT_FOUND):
                    parts.append(f"## {r} specialist (passages [{start}]...)\n{piece}")
            chunks = []
            if parts:
                for tok in self._combine.stream({"parts": "\n\n".join(parts), "question": standalone}):
                    chunks.append(tok)
                    yield tok
            if sql_text:
                chunks.append(f"\n\n{sql_text}" if chunks else sql_text)
                yield chunks[-1]
            text = "".join(chunks).strip()
            if not text:
                text = NOT_FOUND
                yield text

        self.history += [HumanMessage(question), AIMessage(text)]
        yield Answer(text, question, standalone, docs, tuple(routes), sql_result)

    def _run_route(self, route: str, question: str, docs: list[Document], stream: bool, with_rules: bool = False):
        """Run one specialist. Appends its passages to ``docs`` (shared citation numbering)."""
        start = len(docs) + 1
        if route == "SQL":
            scoped = self._has_database_folders()
            context = self._sql_context(question, scoped=scoped)
            docs.extend(context)
            has_schema = (
                self.sql_agent.db is not None
                or self.sql_agent.catalog is not None
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
            rules = "\n\n".join(
                f"[{i}] (source: {citation_label(d)})\n{d.page_content}"
                for i, d in enumerate(context, start)
                if d.metadata.get("category") == "doc"
            ) if with_rules or _USES_RULES.search(question) else ""
            if rules:
                missing = self._missing_rule(question, rules)
                if missing:
                    yield missing
                    return
            result = self.sql_agent.run(question, _numbered(context, start) if context else "", self.schema, examples,
                                        context_fn=self._database_docs if scoped else None)
            result = self._check_offline(result)
            yield result
            if rules and result.sql:
                notes = self._notes_for(question, rules, docs)
                if notes:
                    yield notes + "\n\n"
            summary_tokens = []
            if result.ok and result.executed:
                for tok in (self.sql_agent.stream_summary(result) if stream else [self.sql_agent.summarize(result)]):
                    summary_tokens.append(tok)
                    yield tok
            # a mapping that is itself a query would show the reader a second, different query
            shown = replace(result, mapping="") if _SQL_TEXT.search(result.mapping or "") else result
            rest = format_sql_answer(shown)
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
