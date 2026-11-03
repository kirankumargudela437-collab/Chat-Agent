"""The question-answering chain.

1. **Condense**: with chat history, the LLM rewrites a follow-up ("How do I apply
   for it?") into a standalone question ("How do I apply for the refund
   policy?") before searching. Without history this step is skipped.
2. **Retrieve** the top-k chunks for the standalone question.
3. **Answer** with a strict prompt: only the numbered context, cite as [1], [2],
   and say so plainly when the answer is not there. The answer step gets the
   standalone question, not the chat history: earlier answers carry citation
   numbers that point at a different turn's passages, and facts that are not in
   the current context.

Unlike the original blueprint's ``format_docs``, each context block keeps its
source file, page and section, so the model can cite them and the app can show
them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

NOT_FOUND = "I couldn't find that in the documents."

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite follow-up questions for a document search engine. You never answer them.\n"
            "Given a conversation and a follow-up question, rewrite the follow-up so it can be "
            "understood on its own, replacing pronouns and references like 'it' or 'that one' "
            "with what they refer to. If it is already standalone, return it unchanged.\n"
            "Output only the rewritten question, on one line, with no preamble.\n\n"
            "Example:\n"
            "Conversation:\nUser: What is the refund policy?\nAssistant: Refunds within 30 days [1].\n"
            "Follow-up: How do I apply for it?\n"
            "Rewritten: How do I apply for a refund under the refund policy?",
        ),
        ("human", "Conversation:\n{chat_history}\n\nFollow-up: {question}\nRewritten:"),
    ]
)

_PREAMBLE = re.compile(
    r"^\s*(?:here(?:'s| is) (?:the |a )?(?:rewritten|standalone) question:?|"
    r"(?:rewritten|standalone)(?: question)?:)\s*",
    re.IGNORECASE,
)


def clean_condensed(text: str, fallback: str) -> str:
    """Strip the preambles and quotes small models add around the rewritten question."""
    lines = [ln for ln in (_PREAMBLE.sub("", ln).strip() for ln in text.strip().splitlines()) if ln]
    out = lines[0].strip().strip('"\'`').strip() if lines else ""
    return out or fallback


def history_as_text(history: Sequence[BaseMessage]) -> str:
    out = []
    for m in history:
        who = "User" if isinstance(m, HumanMessage) else "Assistant"
        text = re.sub(r"\s+([.,;:!?])", r"\1", _CITATION.sub("", m.content))
        out.append(f"{who}: {text.strip()}")
    return "\n".join(out)


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions about the user's documents.\n"
            "Rules:\n"
            "- Use ONLY the numbered context passages below. Do not use outside knowledge, "
            "even if you know the answer.\n"
            "- After each claim, cite the passage(s) it came from as [1], [2], etc.\n"
            "- If the passages do not contain the answer, reply exactly: "
            f'"{NOT_FOUND}" and nothing else.\n'
            "- If passages disagree, say so and cite both.\n"
            "- Be concise.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def citation_label(doc: Document) -> str:
    m = doc.metadata
    label = m.get("source", "unknown")
    if m.get("page") is not None:
        label += f", page {m['page']}"
    if m.get("section"):
        label += f", section \"{m['section']}\""
    return label


def format_docs(docs: Sequence[Document]) -> str:
    """Numbered context blocks that keep the source metadata (fixes the blueprint)."""
    if not docs:
        return "(no passages found)"
    return "\n\n".join(
        f"[{i}] (source: {citation_label(d)})\n{d.page_content}" for i, d in enumerate(docs, 1)
    )


# [1]  [1, 2]  [1-3]  [1,3-4]   (also matches the [1][2] form, one bracket at a time)
_CITATION = re.compile(r"\[\s*(\d+(?:\s*[-–,]\s*\d+)*)\s*\]")


def cited_indices(answer: str, n_docs: int) -> list[int]:
    found: set[int] = set()
    for group in _CITATION.findall(answer):
        for part in group.split(","):
            ends = [int(x) for x in re.split(r"[-–]", part)]
            lo, hi = ends[0], ends[-1]
            if 0 <= hi - lo <= n_docs:  # ignore nonsense like [1-9999]
                found.update(range(lo, hi + 1))
    return sorted(i for i in found if 1 <= i <= n_docs)


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower().replace("’", "'").replace("'", ""))


def is_not_found(answer: str) -> bool:
    """True for the fixed refusal, tolerating case, punctuation and trailing text."""
    return _normalise(answer).strip().startswith(_normalise(NOT_FOUND).strip())


@dataclass
class Answer:
    answer: str
    question: str
    standalone_question: str
    docs: list[Document] = field(default_factory=list)

    @property
    def cited_docs(self) -> list[tuple[int, Document]]:
        """Passages the answer cited. Empty for a refusal or an answer with no citations.

        It never falls back to all retrieved passages: listing sources the answer did
        not use makes an unsupported answer look cited.
        """
        if self.not_found:
            return []
        return [(i, self.docs[i - 1]) for i in cited_indices(self.answer, len(self.docs))]

    @property
    def not_found(self) -> bool:
        return is_not_found(self.answer)

    @property
    def uncited(self) -> bool:
        """An answer that claims something but cites no passage: treat with care."""
        return not self.not_found and not self.cited_docs

    def sources_text(self) -> str:
        return "\n".join(f"  [{i}] {citation_label(d)}" for i, d in self.cited_docs)


class DocQA:
    """Stateful chat over the indexed documents."""

    def __init__(self, llm: BaseChatModel, retriever: BaseRetriever, max_history_messages: int = 8):
        self.llm = llm
        self.retriever = retriever
        self.max_history_messages = max_history_messages
        self.history: list[BaseMessage] = []
        self._condense = CONDENSE_PROMPT | llm | StrOutputParser()
        self._answer = ANSWER_PROMPT | llm | StrOutputParser()

    def reset(self) -> None:
        self.history.clear()

    def _recent_history(self) -> list[BaseMessage]:
        return self.history[-self.max_history_messages :]

    def _prepare(self, question: str) -> tuple[str, list[Document], dict]:
        history = self._recent_history()
        standalone = question
        if history:
            raw = self._condense.invoke({"chat_history": history_as_text(history), "question": question})
            standalone = clean_condensed(raw, question)
        docs = list(self.retriever.invoke(standalone))
        inputs = {"context": format_docs(docs), "question": standalone}
        return standalone, docs, inputs

    def _remember(self, question: str, answer: str) -> None:
        self.history += [HumanMessage(question), AIMessage(answer)]

    def ask(self, question: str) -> Answer:
        standalone, docs, inputs = self._prepare(question)
        text = self._answer.invoke(inputs).strip() if docs else NOT_FOUND
        self._remember(question, text)
        return Answer(text, question, standalone, docs)

    def stream(self, question: str) -> Iterator[str | Answer]:
        """Yields answer tokens as they arrive, then the final ``Answer``."""
        standalone, docs, inputs = self._prepare(question)
        if not docs:
            yield NOT_FOUND
            parts = [NOT_FOUND]
        else:
            parts = []
            for tok in self._answer.stream(inputs):
                parts.append(tok)
                yield tok
        text = "".join(parts).strip()
        self._remember(question, text)
        yield Answer(text, question, standalone, docs)
