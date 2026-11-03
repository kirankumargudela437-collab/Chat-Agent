"""The question-answering chain.

1. **Condense**: with chat history, the LLM rewrites a follow-up ("How do I apply
   for it?") into a standalone question ("How do I apply for the refund
   policy?") before searching. Without history this step is skipped.
2. **Retrieve** the top-k chunks for the standalone question.
3. **Answer** with a strict prompt: only the numbered context, cite as [1], [2],
   and say so plainly when the answer is not there.

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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever

NOT_FOUND = "I couldn't find that in the documents."

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the chat history and the latest user question, rewrite the question so it "
            "can be understood on its own, replacing pronouns and references like 'it' or "
            "'that one' with what they refer to. Do NOT answer it. If it is already "
            "standalone, return it unchanged. Output only the question.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

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
        MessagesPlaceholder("chat_history"),
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


def cited_indices(answer: str, n_docs: int) -> list[int]:
    found = sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer)})
    return [i for i in found if 1 <= i <= n_docs]


@dataclass
class Answer:
    answer: str
    question: str
    standalone_question: str
    docs: list[Document] = field(default_factory=list)

    @property
    def cited_docs(self) -> list[tuple[int, Document]]:
        """Passages the answer cited; all retrieved passages if it cited none."""
        if self.answer.strip() == NOT_FOUND:
            return []
        idx = cited_indices(self.answer, len(self.docs)) or range(1, len(self.docs) + 1)
        return [(i, self.docs[i - 1]) for i in idx]

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
            standalone = self._condense.invoke({"chat_history": history, "question": question}).strip()
            standalone = standalone or question
        docs = list(self.retriever.invoke(standalone))
        inputs = {"context": format_docs(docs), "chat_history": history, "question": question}
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
