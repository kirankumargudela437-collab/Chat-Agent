from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever

import pytest

from docqa.chain import NOT_FOUND, Answer, DocQA, clean_condensed, cited_indices, format_docs, is_not_found


class SpyRetriever(BaseRetriever):
    docs: list[Document]
    queries: list[str] = []

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        self.queries.append(query)
        return self.docs


DOCS = [
    Document("Refunds are allowed within 30 days.", metadata={"source": "policies/refunds.pdf", "page": 1}),
    Document("Email billing@example.com to apply.", metadata={"source": "policies/refunds.pdf", "page": 2}),
    Document("25 days of leave.", metadata={"source": "handbook.md", "section": "Handbook > Leave"}),
]


class RecordingLLM(FakeListChatModel):
    prompts: list = []

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        self.prompts.append(messages)
        return super()._call(messages, stop, run_manager, **kwargs)


def test_format_docs_keeps_source_page_and_section():
    text = format_docs(DOCS)
    assert "[1] (source: policies/refunds.pdf, page 1)" in text
    assert "[2] (source: policies/refunds.pdf, page 2)" in text
    assert '[3] (source: handbook.md, section "Handbook > Leave")' in text


def test_follow_up_is_rewritten_before_retrieval():
    retriever = SpyRetriever(docs=DOCS, queries=[])
    llm = RecordingLLM(responses=[
        "Refunds are allowed within 30 days [1].",            # answer 1 (no condense: no history)
        "How do I apply for the refund policy?",              # condense for turn 2
        "Email billing@example.com with your order [2].",    # answer 2
    ], prompts=[])
    qa = DocQA(llm, retriever)

    a1 = qa.ask("What is the refund policy?")
    assert retriever.queries == ["What is the refund policy?"]
    assert a1.standalone_question == a1.question

    a2 = qa.ask("How do I apply for it?")
    assert retriever.queries[-1] == "How do I apply for the refund policy?"
    assert [i for i, _ in a2.cited_docs] == [2]
    assert "page 2" in a2.sources_text()

    # the condense prompt saw the history; the answer prompt was grounded in numbered context
    condense_msgs = llm.prompts[1]
    assert any("refund policy" in m.content for m in condense_msgs)
    system = llm.prompts[2][0].content
    assert "ONLY the numbered context" in system and "source: policies/refunds.pdf, page 2" in system


def test_not_found_has_no_sources():
    qa = DocQA(FakeListChatModel(responses=[NOT_FOUND]), SpyRetriever(docs=DOCS, queries=[]))
    ans = qa.ask("What is the CEO's salary?")
    assert ans.answer == NOT_FOUND and ans.cited_docs == []


def test_empty_index_skips_llm():
    qa = DocQA(FakeListChatModel(responses=[]), SpyRetriever(docs=[], queries=[]))
    assert qa.ask("anything?").answer == NOT_FOUND


def test_stream_yields_tokens_then_answer():
    qa = DocQA(FakeListChatModel(responses=["Thirty days [1]."]), SpyRetriever(docs=DOCS, queries=[]))
    parts = list(qa.stream("Refund window?"))
    assert isinstance(parts[-1], Answer)
    assert "".join(p for p in parts[:-1]) == "Thirty days [1]."
    assert len(qa.history) == 2


def test_uncited_answer_lists_no_sources():
    # Listing every retrieved passage would make an unsupported answer look cited.
    ans = Answer("Thirty days.", "q", "q", DOCS)
    assert ans.cited_docs == [] and ans.uncited


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Thirty days [1].", [1]),
        ("Thirty days [1, 2].", [1, 2]),
        ("Thirty days [1][3].", [1, 3]),
        ("Thirty days [1-3].", [1, 2, 3]),
        ("Thirty days [2,3] and [9].", [2, 3]),
        ("Thirty days (Source 2).", []),
        ("Thirty days [1-9999].", []),
    ],
)
def test_citation_forms(text, expected):
    assert cited_indices(text, 3) == expected


@pytest.mark.parametrize(
    "text",
    [
        NOT_FOUND,
        "I couldn't find that in the documents",
        "i couldn’t find that in the documents!",
        "I couldn't find that in the documents. However, refunds are usually 30 days.",
    ],
)
def test_refusal_variants_have_no_sources(text):
    ans = Answer(text, "q", "q", DOCS)
    assert is_not_found(text) and ans.cited_docs == [] and not ans.uncited


@pytest.mark.parametrize(
    "raw",
    [
        "How do I apply for a refund?",
        "Here is the rewritten question: How do I apply for a refund?",
        "Rewritten question: \"How do I apply for a refund?\"",
        "Standalone question:\nHow do I apply for a refund?\n\nThis replaces 'it'.",
    ],
)
def test_condensed_question_is_cleaned(raw):
    assert clean_condensed(raw, "fallback") == "How do I apply for a refund?"


def test_answer_step_sees_standalone_question_not_old_answers():
    retriever = SpyRetriever(docs=DOCS, queries=[])
    llm = RecordingLLM(responses=[
        "Refunds are allowed within 30 days [1].",
        "How do I apply for the refund policy?",
        "Email billing@example.com [2].",
    ], prompts=[])
    qa = DocQA(llm, retriever)
    qa.ask("What is the refund policy?")
    qa.ask("How do I apply for it?")

    # condense saw the history as text, with old citation markers stripped
    condense_human = llm.prompts[1][-1].content
    assert "Assistant: Refunds are allowed within 30 days." in condense_human
    assert "[1]" not in condense_human

    # answer step: system context + the standalone question only, no earlier turns
    answer_msgs = llm.prompts[2]
    assert len(answer_msgs) == 2
    assert answer_msgs[1].content == "How do I apply for the refund policy?"
