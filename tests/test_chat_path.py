"""What the chat shows when no database is connected: the query as written, cited rules, sections."""

import docx
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

from docqa import Settings
from docqa.agent import Agent
from docqa.indexing import open_record_manager, open_vectorstore, sync_documents
from docqa.loaders import load_file
from docqa.sql import SQLAgent

SCHEMA = "CREATE TABLE trip(trip_id INTEGER PRIMARY KEY, kind TEXT, miles INTEGER);"
GOOD = "SELECT trip_id FROM trip WHERE kind = 'L' AND miles >= 500"
BAD = "SELECT trip_id FROM trip WHERE distance >= 500"


class CountingLLM(FakeListChatModel):
    calls: list = []

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(messages)
        return super()._call(messages, stop, run_manager, **kwargs)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(messages)
        yield from super()._stream(messages, stop, run_manager, **kwargs)


@pytest.fixture
def index(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "travel.md").write_text("# Travel\n\n## Long trips\n\nKind L is long. A long trip has at least 500 miles.\n", encoding="utf-8")
    settings = Settings(docs_dir=docs, data_dir=tmp_path / "data")
    vs = open_vectorstore(settings, DeterministicFakeEmbedding(size=32))
    sync_documents(settings, vs, open_record_manager(settings))
    return vs


def agent(index, responses):
    llm = CountingLLM(responses=responses, calls=[])
    return Agent(llm, index, sql_agent=SQLAgent(llm, dialect="sqlite"), schema=SCHEMA), llm


def test_mixed_answer_keeps_the_query_and_adds_cited_rules_and_the_label(index):
    chat, llm = agent(index, ["SQL, DOC", "READY", "kind and miles", GOOD, "A long trip has at least 500 miles. [1]\nTrips earn nine gold stars daily. [1]"])
    ans = chat.ask("Write SQL listing long trips under the travel policy. Cite it.")
    assert f"```sql\n{GOOD}\n```" in ans.answer
    assert ans.answer.startswith("A long trip has at least 500 miles. [1]")
    assert "gold stars" not in ans.answer  # the cited passage doesn't say this
    assert "not run: no database is connected" in ans.answer
    assert ans.routes == ("SQL",) and len(ans.cited_docs) == 1
    assert not any("Partial answers" in str(m) for m in llm.calls)  # the combining step never saw the query


def test_rule_notes_cannot_add_a_second_query_or_an_uncited_claim(index):
    chat, _ = agent(index, ["SQL", "READY", "kind", GOOD, "Use this:\n```sql\nSELECT 1\n```\nKind L is long. [1]"])
    ans = chat.ask("Write SQL for long trips under the policy.")
    assert "SELECT 1" not in ans.answer and "Kind L is long. [1]" in ans.answer
    chat, _ = agent(index, ["SQL", "READY", "kind", GOOD, "NONE"])
    assert "NONE" not in chat.ask("Write SQL for long trips under the policy.").answer


def test_a_missing_rule_gets_a_question_and_no_query(index):
    chat, llm = agent(index, ["SQL", "ASK: How is the toll charge calculated?", "I couldn't find that in the documents."])
    ans = chat.ask("Write SQL for the toll charge per trip from the policy.")
    assert ans.answer == "How is the toll charge calculated?"
    assert ans.sql is None and "```" not in ans.answer and len(llm.calls) == 3


def test_a_question_the_documents_can_answer_does_not_stop_the_query(index):
    chat, _ = agent(index, ["SQL", "ASK: How long is a long trip?", "At least 500 miles [1].", "kind", GOOD, "Kind L is long. [1]"])
    ans = chat.ask("Write SQL for long trips under the policy.")
    assert GOOD in ans.answer and "How long" not in ans.answer


def test_plain_sql_skips_the_rule_steps(index):
    chat, llm = agent(index, ["SQL", "kind", GOOD])
    ans = chat.ask("Write SQL listing trip IDs for kind L with 500 miles or more.")
    assert f"```sql\n{GOOD}\n```" in ans.answer and len(llm.calls) == 3


def test_a_document_question_offline_does_not_write_sql(index):
    chat, _ = agent(index, ["SQL, DOC", "A long trip has at least 500 miles [1]."])
    ans = chat.ask("How long is a long trip?")
    assert ans.routes == ("DOC",) and ans.sql is None and "```" not in ans.answer


def test_a_query_the_schema_rejects_is_not_called_valid(index):
    chat, _ = agent(index, ["SQL", "kind", BAD])
    ans = chat.ask("Write SQL for trips of 500 miles or more.")
    assert "Checked as valid" not in ans.answer and "no such column: distance" in ans.answer
    assert BAD in ans.answer


def test_a_mapping_written_as_a_query_is_not_shown(index):
    chat, _ = agent(index, ["SQL", "SELECT trip_id FROM trip", GOOD])
    ans = chat.ask("Write SQL for trips of 500 miles or more.")
    assert "How the question maps" not in ans.answer and ans.answer.count("SELECT") == 1


def test_word_headings_become_sections(tmp_path):
    path = tmp_path / "guide.docx"
    d = docx.Document()
    d.add_paragraph("Intro line.")
    d.add_heading("Claims", level=1)
    d.add_paragraph("File within a week.")
    d.add_heading("Limits", level=2)
    d.add_paragraph("Up to 900 cents.")
    d.add_table(rows=1, cols=2).rows[0].cells[0].text = "Grade A"
    d.add_heading("Appeals", level=1)
    d.add_paragraph("Write to the board.")
    d.save(str(path))
    got = load_file(path, "guide.docx")
    assert [x.metadata.get("section") for x in got] == [None, "Claims", "Claims > Limits", "Appeals"]
    assert "File within a week." in got[1].page_content and got[1].page_content.startswith("Claims")
    assert "Grade A" in got[2].page_content
