from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from docqa.config import Settings


def make_pdf(path: Path, pages: list[str]) -> None:
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path))
    for text in pages:
        c.drawString(72, 720, text)
        c.showPage()
    c.save()


def make_docx(path: Path, paragraphs: list[str]) -> None:
    import docx

    d = docx.Document()
    for p in paragraphs:
        d.add_paragraph(p)
    d.save(str(path))


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "docs"
    (d / "policies").mkdir(parents=True)
    make_pdf(d / "policies" / "refunds.pdf", [
        "Refund policy: customers may request a refund within 30 days of purchase.",
        "To apply for a refund, email billing@example.com with your order number.",
    ])
    make_docx(d / "onboarding.docx", ["New hires must complete security training in week one."])
    (d / "handbook.md").write_text(
        "# Handbook\n\nWelcome.\n\n## Leave\n\nEmployees get 25 days of paid leave per year.\n",
        encoding="utf-8",
    )
    (d / ".hidden.md").write_text("should be ignored", encoding="utf-8")
    (d / "notes.xlsx").write_text("unsupported", encoding="utf-8")
    return d


@pytest.fixture
def settings(tmp_path: Path, docs_dir: Path) -> Settings:
    return Settings(docs_dir=docs_dir, data_dir=tmp_path / "data", chunk_size=200, chunk_overlap=40)


@pytest.fixture
def embeddings():
    return DeterministicFakeEmbedding(size=32)
