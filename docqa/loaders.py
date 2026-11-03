"""Turn a folder of PDF / DOCX / Markdown / text files into chunked Documents.

Every chunk carries the metadata a citation needs:

* ``source``  path relative to the docs folder, e.g. ``policies/refunds.pdf``
* ``page``    1-based page number (PDFs only, omitted otherwise)
* ``section`` heading trail for Markdown, e.g. ``Refunds > How to apply``

Other loader metadata (producer, creation date, ...) is dropped on purpose: the
indexing API hashes content *and* metadata, so volatile fields would make
unchanged files look changed on every sync.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .config import Settings

log = logging.getLogger(__name__)

_MD_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


@dataclass
class LoadReport:
    files: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    warnings: dict[str, str] = field(default_factory=dict)


def iter_files(folder: Path, extensions: tuple[str, ...]) -> list[Path]:
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in extensions
        and not any(part.startswith(".") for part in p.relative_to(folder).parts)
        and not p.name.startswith("~$")  # Office lock files
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def _load_pdf(path: Path, rel: str) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": rel, "page": i}))
    return docs


def _load_docx(path: Path, rel: str) -> list[Document]:
    import docx2txt

    text = (docx2txt.process(str(path)) or "").strip()
    return [Document(page_content=text, metadata={"source": rel})] if text else []


def _load_markdown(path: Path, rel: str) -> list[Document]:
    return _load_markdown_text(_read_text(path), rel)


def _load_text(path: Path, rel: str) -> list[Document]:
    text = _read_text(path).strip()
    return [Document(page_content=text, metadata={"source": rel})] if text else []


def _load_docling(path: Path, rel: str) -> list[Document]:
    """Docling export to Markdown: keeps tables and headings, OCRs scans.

    Page numbers are not tracked in this mode; citations name the file and the
    Markdown section instead.
    """
    from langchain_docling import DoclingLoader
    from langchain_docling.loader import ExportType

    out = []
    for d in DoclingLoader(file_path=str(path), export_type=ExportType.MARKDOWN).load():
        out.extend(_load_markdown_text(d.page_content, rel))
    return out


def _load_markdown_text(text: str, rel: str) -> list[Document]:
    splitter = MarkdownHeaderTextSplitter(_MD_HEADERS, strip_headers=False)
    docs = []
    for d in splitter.split_text(text):
        trail = " > ".join(d.metadata[k] for _, k in _MD_HEADERS if k in d.metadata)
        meta = {"source": rel, **({"section": trail} if trail else {})}
        if d.page_content.strip():
            docs.append(Document(page_content=d.page_content, metadata=meta))
    return docs


def load_file(path: Path, rel: str, use_docling: bool = False) -> list[Document]:
    ext = path.suffix.lower()
    if use_docling and ext in (".pdf", ".docx"):
        return _load_docling(path, rel)
    if ext == ".pdf":
        return _load_pdf(path, rel)
    if ext == ".docx":
        return _load_docx(path, rel)
    if ext in (".md", ".markdown"):
        return _load_markdown(path, rel)
    return _load_text(path, rel)


def load_and_split(settings: Settings) -> tuple[list[Document], LoadReport]:
    folder = settings.docs_dir
    if not folder.is_dir():
        raise FileNotFoundError(f"Docs folder not found: {folder.resolve()}")

    report = LoadReport()
    raw: list[Document] = []
    for path in iter_files(folder, settings.extensions):
        rel = path.relative_to(folder).as_posix()
        try:
            docs = load_file(path, rel, settings.use_docling)
        except Exception as e:  # one bad file must not stop the sync
            log.warning("Could not read %s: %s", rel, e)
            report.errors[rel] = str(e)
            continue
        if not docs:
            report.warnings[rel] = (
                "no extractable text (scanned PDF? set DOCQA_USE_DOCLING=1 for OCR)"
                if path.suffix.lower() == ".pdf"
                else "file is empty"
            )
        report.files.append(rel)
        raw.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=False,  # a start index would shift on every edit above a chunk
    )
    return splitter.split_documents(raw), report
