"""Turn a folder of documents, source code and SQL into chunked Documents.

Every chunk is tagged with a ``category`` so questions can search only the
right kind of file:

* ``doc``      PDF, DOCX, Markdown, text, CSV (and XLSX / PPTX when installed)
* ``code``     source files, split along functions and classes for 25+ languages
* ``sql``      ``.sql`` files: DDL, views, scripts
* ``example``  "golden" question-to-SQL pairs found in ``.sql`` files (see below)

and carries the metadata a citation needs:

* ``source``    path relative to the docs folder, e.g. ``policies/refunds.pdf``
* ``page``      1-based page number (PDFs, slide number for PPTX)
* ``section``   heading trail for Markdown, sheet name for XLSX
* ``language``  for code and SQL, e.g. ``python``
* ``lines``     for code and SQL, e.g. ``12-48``
* ``question``  for golden SQL examples

Other loader metadata (producer, creation date, ...) is dropped on purpose: the
indexing API hashes content *and* metadata, so volatile fields would make
unchanged files look changed on every sync. Line numbers do change when lines
are added above a chunk, but that only re-embeds the file that was edited.

Golden SQL examples are written in any ``.sql`` file as a ``-- Question:``
comment followed by the query, ending at the ``;``::

    -- Question: Month-over-month revenue growth by region
    WITH monthly AS (...)
    SELECT ...;
"""

from __future__ import annotations

import importlib.util
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import Language, MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .config import Settings

log = logging.getLogger(__name__)

_MD_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]

DOC_EXTENSIONS = (".pdf", ".docx", ".md", ".markdown", ".txt", ".rst", ".csv")
OPTIONAL_DOC_EXTENSIONS = {".xlsx": "openpyxl", ".pptx": "pptx"}
SQL_EXTENSIONS = (".sql",)

# Extensions with a LangChain language-aware splitter (splits on class / def / {} first).
LANGUAGE_BY_EXT: dict[str, Language] = {
    ".py": Language.PYTHON, ".pyw": Language.PYTHON,
    ".js": Language.JS, ".jsx": Language.JS, ".mjs": Language.JS, ".cjs": Language.JS,
    ".ts": Language.TS, ".tsx": Language.TS,
    ".java": Language.JAVA, ".kt": Language.KOTLIN, ".kts": Language.KOTLIN, ".scala": Language.SCALA,
    ".c": Language.C, ".h": Language.C,
    ".cpp": Language.CPP, ".cc": Language.CPP, ".cxx": Language.CPP, ".hpp": Language.CPP, ".hh": Language.CPP,
    ".cs": Language.CSHARP, ".go": Language.GO, ".rs": Language.RUST, ".swift": Language.SWIFT,
    ".rb": Language.RUBY, ".php": Language.PHP, ".pl": Language.PERL, ".pm": Language.PERL,
    ".lua": Language.LUA, ".hs": Language.HASKELL, ".ex": Language.ELIXIR, ".exs": Language.ELIXIR,
    ".r": Language.R, ".sol": Language.SOL, ".cob": Language.COBOL, ".cbl": Language.COBOL,
    ".ps1": Language.POWERSHELL, ".psm1": Language.POWERSHELL,
    ".bas": Language.VISUALBASIC6, ".vb": Language.VISUALBASIC6,
    ".proto": Language.PROTO, ".html": Language.HTML, ".htm": Language.HTML, ".tex": Language.LATEX,
}

# Code without a dedicated splitter: split on blank lines, then lines. Still works,
# chunks are just less tidy. Add an extension here to index another language.
GENERIC_CODE_EXTENSIONS = (
    ".sh", ".bash", ".zsh", ".bat", ".cmd", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".css", ".scss", ".less", ".vue", ".svelte", ".dart", ".m", ".jl", ".groovy", ".gradle",
    ".tf", ".hcl", ".dockerfile", ".graphql", ".gql", ".xml",
)

# Folders that hold dependencies or build output, never your own code.
IGNORED_DIRS = {
    "node_modules", "__pycache__", "venv", "env", "site-packages", "dist", "build", "target",
    "bower_components", "vendor", "coverage", "out", "bin", "obj",
}

_SQL_SEPARATORS = [";\n\n", ";\n", "\n\n", "\n", " ", ""]
_GENERIC_SEPARATORS = ["\n\n\n", "\n\n", "\n", " ", ""]
_QUESTION = re.compile(r"^\s*--\s*(?:golden\s+)?(?:q|question)\s*:\s*(.+?)\s*$", re.IGNORECASE)


def supported_extensions(include_code: bool = True) -> tuple[str, ...]:
    exts = list(DOC_EXTENSIONS) + [e for e, mod in OPTIONAL_DOC_EXTENSIONS.items() if importlib.util.find_spec(mod)]
    if include_code:
        exts += list(SQL_EXTENSIONS) + list(LANGUAGE_BY_EXT) + list(GENERIC_CODE_EXTENSIONS)
    return tuple(exts)


def category_of(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in SQL_EXTENSIONS:
        return "sql"
    if ext in LANGUAGE_BY_EXT or ext in GENERIC_CODE_EXTENSIONS:
        return "code"
    return "doc"


def language_of(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in SQL_EXTENSIONS:
        return "sql"
    if ext in LANGUAGE_BY_EXT:
        return LANGUAGE_BY_EXT[ext].value
    return ext.lstrip(".")


@dataclass
class LoadReport:
    files: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    warnings: dict[str, str] = field(default_factory=dict)


def iter_files(folder: Path, extensions: tuple[str, ...]) -> list[Path]:
    out = []
    for p in folder.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in extensions:
            continue
        parts = p.relative_to(folder).parts
        if any(part.startswith(".") for part in parts):  # .git, .venv, .env, hidden files
            continue
        if any(part.lower() in IGNORED_DIRS for part in parts[:-1]):
            continue
        if p.name.startswith("~$") or ".min." in p.name.lower():  # Office lock files, minified bundles
            continue
        out.append(p)
    return sorted(out)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


# --------------------------------------------------------------------------- documents


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


def _load_xlsx(path: Path, rel: str) -> list[Document]:
    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    docs = []
    for ws in wb.worksheets:
        rows = [
            " | ".join("" if v is None else str(v) for v in row)
            for row in ws.iter_rows(values_only=True)
            if any(v is not None for v in row)
        ]
        if rows:
            docs.append(Document(page_content="\n".join(rows), metadata={"source": rel, "section": ws.title}))
    wb.close()
    return docs


def _load_pptx(path: Path, rel: str) -> list[Document]:
    from pptx import Presentation

    docs = []
    for i, slide in enumerate(Presentation(str(path)).slides, start=1):
        texts = [sh.text_frame.text for sh in slide.shapes if getattr(sh, "has_text_frame", False)]
        text = "\n".join(t for t in texts if t.strip()).strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": rel, "page": i}))
    return docs


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
    """Load a document file into unsplit Documents (code and SQL go through ``load_code``)."""
    ext = path.suffix.lower()
    if use_docling and ext in (".pdf", ".docx"):
        return _load_docling(path, rel)
    if ext == ".pdf":
        return _load_pdf(path, rel)
    if ext == ".docx":
        return _load_docx(path, rel)
    if ext == ".xlsx":
        return _load_xlsx(path, rel)
    if ext == ".pptx":
        return _load_pptx(path, rel)
    if ext in (".md", ".markdown"):
        return _load_markdown(path, rel)
    return _load_text(path, rel)


# --------------------------------------------------------------------------- code and SQL


def _line_ranges(text: str, chunks: list[str]) -> list[str | None]:
    """'start-end' line numbers of each chunk, found in order in the original text."""
    out, pos = [], 0
    for chunk in chunks:
        idx = text.find(chunk, pos)
        if idx < 0:
            idx = text.find(chunk)
        if idx < 0:
            out.append(None)
            continue
        start = text.count("\n", 0, idx) + 1
        end = start + chunk.count("\n")
        out.append(f"{start}-{end}")
        pos = idx + 1
    return out


def _split_with_lines(text: str, splitter: RecursiveCharacterTextSplitter, meta: dict, offset: int = 0) -> list[Document]:
    pieces = [p for p in splitter.split_text(text) if p.strip()]
    docs = []
    for piece, lines in zip(pieces, _line_ranges(text, pieces)):
        m = dict(meta)
        if lines:
            a, b = (int(x) + offset for x in lines.split("-"))
            m["lines"] = f"{a}-{b}"
        docs.append(Document(page_content=piece, metadata=m))
    return docs


def code_splitter(path: Path, settings: Settings) -> RecursiveCharacterTextSplitter:
    ext = path.suffix.lower()
    kw = dict(chunk_size=settings.code_chunk_size, chunk_overlap=settings.code_chunk_overlap)
    if ext in LANGUAGE_BY_EXT:
        return RecursiveCharacterTextSplitter.from_language(LANGUAGE_BY_EXT[ext], **kw)
    if ext in SQL_EXTENSIONS:
        return RecursiveCharacterTextSplitter(separators=_SQL_SEPARATORS, keep_separator="end", **kw)
    return RecursiveCharacterTextSplitter(separators=_GENERIC_SEPARATORS, **kw)


def split_golden_examples(text: str) -> tuple[list[tuple[str, str, int, int]], str]:
    """Pull ``-- Question:`` + query blocks out of a SQL file.

    Returns ``[(question, sql, first_line, last_line)]`` and the remaining text, with
    example lines blanked (not removed) so the line numbers of the rest stay right.
    """
    lines = text.splitlines()
    examples, keep = [], list(lines)
    i = 0
    while i < len(lines):
        m = _QUESTION.match(lines[i])
        if not m:
            i += 1
            continue
        start, body, j = i, [], i + 1
        while j < len(lines) and not _QUESTION.match(lines[j]):
            body.append(lines[j])
            if lines[j].rstrip().endswith(";"):
                j += 1
                break
            j += 1
        sql = "\n".join(body).strip()
        if sql:
            examples.append((m.group(1), sql, start + 1, j))
            for k in range(start, j):
                keep[k] = ""
        i = j
    return examples, "\n".join(keep)


def load_code(path: Path, rel: str, settings: Settings) -> list[Document]:
    """Read and split a code or SQL file into chunks with line numbers."""
    text = _read_text(path)
    if not text.strip():
        return []
    category, language = category_of(path), language_of(path)
    meta = {"source": rel, "category": category, "language": language}
    docs: list[Document] = []
    if category == "sql":
        examples, text = split_golden_examples(text)
        for question, sql, a, b in examples:
            docs.append(Document(
                page_content=f"Question: {question}\nSQL:\n{sql}",
                metadata={"source": rel, "category": "example", "language": "sql", "question": question, "lines": f"{a}-{b}"},
            ))
        if all(not ln.strip() or ln.strip().startswith("--") for ln in text.splitlines()):
            return docs  # only comments left (e.g. a golden file's header)
    return docs + _split_with_lines(text, code_splitter(path, settings), meta)


# --------------------------------------------------------------------------- whole folder


def load_and_split(settings: Settings) -> tuple[list[Document], LoadReport]:
    folder = settings.docs_dir
    if not folder.is_dir():
        raise FileNotFoundError(f"Docs folder not found: {folder.resolve()}")

    report = LoadReport()
    raw_docs: list[Document] = []
    chunks: list[Document] = []
    max_bytes = settings.max_file_kb * 1024
    for path in iter_files(folder, settings.extensions):
        rel = path.relative_to(folder).as_posix()
        category = category_of(path)
        try:
            if category != "doc" or path.suffix.lower() in (".txt", ".csv"):
                if path.stat().st_size > max_bytes:
                    report.warnings[rel] = f"skipped: larger than {settings.max_file_kb} KB (DOCQA_MAX_FILE_KB)"
                    report.files.append(rel)
                    continue
            docs = load_code(path, rel, settings) if category != "doc" else load_file(path, rel, settings.use_docling)
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
        (raw_docs if category == "doc" else chunks).extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=False,  # a start index would shift on every edit above a chunk
    )
    doc_chunks = splitter.split_documents(raw_docs)
    for c in doc_chunks:
        c.metadata["category"] = "doc"
    return doc_chunks + chunks, report
