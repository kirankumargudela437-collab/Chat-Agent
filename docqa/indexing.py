"""Incremental sync of the docs folder into the vector store.

LangChain's indexing API keeps a SQLite "record manager" next to Chroma with a
hash of every chunk it has written. On each sync:

* unchanged chunks are skipped (no re-embedding),
* chunks of edited files are replaced,
* chunks of deleted files are removed (``cleanup="full"``).

Full cleanup deletes every chunk it did not see in this run, so two guards keep
a bad sync from emptying the index:

* a file that fails to load (locked, not downloaded from OneDrive yet, half
  written) keeps its previous chunks until it loads again,
* a sync that finds no files at all is refused while the index still has
  chunks, unless ``allow_empty=True`` (``docqa sync --allow-empty``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import chromadb

from langchain_chroma import Chroma
from langchain_classic.indexes import SQLRecordManager
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.indexing import index

from .config import Settings
from .loaders import LoadReport, load_and_split
from .providers import make_embeddings

log = logging.getLogger(__name__)


class EmptySyncError(ValueError):
    pass


@dataclass
class SyncResult:
    num_added: int
    num_updated: int
    num_skipped: int
    num_deleted: int
    num_chunks: int
    report: LoadReport
    kept: tuple[str, ...] = ()

    def summary(self) -> str:
        lines = [
            f"{len(self.report.files)} files, {self.num_chunks} chunks: "
            f"{self.num_added} added, {self.num_updated} updated, "
            f"{self.num_skipped} unchanged, {self.num_deleted} deleted"
        ]
        lines += [f"  warning  {f}: {m}" for f, m in self.report.warnings.items()]
        lines += [
            f"  error    {f}: {m}" + ("  (kept its previous chunks)" if f in self.kept else "")
            for f, m in self.report.errors.items()
        ]
        return "\n".join(lines)


def open_vectorstore(settings: Settings, embeddings: Embeddings | None = None) -> Chroma:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings or make_embeddings(settings),
        persist_directory=str(settings.chroma_dir),
        client_settings=chromadb.config.Settings(
            anonymized_telemetry=False, is_persistent=True, persist_directory=str(settings.chroma_dir)
        ),
    )


def open_record_manager(settings: Settings) -> SQLRecordManager:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    rm = SQLRecordManager(f"chroma/{settings.collection_name}", db_url=settings.record_manager_url)
    rm.create_schema()
    return rm


def _stored_chunks(vectorstore: Chroma, source: str) -> list[Document]:
    """The chunks currently indexed for one file, exactly as they were written."""
    got = vectorstore.get(where={"source": source}, include=["documents", "metadatas"])
    return [Document(page_content=t, metadata=m) for t, m in zip(got["documents"], got["metadatas"])]


def sync_documents(
    settings: Settings,
    vectorstore: Chroma | None = None,
    record_manager: SQLRecordManager | None = None,
    allow_empty: bool = False,
) -> SyncResult:
    vectorstore = vectorstore or open_vectorstore(settings)
    record_manager = record_manager or open_record_manager(settings)

    chunks, report = load_and_split(settings)

    if not report.files and not report.errors and not allow_empty:
        indexed = len(vectorstore.get(include=[])["ids"])
        if indexed:
            raise EmptySyncError(
                f"No documents found in {settings.docs_dir.resolve()}, but the index has {indexed} chunks. "
                "Refusing to delete them all; check DOCQA_DOCS_DIR, or run "
                "'docqa sync --allow-empty' if you really emptied the folder."
            )

    # Re-submit the stored chunks of files that failed to load: their hashes match,
    # so they count as unchanged instead of being deleted by the full cleanup.
    kept = []
    for source in report.errors:
        previous = _stored_chunks(vectorstore, source)
        if previous:
            log.warning("Keeping %d previous chunks of %s until it loads again", len(previous), source)
            chunks.extend(previous)
            kept.append(source)

    result = index(
        chunks,
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="source",
        key_encoder="sha256",
    )
    return SyncResult(
        num_added=result["num_added"],
        num_updated=result["num_updated"],
        num_skipped=result["num_skipped"],
        num_deleted=result["num_deleted"],
        num_chunks=len(chunks),
        report=report,
        kept=tuple(kept),
    )
