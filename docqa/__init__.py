"""docqa: a grounded Q&A chat agent over a folder of changing documents."""

from .chain import NOT_FOUND, Answer, DocQA, format_docs
from .config import Settings
from .indexing import open_vectorstore, sync_documents

__all__ = ["Answer", "DocQA", "NOT_FOUND", "Settings", "build_agent", "format_docs", "open_vectorstore", "sync_documents"]


def build_agent(settings: Settings | None = None) -> DocQA:
    """Wire the configured LLM and vector store into a DocQA chat."""
    from .providers import make_llm

    settings = settings or Settings.from_env()
    retriever = open_vectorstore(settings).as_retriever(search_kwargs={"k": settings.top_k})
    return DocQA(make_llm(settings), retriever)
