"""docqa: a grounded chat agent over your business documents, source code and SQL databases."""

from .agent import Agent
from .chain import NOT_FOUND, Answer, DocQA, format_docs
from .config import Settings
from .indexing import open_vectorstore, sync_documents

__all__ = [
    "Agent", "Answer", "DocQA", "NOT_FOUND", "Settings", "build_agent", "build_sql_agent",
    "format_docs", "open_vectorstore", "sync_documents",
]


def build_sql_agent(settings: Settings, llm=None):
    """The SQL specialist, with a read-only database connection when DOCQA_DATABASE_URL is set."""
    from .providers import make_llm
    from .sql import SQLAgent, open_database

    db = open_database(settings.database_url, settings.sql_tables) if settings.database_url else None
    return SQLAgent(
        llm or make_llm(settings),
        db,
        dialect=settings.sql_dialect,
        max_attempts=settings.sql_max_attempts,
        max_rows=settings.sql_max_rows,
        schema_linking=settings.sql_schema_linking,
        max_tables=settings.sql_max_tables,
        schema_chars=settings.sql_schema_chars,
    )


def build_agent(settings: Settings | None = None, vectorstore=None, schema: str = "") -> Agent:
    """Wire the configured LLM, vector store and (optional) database into one Agent."""
    from .providers import make_llm

    settings = settings or Settings.from_env()
    llm = make_llm(settings)
    return Agent(
        llm,
        vectorstore or open_vectorstore(settings),
        top_k=settings.top_k,
        sql_agent=build_sql_agent(settings, llm),
        schema=schema,
        hybrid=settings.hybrid_search,
    )
