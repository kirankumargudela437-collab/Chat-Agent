"""docqa: a grounded chat agent over your business documents, source code and SQL databases."""

from .agent import Agent
from .chain import NOT_FOUND, Answer, DocQA, format_docs
from .config import Settings
from .indexing import open_vectorstore, sync_documents

__all__ = [
    "Agent", "Answer", "DocQA", "NOT_FOUND", "Settings", "build_agent", "build_sql_agent", "open_catalog",
    "format_docs", "open_vectorstore", "sync_documents",
]


def open_catalog(settings: Settings, embeddings=None, use_embeddings: bool = True):
    """The folder of SQLite databases in DOCQA_DATABASE_DIR, with cached routing cards."""
    from .multidb import DatabaseCatalog
    from .providers import make_embeddings

    if use_embeddings and embeddings is None and settings.embed_provider:
        embeddings = make_embeddings(settings)
    return DatabaseCatalog(
        settings.database_dir,
        embeddings=embeddings if use_embeddings else None,
        cache_dir=settings.data_dir,
        cache_key=f"{settings.embed_provider}:{settings.embed_model}" if use_embeddings else "no-embeddings",
    )


def build_sql_agent(settings: Settings, llm=None, catalog=None):
    """The SQL specialist: one read-only database (DOCQA_DATABASE_URL) or a folder of them (DOCQA_DATABASE_DIR)."""
    from .providers import make_llm
    from .sql import SQLAgent, open_database

    if settings.database_dir is not None and catalog is None:
        catalog = open_catalog(settings)
    db = None
    if catalog is None and settings.database_url:
        db = open_database(settings.database_url, settings.sql_tables)
    return SQLAgent(
        llm or make_llm(settings),
        db,
        dialect=settings.sql_dialect,
        max_attempts=settings.sql_max_attempts,
        max_rows=settings.sql_max_rows,
        schema_linking=settings.sql_schema_linking,
        max_tables=settings.sql_max_tables,
        schema_chars=settings.sql_schema_chars,
        catalog=catalog,
        db_router=settings.sql_db_router,
        db_candidates=settings.sql_db_candidates,
        value_lookup=settings.sql_value_lookup,
        sql_patterns=settings.sql_patterns,
        patterns_dir=str(settings.sql_patterns_dir) if settings.sql_patterns_dir else None,
        sql_candidates=settings.sql_candidates,
        route_margin=settings.sql_route_margin,
        route_choices=settings.sql_route_choices,
        answer_checks=settings.sql_answer_checks,
        review=settings.sql_review,
        independent_plans=settings.sql_independent_plans,
        schema_ttl=settings.sql_schema_ttl,
        answer_shape=settings.sql_answer_shape,
        grain_plan=settings.sql_grain_plan,
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
