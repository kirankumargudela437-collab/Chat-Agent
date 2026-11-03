"""Settings, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


DEFAULT_LLM_MODELS = {
    "ollama": "llama3.1",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5-5",
    "google": "gemini-2.5-flash",
}

DEFAULT_EMBED_MODELS = {
    "ollama": "nomic-embed-text",
    "openai": "text-embedding-3-small",
    "google": "models/text-embedding-004",
}


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _flag(name: str, default: str) -> bool:
    return _env(name, default).lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    docs_dir: Path = Path("./my_files")
    data_dir: Path = Path("./data")
    llm_provider: str = "ollama"
    llm_model: str = ""
    embed_provider: str = "ollama"
    embed_model: str = ""
    ollama_base_url: str | None = None
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 4
    use_docling: bool = False
    temperature: float = 0.0
    num_ctx: int = 8192
    # code and SQL
    include_code: bool = True
    code_chunk_size: int = 1500
    code_chunk_overlap: int = 150
    max_file_kb: int = 1024
    database_url: str | None = None
    database_dir: Path | None = None  # a folder of SQLite files; each question is routed to one of them
    sql_db_router: str = "llm"        # llm: the model picks among the top candidates; retrieval: top hit
    sql_db_candidates: int = 5
    sql_route_margin: float = 0.25    # schema-fit lead over #2 that skips the model's database pick
    sql_route_choices: int = 2        # otherwise the model chooses among this many top databases
    sql_dialect: str = ""            # used without a database; with one, taken from database_url
    sql_tables: tuple[str, ...] = ()  # limit the live schema to these tables (big databases)
    sql_max_attempts: int = 3
    sql_max_rows: int = 50
    sql_schema_linking: bool = True
    sql_value_lookup: bool = True     # find database values the question mentions (exact spellings)
    sql_answer_checks: bool = True    # EXPLAIN first; re-ask once when a result looks wrong (empty, NULLs, ...)
    sql_review: bool = False          # the model reviews its own query and first rows (one more call)
    sql_patterns: bool = True         # show worked examples of the SQL techniques the question needs
    sql_patterns_dir: Path | None = None  # your own pattern files (same format as docqa/sql_patterns)
    sql_candidates: int = 1           # >1: write several queries, keep the result most of them return
    hybrid_search: bool = True        # keyword (BM25) + vector search
    sql_max_tables: int = 20          # bigger databases: show only the most relevant tables
    sql_schema_chars: int = 12000     # cap on the schema text sent to the model
    extensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.extensions:
            from .loaders import supported_extensions

            self.extensions = supported_extensions(self.include_code, self.use_docling)
        from .sql import dialect_from_url

        if self.database_dir is not None:
            self.database_dir = Path(self.database_dir)
            self.sql_dialect = "sqlite"
        elif self.database_url:  # a connected database decides; DOCQA_SQL_DIALECT is for when there is none
            self.sql_dialect = dialect_from_url(self.database_url)
        elif not self.sql_dialect:
            self.sql_dialect = "postgres"
        self.llm_provider = self.llm_provider.lower()
        self.embed_provider = self.embed_provider.lower()
        if not self.llm_model:
            self.llm_model = DEFAULT_LLM_MODELS.get(self.llm_provider, "")
        if not self.embed_model:
            self.embed_model = DEFAULT_EMBED_MODELS.get(self.embed_provider, "")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("DOCQA_CHUNK_OVERLAP must be smaller than DOCQA_CHUNK_SIZE")
        if self.code_chunk_overlap >= self.code_chunk_size:
            raise ValueError("DOCQA_CODE_CHUNK_OVERLAP must be smaller than DOCQA_CODE_CHUNK_SIZE")
        self.sql_db_router = self.sql_db_router.lower()
        if self.sql_db_router not in ("llm", "retrieval"):
            raise ValueError("DOCQA_DB_ROUTER must be 'llm' or 'retrieval'")
        if self.sql_max_attempts < 1:
            raise ValueError("DOCQA_SQL_MAX_ATTEMPTS must be at least 1")

    @classmethod
    def from_env(cls, env_file: str | os.PathLike | None = ".env") -> "Settings":
        if load_dotenv and env_file and Path(env_file).exists():
            load_dotenv(env_file, override=False)
        return cls(
            docs_dir=Path(_env("DOCQA_DOCS_DIR", "./my_files")),
            data_dir=Path(_env("DOCQA_DATA_DIR", "./data")),
            llm_provider=_env("DOCQA_LLM_PROVIDER", "ollama"),
            llm_model=_env("DOCQA_LLM_MODEL", ""),
            embed_provider=_env("DOCQA_EMBED_PROVIDER", "ollama"),
            embed_model=_env("DOCQA_EMBED_MODEL", ""),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL") or None,
            chunk_size=int(_env("DOCQA_CHUNK_SIZE", "800")),
            chunk_overlap=int(_env("DOCQA_CHUNK_OVERLAP", "150")),
            top_k=int(_env("DOCQA_TOP_K", "4")),
            use_docling=_flag("DOCQA_USE_DOCLING", "0"),
            num_ctx=int(_env("DOCQA_NUM_CTX", "8192")),
            include_code=_flag("DOCQA_INCLUDE_CODE", "1"),
            code_chunk_size=int(_env("DOCQA_CODE_CHUNK_SIZE", "1500")),
            code_chunk_overlap=int(_env("DOCQA_CODE_CHUNK_OVERLAP", "150")),
            max_file_kb=int(_env("DOCQA_MAX_FILE_KB", "1024")),
            database_url=os.getenv("DOCQA_DATABASE_URL") or None,
            database_dir=Path(os.environ["DOCQA_DATABASE_DIR"]) if os.getenv("DOCQA_DATABASE_DIR") else None,
            sql_db_router=_env("DOCQA_DB_ROUTER", "llm"),
            sql_db_candidates=int(_env("DOCQA_DB_CANDIDATES", "5")),
            sql_route_margin=float(_env("DOCQA_ROUTE_MARGIN", "0.25")),
            sql_route_choices=int(_env("DOCQA_ROUTE_CHOICES", "2")),
            sql_dialect=_env("DOCQA_SQL_DIALECT", "").lower(),
            sql_tables=tuple(t.strip() for t in _env("DOCQA_SQL_TABLES", "").split(",") if t.strip()),
            sql_max_attempts=int(_env("DOCQA_SQL_MAX_ATTEMPTS", "3")),
            sql_max_rows=int(_env("DOCQA_SQL_MAX_ROWS", "50")),
            sql_schema_linking=_flag("DOCQA_SQL_SCHEMA_LINKING", "1"),
            sql_value_lookup=_flag("DOCQA_SQL_VALUE_LOOKUP", "1"),
            sql_patterns=_flag("DOCQA_SQL_PATTERNS", "1"),
            sql_answer_checks=_flag("DOCQA_SQL_CHECKS", "1"),
            sql_review=_flag("DOCQA_SQL_REVIEW", "0"),
            sql_patterns_dir=Path(os.environ["DOCQA_SQL_PATTERNS_DIR"]) if os.getenv("DOCQA_SQL_PATTERNS_DIR") else None,
            sql_candidates=int(_env("DOCQA_SQL_CANDIDATES", "1")),
            sql_max_tables=int(_env("DOCQA_SQL_MAX_TABLES", "20")),
            hybrid_search=_flag("DOCQA_HYBRID_SEARCH", "1"),
            sql_schema_chars=int(_env("DOCQA_SQL_SCHEMA_CHARS", "12000")),
        )

    @property
    def collection_name(self) -> str:
        """One collection per embedding model *and* docs folder.

        Vectors from different embedding models are not comparable (and often have
        different dimensions), so switching models gets a fresh collection and a
        fresh record-manager namespace instead of silently mixing them. The docs
        folder is part of the key too: the sync deletes every chunk it did not see,
        so two folders sharing one collection would wipe each other's chunks.

        Chroma names must end in a letter or digit, so the readable slug is cut
        short and a hash of the full key keeps long model names distinct.
        """
        key = f"{self.embed_provider}:{self.embed_model}:{self.docs_dir.resolve().as_posix()}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:10]
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", f"{self.embed_provider}-{self.embed_model}").strip("-")[:40].strip("-")
        return f"docs-{slug}-{digest}"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def record_manager_url(self) -> str:
        return f"sqlite:///{(self.data_dir / 'record_manager.sqlite').as_posix()}"
