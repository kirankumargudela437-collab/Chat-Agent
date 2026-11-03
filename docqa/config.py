"""Settings, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
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
    extensions: tuple[str, ...] = field(default=(".pdf", ".docx", ".md", ".markdown", ".txt"))

    def __post_init__(self) -> None:
        self.llm_provider = self.llm_provider.lower()
        self.embed_provider = self.embed_provider.lower()
        if not self.llm_model:
            self.llm_model = DEFAULT_LLM_MODELS.get(self.llm_provider, "")
        if not self.embed_model:
            self.embed_model = DEFAULT_EMBED_MODELS.get(self.embed_provider, "")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("DOCQA_CHUNK_OVERLAP must be smaller than DOCQA_CHUNK_SIZE")

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
            use_docling=_env("DOCQA_USE_DOCLING", "0").lower() in ("1", "true", "yes"),
        )

    @property
    def collection_name(self) -> str:
        """One collection per embedding model.

        Vectors from different embedding models are not comparable (and often have
        different dimensions), so switching models gets a fresh collection and a
        fresh record-manager namespace instead of silently mixing them.
        """
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", f"{self.embed_provider}-{self.embed_model}").strip("-")
        return f"docs-{slug}"[:63]

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def record_manager_url(self) -> str:
        return f"sqlite:///{(self.data_dir / 'record_manager.sqlite').as_posix()}"
