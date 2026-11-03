"""Factories for the chat model and the embedding model.

Imports are lazy so you only need the package for the provider you actually use.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .config import Settings


class ProviderError(RuntimeError):
    pass


def _missing(pkg: str, extra: str) -> ProviderError:
    return ProviderError(f"{pkg} is not installed. Run: pip install -e \".[{extra}]\"")


def make_llm(s: Settings) -> BaseChatModel:
    p = s.llm_provider
    if p == "ollama":
        from langchain_ollama import ChatOllama

        kwargs = {"base_url": s.ollama_base_url} if s.ollama_base_url else {}
        return ChatOllama(model=s.llm_model, temperature=s.temperature, **kwargs)
    if p == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise _missing("langchain-openai", "openai") from e
        return ChatOpenAI(model=s.llm_model, temperature=s.temperature)
    if p == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise _missing("langchain-anthropic", "anthropic") from e
        return ChatAnthropic(model=s.llm_model, temperature=s.temperature)
    if p == "google":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise _missing("langchain-google-genai", "google") from e
        return ChatGoogleGenerativeAI(model=s.llm_model, temperature=s.temperature)
    raise ProviderError(f"Unknown DOCQA_LLM_PROVIDER '{p}'. Use ollama, openai, anthropic or google.")


def make_embeddings(s: Settings) -> Embeddings:
    p = s.embed_provider
    if p == "ollama":
        from langchain_ollama import OllamaEmbeddings

        kwargs = {"base_url": s.ollama_base_url} if s.ollama_base_url else {}
        return OllamaEmbeddings(model=s.embed_model, **kwargs)
    if p == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as e:
            raise _missing("langchain-openai", "openai") from e
        return OpenAIEmbeddings(model=s.embed_model)
    if p == "google":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as e:
            raise _missing("langchain-google-genai", "google") from e
        return GoogleGenerativeAIEmbeddings(model=s.embed_model)
    raise ProviderError(
        f"Unknown DOCQA_EMBED_PROVIDER '{p}'. Use ollama, openai or google "
        "(Anthropic has no embedding model; pair it with ollama or openai embeddings)."
    )
