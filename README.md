# docqa: chat with a folder of changing documents

A Q&A chat agent over PDF, DOCX, Markdown and text files. Ingestion and answering
are separate: `docqa sync` (or `docqa watch`) keeps a local vector index in step
with the folder, and `docqa chat` answers from that index with citations.

Everything runs locally by default (Ollama + Chroma). Switching the chat model or
the embeddings to a cloud API is one line in `.env`; your files and the vector DB
stay on your machine either way, and in cloud mode only the retrieved snippets are
sent to the API.

## Quick start (local, Ollama)

```bash
# 1. Ollama (https://ollama.com/download), then:
ollama pull llama3.1
ollama pull nomic-embed-text

# 2. Install
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .

# 3. Point it at your files
cp .env.example .env          # set DOCQA_DOCS_DIR=/path/to/your/files
docqa sync                    # first run embeds everything; later runs only changes
docqa chat                    # interactive, remembers follow-ups
```

One-off question: `docqa ask "What is the refund policy?"`.
Keep the index fresh while you edit files: `docqa watch --interval 10`.
Inside chat: `/sync` re-indexes, `/reset` clears history, `/quit` exits.
`docqa -v chat` also prints how each follow-up was rewritten before searching.

Example:

```
you> What is the refund policy?
bot> Customers may request a refund within 30 days of purchase [1].
Sources:
  [1] policies/refunds.pdf, page 1

you> How do I apply for it?
bot> Email billing@example.com with your order number [2].
Sources:
  [2] policies/refunds.pdf, page 2
```

## Using a cloud model instead

```bash
pip install -e ".[openai]"     # or .[anthropic] / .[google]
```

```ini
# .env
DOCQA_LLM_PROVIDER=openai      # openai | anthropic | google | ollama
DOCQA_LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...

# Embeddings can stay local (no document text leaves your machine during sync)
DOCQA_EMBED_PROVIDER=ollama
```

Anthropic has no embedding model, so pair `DOCQA_LLM_PROVIDER=anthropic` with
Ollama or OpenAI embeddings. Each embedding model gets its own collection, so
switching `DOCQA_EMBED_*` re-embeds into a fresh index rather than mixing vectors.

## How it handles the five "missing pieces"

| Problem | What docqa does | Where |
|---|---|---|
| Changing files | LangChain indexing API with a SQLite record manager and `cleanup="full"`: unchanged chunks are skipped, edited files are replaced, deleted files are removed. | `docqa/indexing.py` |
| Multi-format parsing | pypdf (per page), docx2txt, Markdown split by headings. Optional Docling (`DOCQA_USE_DOCLING=1`, `pip install -e ".[docling]"`) for tables and scanned PDFs. | `docqa/loaders.py` |
| Chunking with overlap | `RecursiveCharacterTextSplitter`, 800 chars / 150 overlap by default (`DOCQA_CHUNK_SIZE`, `DOCQA_CHUNK_OVERLAP`). | `docqa/loaders.py` |
| Follow-up questions | With history, the LLM first rewrites the question into a standalone one, and that is what gets searched. | `docqa/chain.py` |
| Grounding and citations | Context is numbered and labelled with file, page and section; the prompt allows only that context, requires `[n]` citations, and has a fixed "not found" reply. The app prints the sources the answer actually cited. | `docqa/chain.py` |

## Changes from the original blueprint

* `format_docs` dropped all metadata, so the model could not cite anything. Each
  passage now carries `[n] (source: file, page N, section "...")`.
* `from langchain.indexes import SQLRecordManager` fails on LangChain 1.x; it now
  lives in `langchain_classic.indexes` and needs `sqlalchemy[asyncio]`.
* `source` is the path relative to the docs folder, and `page` is 1-based (pypdf
  counts from 0).
* Loader metadata such as PDF producer and dates is dropped. The record manager
  hashes metadata too, so volatile fields would make every sync re-embed.
* Empty pages, Office lock files (`~$...`), hidden files and unreadable files are
  skipped and reported instead of crashing the sync.
* If nothing is retrieved, the LLM is not called at all.
* Chunk hashes use SHA-256 instead of the default SHA-1.

## Limits worth knowing

* Scanned PDFs have no text layer; sync warns about them. Use Docling for OCR.
* In Docling mode citations show file and section but not page numbers.
* DOCX files have no page numbers to cite (Word computes pages at render time).
* `cleanup="full"` means a file that fails to parse is removed from the index
  until it parses again; sync lists it under `error`.
* Small local models follow the citation rules less reliably than large ones.
  `qwen2.5:7b` or `llama3.1:8b` are a reasonable floor.

## Using it from Python

```python
from docqa import Settings, build_agent, sync_documents

s = Settings.from_env()
print(sync_documents(s).summary())
agent = build_agent(s)
ans = agent.ask("What is the refund policy?")
print(ans.answer, ans.sources_text(), sep="\n")
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Tests use fake embeddings and a scripted chat model, so they need neither Ollama
nor an API key. They cover loading all formats with citation metadata, overlap,
incremental add/skip/edit/delete, persistence across runs, follow-up rewriting,
grounded prompts, citations and the CLI.
