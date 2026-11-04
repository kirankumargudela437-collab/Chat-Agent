"""A record of what the model saw and did on every call, for diagnosing SQL failures.

Set ``DOCQA_TRACE_FILE=traces.jsonl`` (or pass ``tracer=CallTracer(...)`` to ``SQLAgent``)
and every model call of the SQL agent is appended as one JSON line:

* ``type: run``: the settings once per process (model, temperature, seed, context window,
  output-token limit, agent version);
* ``type: call``: the step (``plan`` / ``generate`` / ``repair`` / ``review`` / ``summary``),
  candidate and attempt, the exact messages sent, the raw reply before SQL extraction,
  prompt and output token counts, the stop reason (``stop`` or ``length`` = cut off),
  seconds, and a repetition score (share of repeated 6-word runs; a loop scores high);
* ``type: check``: the SQL extracted from a reply and whether it passed (or the error).

A long reply or a timeout alone does not prove a loop; the repetition score and the
stop reason are recorded so a reviewer can tell repetition from truncation.
"""

from __future__ import annotations

import contextvars
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

_step = contextvars.ContextVar("docqa_trace_step", default="")
_info = contextvars.ContextVar("docqa_trace_info", default={})


def set_step(step: str, **info: Any) -> None:
    """Label the next model calls (``plan``, ``generate``...) with ``info`` (attempt, candidate)."""
    _step.set(step)
    _info.set({**_info.get(), **info})


def set_question(question: str, **info: Any) -> None:
    _info.set({"question": question, "trace_id": uuid.uuid4().hex[:12], **info})


def repetition(text: str, n: int = 6) -> float:
    """Share of ``n``-word runs that already occurred earlier in the text (0 = none, ~1 = a loop)."""
    words = re.findall(r"\S+", text)
    if len(words) < 4 * n:
        return 0.0
    seen: set[tuple] = set()
    repeats = 0
    for i in range(len(words) - n + 1):
        gram = tuple(words[i:i + n])
        if gram in seen:
            repeats += 1
        seen.add(gram)
    return round(repeats / (len(words) - n + 1), 3)


def _messages(batch) -> list[dict]:
    out = []
    for m in batch:
        content = m.content if isinstance(m.content, str) else json.dumps(m.content, ensure_ascii=False)
        out.append({"role": getattr(m, "type", "message"), "content": content})
    return out


class CallTracer(BaseCallbackHandler):
    """Appends one JSON line per model call (see the module docstring)."""

    def __init__(self, path: str | Path, settings: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._open: dict = {}
        self.write({"type": "run", "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **(settings or {})})

    def write(self, record: dict) -> None:
        record.setdefault("time", time.strftime("%Y-%m-%dT%H:%M:%S"))  # every line, so a run can be audited
        with self._lock, open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def event(self, kind: str, **data: Any) -> None:
        self.write({"type": kind, "step": _step.get(), **_info.get(), **data})

    # -- callbacks ------------------------------------------------------------------

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._open[run_id] = (time.monotonic(), _messages(messages[0]) if messages else [], _step.get(), dict(_info.get()))

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):  # plain completion models
        self._open[run_id] = (time.monotonic(), [{"role": "prompt", "content": p} for p in prompts],
                              _step.get(), dict(_info.get()))

    def on_llm_end(self, response, *, run_id, **kwargs):
        start, messages, step, info = self._open.pop(run_id, (time.monotonic(), [], _step.get(), dict(_info.get())))
        gen = response.generations[0][0] if response.generations and response.generations[0] else None
        text = gen.text if gen is not None else ""
        meta: dict = {}
        msg = getattr(gen, "message", None)
        if msg is not None:
            meta = dict(getattr(msg, "response_metadata", {}) or {})
            usage = getattr(msg, "usage_metadata", None) or {}
        else:
            usage = {}
        meta.update(getattr(gen, "generation_info", None) or {})
        self.write({
            "type": "call",
            "step": step,
            **info,
            "seconds": round(time.monotonic() - start, 2),
            "model": meta.get("model") or meta.get("model_name"),
            "prompt_tokens": meta.get("prompt_eval_count") or usage.get("input_tokens"),
            "output_tokens": meta.get("eval_count") or usage.get("output_tokens"),
            "stop_reason": meta.get("done_reason") or meta.get("finish_reason") or meta.get("stop_reason"),
            "prompt_chars": sum(len(m["content"]) for m in messages),
            "output_chars": len(text),
            "repetition": repetition(text),
            "messages": messages,
            "raw": text,
        })

    def on_llm_error(self, error, *, run_id, **kwargs):
        start, messages, step, info = self._open.pop(run_id, (time.monotonic(), [], _step.get(), dict(_info.get())))
        self.write({"type": "call", "step": step, **info, "seconds": round(time.monotonic() - start, 2),
                    "error": f"{type(error).__name__}: {error}", "messages": messages, "raw": ""})


def tracer_from_env(settings: dict | None = None) -> CallTracer | None:
    path = os.getenv("DOCQA_TRACE_FILE")
    return CallTracer(path, settings) if path else None
