"""Approach B: classify with a self-hosted LLM over an OpenAI-compatible endpoint.

The OpenAI client points at Ollama's /v1 endpoint. Since it's the OpenAI wire format,
the same code runs against a vLLM server too; only OLLAMA_HOST and LLM_MODEL change.

We ask for a JSON-schema-constrained response. If the backend ignores the schema we
still parse the JSON and validate the label, so a chatty model won't break the caller.
"""

from __future__ import annotations

import json
import os
import re
import time
from functools import lru_cache

from openai import OpenAI

from src.llm_classifier.prompts import build_messages
from src.schema import LABELS, Prediction, json_schema


@lru_cache(maxsize=1)
def _client() -> tuple[OpenAI, str]:
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
    timeout = float(os.getenv("LLM_TIMEOUT_S", "120"))
    # api_key is required by the client but unused by local servers.
    return OpenAI(base_url=f"{host}/v1", api_key="not-needed", timeout=timeout), model


def _extract_label(raw: str) -> str:
    """Pull a valid intent label out of the model's reply, defensively."""
    try:
        obj = json.loads(raw)
        candidate = str(obj.get("intent", "")).strip().lower()
    except (json.JSONDecodeError, AttributeError):
        # Fall back: find the first label that appears anywhere in the text.
        candidate = ""
        match = re.search(r'"intent"\s*:\s*"([^"]+)"', raw)
        if match:
            candidate = match.group(1).strip().lower()
    if candidate in LABELS:
        return candidate
    for label in LABELS:  # last resort: substring match
        if label in raw.lower():
            return label
    return "complaint"  # safest catch-all for an unparseable support message


def classify(text: str) -> Prediction:
    client, model = _client()
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=build_messages(text),
        temperature=0,
        response_format={"type": "json_schema", "json_schema": json_schema()},
    )
    latency_ms = (time.perf_counter() - start) * 1000
    raw = response.choices[0].message.content or ""
    return Prediction(
        intent=_extract_label(raw),
        confidence=None,  # local chat models don't expose a calibrated probability here
        latency_ms=latency_ms,
        method="llm",
    )
