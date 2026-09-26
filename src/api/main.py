"""FastAPI service exposing both classifiers behind one /classify endpoint.

POST /classify  { "text": "...", "method": "encoder" | "llm" }
-> { "intent", "confidence", "latency_ms", "method" }

The .NET Minimal API in dotnet/ calls this service. Models load lazily on first
use of each method, so the API starts instantly even before the encoder is trained.
"""

from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import classifiers
from src.schema import LABELS

load_dotenv()

app = FastAPI(title="Intent Router", version="0.1.0")


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Message to classify")
    method: str = Field("encoder", pattern="^(encoder|llm)$")


class ClassifyResponse(BaseModel):
    intent: str
    confidence: float | None = None
    latency_ms: float | None = None
    method: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "labels": LABELS}


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    try:
        pred = classifiers.get(req.method)(req.text)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001 - backend (e.g. Ollama) unreachable
        raise HTTPException(status_code=502, detail=f"{req.method} backend error: {e}") from e

    return ClassifyResponse(
        intent=pred.intent,
        confidence=pred.confidence,
        latency_ms=pred.latency_ms,
        method=pred.method or req.method,
    )
