"""Approach A inference: load the fine-tuned encoder and classify text.

The model is loaded once and cached, so repeated calls (eval loop, API) pay the load
cost only once.
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.schema import Prediction

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "encoder"


@lru_cache(maxsize=1)
def _load():
    if not (MODEL_DIR / "config.json").exists():
        raise FileNotFoundError(
            f"No trained encoder at {MODEL_DIR}. Run: python -m src.encoder_classifier.train"
        )
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    # low_cpu_mem_usage=False forces a full load onto CPU. With accelerate installed,
    # the default can lazy-load weights onto the "meta" device, which then errors at
    # inference ("Tensor on device meta is not on the expected device cpu").
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), low_cpu_mem_usage=False, torch_dtype=torch.float32
    )
    model.to("cpu")
    model.eval()
    return tokenizer, model


def classify(text: str) -> Prediction:
    tokenizer, model = _load()
    start = time.perf_counter()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    idx = int(torch.argmax(probs))
    latency_ms = (time.perf_counter() - start) * 1000
    return Prediction(
        intent=model.config.id2label[idx],
        confidence=float(probs[idx]),
        latency_ms=latency_ms,
        method="encoder",
    )


if __name__ == "__main__":
    import sys

    text = " ".join(sys.argv[1:]) or "I want to cancel my subscription"
    print(classify(text))
