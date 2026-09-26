"""Pick a classifier by name: the one place that maps "encoder" / "llm" to code.

demo.py, the API and the eval harness all call `get(method)` and receive a
`classify(text) -> Prediction` function. Imports are lazy, so asking for the LLM never
loads torch/transformers and asking for the encoder never needs the OpenAI client.
"""

from __future__ import annotations

from collections.abc import Callable

from src.schema import Prediction

METHODS: tuple[str, ...] = ("encoder", "llm")


def get(method: str) -> Callable[[str], Prediction]:
    """Return the classify function for `method` ("encoder" or "llm")."""
    if method == "encoder":
        from src.encoder_classifier.predict import classify

        return classify
    if method == "llm":
        from src.llm_classifier.classifier import classify

        return classify
    raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")
