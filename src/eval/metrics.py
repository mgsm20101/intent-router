"""Pure metric helpers for the eval harness. No I/O, easy to unit-test."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    return float(accuracy_score(y_true, y_pred))


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def percentile(values: list[float], p: float) -> float:
    """p in [0,100]. Linear interpolation; no numpy dependency needed here."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def cost_per_1k(latencies_ms: list[float], usd_per_hour: float) -> float:
    """Compute-based cost: avg seconds/request * ($/hour / 3600) * 1000 requests.

    Both approaches are self-hosted, so cost is driven by how long the serving box
    is busy per request, not by per-token API pricing.
    """
    if not latencies_ms:
        return 0.0
    avg_seconds = (sum(latencies_ms) / len(latencies_ms)) / 1000.0
    return avg_seconds * (usd_per_hour / 3600.0) * 1000.0
