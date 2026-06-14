"""Run both approaches over the held-out set and print a side-by-side comparison of
accuracy, latency, throughput and compute cost.

Run:  python -m src.eval.run_eval            # both approaches
      python -m src.eval.run_eval --only encoder
      python -m src.eval.run_eval --only llm

Same eval set and same metrics for both, so the fine-tune vs. prompt call is measured
rather than guessed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.eval.metrics import accuracy, cost_per_1k, macro_f1, percentile

load_dotenv()
console = Console()

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_MD = Path(__file__).resolve().parents[2] / "docs" / "results.md"


def _load_eval() -> list[dict]:
    path = DATA_DIR / "eval_set.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _run_approach(name: str, classify, rows: list[dict]) -> dict:
    y_true, y_pred, latencies = [], [], []
    with console.status(f"Running {name} over {len(rows)} examples..."):
        for row in rows:
            pred = classify(row["text"])
            y_true.append(row["label"])
            y_pred.append(pred.intent)
            latencies.append(pred.latency_ms or 0.0)

    usd_per_hour = float(
        os.getenv("ENCODER_HOST_USD_PER_HOUR" if name == "encoder" else "LLM_HOST_USD_PER_HOUR", "0.10")
    )
    avg_ms = sum(latencies) / len(latencies)
    return {
        "name": name,
        "accuracy": accuracy(y_true, y_pred),
        "f1_macro": macro_f1(y_true, y_pred),
        "p50_ms": percentile(latencies, 50),
        "p95_ms": percentile(latencies, 95),
        "throughput_rps": 1000.0 / avg_ms if avg_ms else 0.0,
        "cost_per_1k_usd": cost_per_1k(latencies, usd_per_hour),
        "n": len(rows),
    }


def _print_table(results: list[dict]) -> None:
    table = Table(title="Intent Router - Approach Comparison", header_style="bold")
    table.add_column("Approach")
    table.add_column("Accuracy", justify="right")
    table.add_column("Macro-F1", justify="right")
    table.add_column("p50 (ms)", justify="right")
    table.add_column("p95 (ms)", justify="right")
    table.add_column("Throughput (req/s)", justify="right")
    table.add_column("Cost / 1k ($)", justify="right")
    for r in results:
        table.add_row(
            r["name"],
            f"{r['accuracy']:.1%}",
            f"{r['f1_macro']:.3f}",
            f"{r['p50_ms']:.1f}",
            f"{r['p95_ms']:.1f}",
            f"{r['throughput_rps']:.1f}",
            f"{r['cost_per_1k_usd']:.4f}",
        )
    console.print(table)


def _write_markdown(results: list[dict]) -> None:
    lines = [
        "# Eval Results",
        "",
        f"Held-out eval set: {results[0]['n']} bilingual (AR/EN) examples.",
        "",
        "| Approach | Accuracy | Macro-F1 | p50 (ms) | p95 (ms) | Throughput (req/s) | Cost / 1k ($) |",
        "|----------|---------:|---------:|---------:|---------:|-------------------:|--------------:|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['accuracy']:.1%} | {r['f1_macro']:.3f} | "
            f"{r['p50_ms']:.1f} | {r['p95_ms']:.1f} | {r['throughput_rps']:.1f} | "
            f"{r['cost_per_1k_usd']:.4f} |"
        )
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    console.print(f"[green]Wrote[/green] {RESULTS_MD}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["encoder", "llm"], help="run a single approach")
    args = parser.parse_args()

    rows = _load_eval()
    results: list[dict] = []

    if args.only in (None, "encoder"):
        try:
            from src.encoder_classifier.predict import classify as encoder_classify

            results.append(_run_approach("encoder", encoder_classify, rows))
        except Exception as e:  # noqa: BLE001 - surface as a skipped row, don't crash the run
            console.print(f"[yellow]Skipped encoder:[/yellow] {e}")

    if args.only in (None, "llm"):
        try:
            from src.llm_classifier.classifier import classify as llm_classify

            results.append(_run_approach("llm", llm_classify, rows))
        except Exception as e:  # noqa: BLE001
            console.print(f"[yellow]Skipped llm:[/yellow] {e}")

    if not results:
        console.print("[red]No approach ran. Train the encoder and/or start Ollama first.[/red]")
        return

    _print_table(results)
    _write_markdown(results)


if __name__ == "__main__":
    main()
