"""Quick CLI demo: classify a single message with either approach.

    python demo.py "I want to cancel my subscription"
    python demo.py "عايز ألغي اشتراكي" --method llm
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from src.schema import Prediction


def run(text: str, method: str) -> Prediction:
    if method == "encoder":
        from src.encoder_classifier.predict import classify
    else:
        from src.llm_classifier.classifier import classify
    return classify(text)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Classify a support message.")
    parser.add_argument("text", help="the message to classify")
    parser.add_argument("--method", choices=["encoder", "llm"], default="encoder")
    args = parser.parse_args()

    pred = run(args.text, args.method)
    conf = f"{pred.confidence:.1%}" if pred.confidence is not None else "n/a"
    print(f"text     : {args.text}")
    print(f"method   : {pred.method}")
    print(f"intent   : {pred.intent}")
    print(f"confidence: {conf}")
    print(f"latency  : {pred.latency_ms:.1f} ms")


if __name__ == "__main__":
    main()
