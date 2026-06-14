"""Approach A: fine-tune a small multilingual encoder for intent classification.

Model: distilbert-base-multilingual-cased (~134M params). Small enough to train on
CPU in a few minutes on this dataset, and multilingual so it handles AR and EN.

Run:  python -m src.encoder_classifier.train
Saves the model, tokenizer and label map to models/encoder/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.schema import LABELS

MODEL_NAME = "distilbert-base-multilingual-cased"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "models" / "encoder"

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def _load_jsonl(path: Path) -> Dataset:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return Dataset.from_list(
        [{"text": r["text"], "label": LABEL2ID[r["label"]]} for r in rows]
    )


def _metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main() -> None:
    train_ds = _load_jsonl(DATA_DIR / "intents.jsonl")
    eval_ds = _load_jsonl(DATA_DIR / "eval_set.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

    train_ds = train_ds.map(tokenize, batched=True)
    eval_ds = eval_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "_checkpoints"),
        num_train_epochs=20,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=20,
        report_to=[],
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=_metrics,
    )

    trainer.train()
    print("Eval:", trainer.evaluate())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    (OUTPUT_DIR / "labels.json").write_text(json.dumps(LABELS, ensure_ascii=False), encoding="utf-8")
    print(f"Saved encoder model -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
