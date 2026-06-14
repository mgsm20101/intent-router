# Fine-tune vs. prompt

Notes on why this repo ships two classifiers instead of one, and how I'd pick between
them in a real project.

## The setup

Intent classification on a small, bilingual (Arabic/English) support queue with a
fixed set of eight intents. Two reasonable approaches:

| | A: fine-tuned encoder | B: self-hosted LLM + prompt |
|---|---|---|
| Model | `distilbert-base-multilingual-cased` (~134M) | `qwen2.5:7b-instruct` via Ollama (14b on GPU) |
| How it learns | gradient updates on labelled data | instruction plus few-shot, weights frozen |
| Adding an intent | collect data, retrain | edit the prompt |
| Output validity | always a valid class id | constrained by JSON schema, validated in code |
| Footprint | tiny, runs on CPU | needs a GPU to be fast |

## How I decide

1. Data volume. Fine-tuning pays off when there are enough labelled examples, hundreds
   to thousands per class. With a handful per class, a strong instruction-tuned model
   with few-shot usually does better, because there's little for the encoder to learn
   that the LLM doesn't already know.
2. Label churn. If the intents change often, a prompt is cheaper to iterate on: you edit
   text instead of retraining and re-validating a model.
3. Latency and cost at scale. Once data is plentiful and the taxonomy is stable, the
   small encoder is much cheaper and faster per request. That's when fine-tuning wins.
4. Output guarantees. The encoder can only emit a known class. The LLM needs a
   structured-output schema and a validation/fallback layer (see
   `src/llm_classifier/classifier.py`, `_extract_label`) before it's safe in production.

## What the numbers said

On this 96-example training set the LLM was more accurate than the fine-tuned encoder
(93.8% vs 71.9%) with no training, which lines up with point 1 above: not enough data
for the encoder yet. The encoder was about 130x faster and far cheaper per request
(27 ms vs 3.6 s, $0.0006 vs $1.08 per 1k), which is point 3.

So there's no single winner, which is the whole reason for shipping a harness rather
than an opinion. Cold start, scarce data or a shifting taxonomy points to the prompt.
High volume, stable labels and a tight latency or cost budget points to the encoder.

A common production path that gets both: start with the prompt to ship, log the
traffic, then distil that traffic into a fine-tuned encoder once the volume justifies it.
