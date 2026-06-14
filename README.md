# Intent Router

Classifies short customer-support messages (Arabic and English) into one of eight
intents. I built it two ways on purpose: a fine-tuned multilingual encoder and a
self-hosted LLM with a constrained JSON output, then wrote an eval harness to compare
them on accuracy, latency, throughput and cost. The interesting part isn't either
model, it's the comparison and what it says about when to fine-tune vs. when to prompt.

## Problem

A support queue needs every incoming message routed to the right intent
(`billing_issue`, `cancel_subscription`, `refund_request`, and so on). The usual
question is whether to train a small classifier or just prompt an LLM. Most of the time
that gets answered by gut feel. Here I answered it by running both on the same labelled
set and measuring. Input is bilingual, and the Arabic side is treated as a first-class
case, not an afterthought.

## Architecture

```
  client ──HTTP──▶ .NET 8 Minimal API ──HTTP──▶ FastAPI /classify
                   (validation, typed                  │
                    HttpClient)                         ├─▶ A: fine-tuned DistilBERT
                                                        │      (multilingual)
                                                        └─▶ B: self-hosted LLM
                                                               (Ollama / vLLM,
                                                                JSON-schema output)
                                                  │
                                                  ▼
                                          eval harness ─▶ docs/results.md
                                    (accuracy, p50/p95, throughput, cost/1k)
```

- **A** fine-tunes `distilbert-base-multilingual-cased` on the labelled data.
- **B** calls a local model (`qwen2.5:7b-instruct`, or `qwen2.5:14b` on a GPU) through
  an OpenAI-compatible endpoint, so the same code runs unchanged against vLLM later.
- The .NET 8 Minimal API sits in front of the Python service and does request
  validation and forwarding through a typed `HttpClient`.

The reasoning behind fine-tune vs. prompt is written up in [`docs/DESIGN.md`](docs/DESIGN.md),
and the code conventions the repo follows are in [`CONVENTIONS.md`](CONVENTIONS.md).

## Results

Held-out set of 32 bilingual examples. Encoder is the fine-tuned DistilBERT on CPU; the
LLM is `qwen2.5:7b-instruct` through Ollama, also on CPU (this box has no GPU).

| Approach | Accuracy | Macro-F1 | p50 (ms) | p95 (ms) | Throughput (req/s) | Cost / 1k ($) |
|----------|---------:|---------:|---------:|---------:|-------------------:|--------------:|
| encoder  |    71.9% |    0.725 |     27.5 |     31.6 |               23.7 |        0.0006 |
| llm      |    93.8% |    0.933 |   3592.9 |   4429.8 |                0.2 |        1.0757 |

A few things stand out:

- The LLM is more accurate with no training at all (93.8% vs 71.9%). On a small set
  there just isn't enough data for the encoder to learn what the LLM already knows.
- The encoder is roughly 130x faster and far cheaper per request (27 ms vs 3.6 s,
  $0.0006 vs $1.08 per 1k). On CPU the 7B is slow; a GPU narrows the gap but the encoder
  still wins on cost.
- Practical read: prompt to get something shipped and while data is scarce or the label
  set keeps changing; move to a fine-tuned encoder once volume is high and latency/cost
  start to matter. `docs/DESIGN.md` goes into this.

Regenerate the numbers any time with `python -m src.eval.run_eval`, which rewrites
[`docs/results.md`](docs/results.md).

## Running it

```bash
python -m venv .venv
. .venv/Scripts/activate          # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env              # edit LLM_MODEL if you want a different model

python data/generate.py           # build the bilingual dataset
python -m src.encoder_classifier.train   # train approach A

ollama pull qwen2.5:7b-instruct   # for approach B

python -m src.eval.run_eval       # compare both, writes docs/results.md

python demo.py "I want to cancel my subscription"
python demo.py "عايز ألغي اشتراكي" --method llm
```

Serving:

```bash
python -m uvicorn src.api.main:app --port 8000   # http://127.0.0.1:8000/docs
cd dotnet/IntentRouter.Api && dotnet run          # forwards to the FastAPI service
```

Use `python -m uvicorn` (not bare `uvicorn`) so it runs under the same interpreter that
has the dependencies installed.

Example call through the .NET layer:

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text":"اتخصم مني المبلغ مرتين","method":"encoder"}'
```

> On Windows, use `127.0.0.1` rather than `localhost`. `localhost` can resolve to IPv6
> `::1` while Ollama and uvicorn bind `127.0.0.1`, which shows up as a hang.

## Tech

| Layer | Choice | Why |
|-------|--------|-----|
| Encoder | DistilBERT multilingual (HF `transformers`) | small, trains on CPU, handles AR and EN |
| LLM | `qwen2.5:7b-instruct` via Ollama (OpenAI-compatible) | multilingual, swappable for vLLM/14b later |
| Structured output | JSON-schema constraint + a validation fallback in code | keeps a chat model's output to valid labels |
| API | FastAPI + .NET 8 Minimal API | ML in Python, public contract in .NET |
| Eval | scikit-learn plus a small cost model | the actual comparison |

## Layout

```
data/        dataset and a deterministic generator
src/
  schema.py            labels and the prediction type, in one place
  encoder_classifier/  approach A: train.py, predict.py
  llm_classifier/      approach B: classifier.py, prompts.py
  eval/                run_eval.py, metrics.py
  api/main.py          FastAPI /classify
dotnet/IntentRouter.Api/   .NET 8 Minimal API wrapper
docs/        DESIGN.md and the generated results.md
demo.py      one-shot CLI
```
