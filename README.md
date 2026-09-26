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

## Structure

### Two paths

**Serving** (one message in, one intent out, over HTTP):

```
client
  → .NET gateway :5080      dotnet/IntentRouter.Api/Program.cs   validation + typed HttpClient (optional)
  → FastAPI :8000           src/api/main.py : classify
  → classifiers.get(method) src/classifiers.py
  → encoder                 src/encoder_classifier/predict.py    reads models/encoder/
    or LLM                  src/llm_classifier/classifier.py     calls Ollama :11434
  → JSON {intent, confidence, latency_ms, method}
```

**Offline** (build the data, train, measure; no gateway, no FastAPI):

```
data/generate.py               → data/intents.jsonl, data/eval_set.jsonl
src/encoder_classifier/train.py → models/encoder/
python -m src.eval.run_eval    → classifiers.get → both classifiers in-process → docs/results.md
demo.py                        → classifiers.get → one prediction printed to the terminal
```

The .NET layer is an optional gateway: request validation and a typed `HttpClient` in
front of the Python model service, which is the common enterprise shape (public contract
in .NET, ML in Python). Nothing requires it. You can call FastAPI on :8000 directly, and
**the eval does not go through it** — `run_eval` imports the classifiers and calls them
in-process, so the published latencies contain no gateway or FastAPI overhead.

### Entry points

| Command | Reads | Writes |
|---------|-------|--------|
| `python -m src.eval.run_eval [--only encoder\|llm]` | `data/eval_set.jsonl`, `models/encoder/`, Ollama, cost rates from `.env` | `docs/results.md` (overwritten) + a table on the console |
| `python data/generate.py` | the utterance lists inside the script | `data/intents.jsonl`, `data/eval_set.jsonl` |
| `python -m src.encoder_classifier.train` | `data/intents.jsonl`, `data/eval_set.jsonl`, base model from the HF Hub | `models/encoder/` (git-ignored) |
| `python demo.py "text" [--method llm]` | `models/encoder/` or Ollama | the prediction on stdout |
| `python -m uvicorn src.api.main:app --port 8000` | `models/encoder/` or Ollama, on the first request per method | HTTP responses only |
| `cd dotnet/IntentRouter.Api && dotnet run` | `appsettings.json` (`PYTHON_API_BASE`), port from `Properties/launchSettings.json` | HTTP responses forwarded from :8000 |
| `python -m pytest` | `tests/` (no model, no Ollama) | nothing |

### Code map

```
src/schema.py                        the eight labels, the Prediction type, the LLM's JSON schema
src/classifiers.py                   get("encoder" | "llm") -> classify function; the one switch
src/api/main.py                      FastAPI: GET /health, POST /classify
src/encoder_classifier/train.py      approach A: fine-tune DistilBERT, save to models/encoder/
src/encoder_classifier/predict.py    approach A: load models/encoder/ once, classify
src/llm_classifier/classifier.py     approach B: call Ollama, parse and validate the label
src/llm_classifier/prompts.py        approach B: system prompt and few-shot examples
src/eval/run_eval.py                 run both over the eval set, print and write docs/results.md
src/eval/metrics.py                  accuracy, macro-F1, percentiles, modelled cost (pure functions)
src/__init__.py, src/*/__init__.py   empty package markers
data/generate.py                     deterministic bilingual dataset generator
data/intents.jsonl                   96 training examples (generated)
data/eval_set.jsonl                  32 held-out eval examples (generated)
demo.py                              one-shot CLI
dotnet/IntentRouter.Api/Program.cs   .NET 8 gateway: validate, forward to FastAPI
dotnet/IntentRouter.Api/IntentRouter.Api.csproj          project file
dotnet/IntentRouter.Api/appsettings.json                 upstream URL (PYTHON_API_BASE)
dotnet/IntentRouter.Api/Properties/launchSettings.json   gateway port 5080
tests/test_metrics.py                metric helpers
tests/test_extract_label.py          LLM reply parsing
tests/test_classifiers.py            classifiers.get, with the model modules stubbed
tests/test_api.py                    /classify responses and request validation
tests/__init__.py                    package marker
README.md                            this file
docs/results.md                      eval output (written by run_eval)
docs/DESIGN.md                       why two classifiers, and how to choose
CONVENTIONS.md                       code conventions the repo follows
.env.example                         Python settings: Ollama host, model, timeout, cost rates
requirements.txt                     full runtime dependencies
requirements-ci.txt                  test-only dependencies (no torch)
.github/workflows/tests.yml          CI: pytest on push and pull request
pyproject.toml                       ruff and black settings
.gitignore                           ignores .env, models/, build output
```

### Read the code in this order

1. `src/schema.py` — the contract everything else speaks.
2. `src/classifiers.py` — how a method name becomes a function.
3. `src/api/main.py` — the serving entry point.
4. `src/encoder_classifier/predict.py`, then `src/llm_classifier/classifier.py` and `prompts.py` — the two approaches.
5. `src/eval/run_eval.py` and `metrics.py` — how the numbers below are produced.
6. `dotnet/IntentRouter.Api/Program.cs` — the optional gateway.
7. `data/generate.py` and `src/encoder_classifier/train.py` — where the data and the trained model come from.

## Architecture

```
  Serving (HTTP)
  client ──▶ .NET 8 gateway :5080 ──▶ FastAPI :8000 /classify ──▶ classifiers.get(method)
             (optional: validation,                                  ├─▶ A: fine-tuned DistilBERT
              typed HttpClient)                                      │      (multilingual, models/encoder/)
                                                                     └─▶ B: self-hosted LLM
                                                                            (Ollama / vLLM,
                                                                             JSON-schema output)

  Offline eval (in-process, no gateway, no FastAPI)
  run_eval.py ──▶ classifiers.get(method) ──▶ A and B ──▶ metrics.py ──▶ docs/results.md
                                                   (accuracy, p50/p95, throughput, cost/1k)
```

- **A** fine-tunes `distilbert-base-multilingual-cased` on the labelled data.
- **B** calls a local model (`qwen2.5:7b-instruct`, or `qwen2.5:14b` on a GPU) through
  an OpenAI-compatible endpoint, so the same code runs unchanged against vLLM later.
- The .NET 8 Minimal API is the optional gateway described above.

The reasoning behind fine-tune vs. prompt is written up in [`docs/DESIGN.md`](docs/DESIGN.md),
and the code conventions the repo follows are in [`CONVENTIONS.md`](CONVENTIONS.md).

## Results

Held-out set of 32 bilingual examples, run 2026-06-14. The encoder is the fine-tuned
DistilBERT on CPU; the LLM is `qwen2.5:7b-instruct` through Ollama.

| Approach | Accuracy | Macro-F1 | p50 (ms) | p95 (ms) | Throughput (req/s) | Cost / 1k ($)* |
|----------|---------:|---------:|---------:|---------:|-------------------:|---------------:|
| encoder  |    71.9% |    0.725 |     27.5 |     31.6 |               23.7 |         0.0006 |
| llm      |    93.8% |    0.933 |   3592.9 |   4429.8 |                0.2 |         1.0757 |

\* **The cost column is modelled, not measured.** Nothing here was billed. It is
measured latency multiplied by an assumed hourly rate for the serving box —
`ENCODER_HOST_USD_PER_HOUR=0.05` and `LLM_HOST_USD_PER_HOUR=0.90` from
`.env.example` (`src/eval/metrics.py:cost_per_1k`). Change the rate and the column
changes with it. Read it as "how the latency gap would price out on those
assumptions", never as a bill.

The LLM is more accurate with no training at all (93.8% vs 71.9%); the encoder is
roughly 130x faster per request (27 ms vs 3.6 s), and the cost ratio follows
arithmetically from that gap. What that means for choosing between them is in
[`docs/DESIGN.md`](docs/DESIGN.md). The generated table is
[`docs/results.md`](docs/results.md); `python -m src.eval.run_eval` rewrites it.

## What these numbers do not establish

- **32 examples.** No confidence intervals were computed. Treat 93.8% vs 71.9% as a
  direction, not a margin — one example is 3.1 points.
- **Device placement was not recorded.** The eval harness saves no hardware or
  device-split metadata, so this README cannot say how much of the 7B ran on GPU. The
  machine it was run on has an NVIDIA GTX 1050 Ti (4 GB), and Ollama does offload part
  of a model that size into it, so "CPU-only" would be wrong — an earlier version of
  this file said exactly that and has been corrected. The latency figures are what the
  box produced; which unit produced them is not in the record.

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

python -m pytest                  # tests; need neither the model nor Ollama
```

Serving:

```bash
python -m uvicorn src.api.main:app --port 8000   # http://127.0.0.1:8000/docs
cd dotnet/IntentRouter.Api && dotnet run          # optional gateway on :5080, forwards to :8000
```

Use `python -m uvicorn` (not bare `uvicorn`) so it runs under the same interpreter that
has the dependencies installed.

Example call through the .NET gateway:

```bash
curl -X POST http://127.0.0.1:5080/classify \
  -H "Content-Type: application/json" \
  -d '{"text":"اتخصم مني المبلغ مرتين","method":"encoder"}'
```

The same call straight to FastAPI, without the gateway: replace `5080` with `8000`.

> On Windows, use `127.0.0.1` rather than `localhost`. `localhost` can resolve to IPv6
> `::1` while Ollama and uvicorn bind `127.0.0.1`, which shows up as a hang.

## Tech

| Layer | Choice | Why |
|-------|--------|-----|
| Encoder | DistilBERT multilingual (HF `transformers`) | small, trains on CPU, handles AR and EN |
| LLM | `qwen2.5:7b-instruct` via Ollama (OpenAI-compatible) | multilingual, swappable for vLLM/14b later |
| Structured output | JSON-schema constraint + a validation fallback in code | keeps a chat model's output to valid labels |
| API | FastAPI + optional .NET 8 Minimal API gateway | ML in Python, public contract in .NET |
| Eval | scikit-learn plus a small cost model | the actual comparison |
| Tests | pytest, model-free, run in CI | refactors can't silently change parsing or metrics |
