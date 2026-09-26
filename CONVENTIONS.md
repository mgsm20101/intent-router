# Code conventions

The rules this codebase follows. Each one lists the principle, where it shows up here, and
the mistake it avoids. They generalize to any applied-LLM project, not just this one.

## Structure

**1. Single source of truth.** Every fact is defined once.
`LABELS` lives only in `src/schema.py`; the encoder, the LLM prompt, the eval harness and
the API all read it. Duplicating the label set across files lets them drift apart silently.

**2. Program to the contract, not the implementation.** Callers depend on a stable type and
signature. Everything here speaks `Prediction` and `classify(text) -> Prediction`, so the two
classifiers are interchangeable and a caller never touches model internals.

**3. Dependencies point inward, no cycles.** Outer layers depend on inner ones, never the
reverse. Everything depends on `schema`; `schema` depends on nothing. `schema` importing from
`api` would create a cycle and make the core untestable.

**4. Pure core, impure edges.** Logic is pure functions; I/O lives at the boundaries.
`src/eval/metrics.py` is pure (inputs to outputs); reading files and printing happen in
`run_eval.py`. A calculation that also prints and reads files is hard to test and reuse.

**5. One reason to change per module (SRP).** `prompts.py` holds text, `classifier.py` makes
the call, `metrics.py` computes. One file doing prompt + call + metrics + printing is a
refactor magnet.

## Behavior

**6. Load lazily, cache the expensive thing.** Heavy resources load on first use, once.
`@lru_cache` wraps `_load()` (the encoder) and `_client()` (the LLM client). Loading the model
per request is a performance killer.

**7. Validate at the boundary.** Never trust external input or output. `_extract_label` in the
LLM classifier always returns a valid label even if the model rambles; Pydantic plus a regex
guards `method` at the API. Passing an LLM reply through unchecked blows up downstream.

**8. Handle errors in the right layer.** Each layer translates failure into its own language.
`predict` raises `FileNotFoundError`; the API maps that to HTTP 503 and any other failure to
502. Don't swallow errors silently, and don't leak a raw stack trace to the client.

**9. Be deterministic.** Same input, same output. `data/generate.py` has no randomness, training
uses `seed=42`, the LLM runs at `temperature=0`. Random data makes eval results unrepeatable.

## Form

**10. Config, not hardcoded constants.** Hosts and keys come from the environment.
`OLLAMA_HOST`, `LLM_MODEL`, `LLM_TIMEOUT_S` are read from `.env`. A hardcoded URL breaks the
moment the environment changes (local Ollama vs. a vLLM box).

**11. Type everything.** Type hints and dataclasses document and prevent errors.
`from __future__ import annotations`, the `Prediction` dataclass, and typed signatures throughout.
Passing a bare dict around means nobody knows its keys.

**12. Small functions, obvious happy path.** A function does one thing you can read at a glance.
`classify` is tokenize, forward, softmax, return. Nothing more.

**13. Docstrings state intent, not narration.** Explain why and how to run, not what the next
line literally does. Every module docstring here carries a `Run:` line. `# increment i by 1`
above `i += 1` is noise.

**14. Let tooling enforce style.** Consistency is a tool's job, not a debate.
`ruff` and `black` are configured in `pyproject.toml` (line length 100, rules E/F/I/UP/B).

## Review checklist

Before merging, scan for these:

- [ ] No fact duplicated; single source of truth
- [ ] Callers depend on a type/signature, not on internals
- [ ] Dependencies point inward; no import cycles
- [ ] Logic is pure; I/O is at the edges
- [ ] Heavy resources loaded lazily and cached
- [ ] All external input/output validated
- [ ] Config via env; no hardcoded hosts or secrets
- [ ] Signatures and types are explicit
- [ ] Errors translated in the right layer
- [ ] Deterministic (seed / temperature=0)
- [ ] Each module has one reason to change
- [ ] Functions small, happy path clear
- [ ] Docstrings state intent and how to run
- [ ] `ruff` and `black` clean
