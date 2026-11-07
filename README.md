# CaseStruct-LoRA

An offline-first, reproducible **Qwen3-0.6B LoRA** experiment for structured extraction of *fictional* personal-injury intake narratives. This is a research/portfolio demonstration, **not a legal tool or legal advice**. No real client information is used.

## Quick start (Linux + NVIDIA GPU)

Requires an NVIDIA driver, internet for the initial download, and [`uv`](https://docs.astral.sh/uv/). Tested design target: RTX 3070 Ti 8 GB. **The end-to-end training has NOT been executed in the build environment**; the included automated tests exercise generation and scoring, not GPU training.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # only if uv is absent
bash scripts/run_all.sh
```

That script installs dependencies, runs tests, checks CUDA, generates data, evaluates 12 zero-shot and 12 few-shot examples, trains an adapter, evaluates 12 adapter examples, and writes `reports/summary.md`. Steps with already-generated artifacts are skipped. Run from any directory.

**Optional W&B online mode:** change `.env` from `WANDB_MODE=offline` to `WANDB_MODE=online`, then `uv run wandb login` before starting. Offline runs still log locally and can later be synced using `uv run wandb sync wandb/offline-run-*`. No W&B account is required to get started.

**Full test/challenge evaluation** (much slower than the 12-example smoke comparisons):

```bash
bash scripts/evaluate_full.sh
```

`run_all.sh` never executes the full evaluation automatically because generation can be comparatively slow. The full-evaluation script evaluates each of the three model configurations on both test and challenge sets.

## Architecture

- `data.py`: generated narratives with controlled omission cases, distractors, labels, and disjoint phrase-family partitions. Deterministic seed and JSONL files.
- `train.py`: completion-only supervised fine-tuning with TRL + PEFT. LoRA targets linear projections, gradient checkpointing enabled, sequence limit 768. Saves just the adapter and tokenizer.
- `evaluate.py`: identical schema/prompts and deterministic decoding across zero-shot, few-shot and LoRA. Saved per-record predictions, JSON, schema and per-field metrics.
- `report.py`: generates Markdown comparison from persisted metric files.
- `scoring.py`: conservative exact-match and macro-field scoring; predictions invalid under the schema cannot score an exact record match.

## Try it on a fictional narrative

After training, run:

```bash
uv run casestruct extract --text "A van hit a stopped car on 2025-03-11 on King Street, Toronto. The other driver ran a red light. The client suffered a sprained wrist and visited urgent care."
```

This is a structured extraction demo, not an assessment of liability. There is no server, API key, or real client data.

## Custom operations

```bash
uv sync --extra dev
uv run pytest -q
uv run casestruct generate
uv run casestruct train --config configs/local.yaml
uv run casestruct evaluate --mode adapter --split test --limit 12
uv run casestruct report
```

Change `configs/local.yaml` for rank, training epochs and effective batch. Be careful with higher max sequence length on 8 GB cards. The default model is FP16/BF16, not quantized: 0.6B is small enough for a practical local LoRA experiment. If you hit OOM, close other GPU-heavy applications, lower `max_length` to 512, and consider adjusting training microbatch (already 1).

## Evaluation limitations

A generator with disjoint templates is *not* equivalent to independently written human narratives. The event descriptions, injury vocabulary and rendering rules are shared across splits. The test is stronger than a random row split, but is still template-correlated. No performance numbers should be taken as evidence of clinical, legal, or general document understanding. All claims must be backed by generated prediction files. A few-shot baseline uses the first three training records; adding examples costs prompt tokens and can impact latency.

`unsupported_field_rate` is an **approximate mismatch proxy**, not a verified semantic hallucination metric. A structurally valid incorrect string can be counted as unsupported even if it is a harmless paraphrase.

## GPU training and W&B

A model download is necessary once. W&B defaults to offline mode; checkpoints and generated artifacts are excluded from Git by `.gitignore`. Evaluation runs currently save to disk; W&B integration is active for training runs. This project deliberately does **not** auto-publish, upload to Hugging Face, or send data to W&B online without your opt-in.

## Next iteration

If this first comparison yields meaningful gains, expand to independently authored intake narratives, confidence calibration, optional human review, and at least three tracked hyperparameter runs. This initial repo is a working experiment scaffold, **not** a claim that such experiments have already occurred.
