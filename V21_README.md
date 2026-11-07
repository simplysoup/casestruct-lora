# CaseStruct-LoRA v2.1 — Evaluation hardening

## What exists (inspected v2 source)

- `src/casestruct/v2/data.py` prepares grouped, label-stratified ConstructCIE accident-type and OSHA injury-nature classification data in `data/v2/{train,validation,test}.jsonl`.
- `src/casestruct/v2/train.py` uses `SFTTrainer`, a prompt/completion dataset, `completion_only_loss=True`, PEFT LoRA (rank 8, alpha 16), validation evaluation and W&B offline tracking.
- `src/casestruct/v2/evaluate.py` generates JSON and uses `src/casestruct/v2/model.py::extract_label` for strict output parsing.
- `reports/v2/{base,v1,v2}-test.predictions.jsonl` and adapters are user-machine artifacts. They **are not included in the source archive**, so they cannot be validated or rerun here.
- Previously reported scores were 92.5% ConstructCIE accuracy (80 cases) and 86.5% OSHA accuracy (333 cases) for v2. Those are **prior preliminary results**, not v2.1 results.

## Run

Copy the update ZIP contents into the project root, preserving directory structure, then:

```bash
bash run_v21.sh audit    # no GPU, splits and source audit
bash run_v21.sh quick    # 12 cases per task, all 3 models x 3 methods (smoke test)
bash run_v21.sh          # full 9-condition benchmark, 10k bootstrap, report
bash run_v21.sh report   # regenerate reports from saved full predictions
```

`run_v21.sh` does **not retrain** or reprepare data; it does not modify v1/v2 reports, datasets, or weights. It reuses `outputs/local/adapter` (v1) and `outputs/v2/adapter` (v2). Stronger baseline experiments can be slow: conditional label scoring performs a separate forward pass for every allowed label. Only training/validation examples are used as few-shot demonstrations (k=4, seed=71, round-robin class balancing, deterministic sample order); prompt and scoring rules are fixed before testing.

## Evaluation methodology

Nine configurations: base, v1 adapter, v2 adapter × zero-shot JSON, 4-shot JSON and constrained class scoring. The same saved test records, same class vocabulary and strict JSON parser are used for all generated-output conditions. Constrained scoring compares **total conditional log-likelihood** of every complete JSON candidate, including `<|im_end|>`; its secondary length-normalized class prediction and per-label scores are saved. Constrained evaluation does not generate JSON, so strict JSON validity is **N/A**, not 100%.

Results (when run) are written under `reports/v2.1/`, including `summary.md`, `statistics.json`, raw `*.predictions.jsonl`, metrics, all-model confusion matrices and per-class precision/recall/F1, and `errors_to_review.csv`. Bootstrap uncertainty uses 10,000 resamples and seed 71, with samples paired by `(task,id)` and the same fixed label vocabulary. Per-task headline compares **v2 constrained** to the strongest non-v2 baseline by macro-F1; baseline selection is a reporting rule only and does not tune anything. ConstructCIE has only 80 test records.

Run your own fictional narrative through the adapter:

```bash
uv run python -m casestruct.v21.demo --task constructcie --text 'A worker fell from a ladder.'
uv run python -m casestruct.v21.demo --task constructcie --text 'A worker fell from a ladder.' --runs 51
```

The benchmark saves GPU name, warmup-excluded latency p50/p95, generated tokens/s and peak CUDA allocated bytes. The loader currently requires CUDA; CPU benchmarking is **not supported** by the pre-existing runtime. The benchmark measures allocator peak (not full process RSS) and measures a single repeated input, not a representative workload distribution.

## Audit and publication restrictions

`audit.md` reports normalized exact duplicates, approximate cross-split 4-gram near-duplicate candidates and counts at thresholds 0.80/0.90/0.95. The launcher halts on exact duplicates, near duplicates >=0.95, incompatible vocabularies or incorrect checked-in completion-loss settings; resolve concerns **before** evaluating more baselines. Candidate retrieval can miss paraphrases and shared underlying incidents.

Historical W&B provenance, whether test performance influenced model selection, and actual training-collator `-100` prompt masking cannot be conclusively verified from source alone; the audit explicitly marks these **NOT VERIFIED**. A token-mask contract unit test is included, but it does not establish historical collator behavior. Those must be confirmed by the owner before claiming a fully passed audit or a final publication-grade result. If historical leakage or masking errors are found, stop; retraining is outside v2.1 scope.

Do not claim the imported datasets are legal cases. ConstructCIE and OSHA are different tasks and their scores must not be averaged. OSHA codes may not be inferable from narratives alone. External narratives and injury descriptions may have redistribution/privacy conditions; check source licenses and content before publishing outputs. Test scores were already viewed during v2, so these are exploratory analyses rather than a newly untouched preregistered holdout. No legal, health, or safety fitness is implied.

## Data provenance

- ConstructCIE: https://huggingface.co/datasets/lab-flair/constructcie (dataset card lists Apache-2.0; confirm source terms)
- OSHA Severe Injury Reports enriched mirror: https://huggingface.co/datasets/masonmarker/osha-severe-injury-reports-enriched (mirror lists MIT; check original upstream terms)
- Source hashes and random seed are recorded in `data/v2/manifest.json`.

## Directory map

- Data preparation: `src/casestruct/v2/data.py`
- Original training (reproduction only, **not invoked by v2.1**): `src/casestruct/v2/train.py`
- Original evaluation: `src/casestruct/v2/evaluate.py`
- New audit: `src/casestruct/v21/audit.py`
- New baseline evaluation: `src/casestruct/v21/evaluate.py`, `baselines.py`
- New uncertainty and errors: `src/casestruct/v21/report.py`
- Demo and perf probe: `src/casestruct/v21/demo.py`
- One-command launcher: `run_v21.sh`

Reproduce original data and model training only if independently needed: `bash fetch.sh`, `uv run python -m casestruct.v2.data`, `uv run python -m casestruct.v2.train`. Do not rerun training during a v2.1 audit.
