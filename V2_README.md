# CaseStruct v2 — external injury-report classification

This repository **extends v1** without replacing v1 code, model, or metrics. Two distinct independently annotated classification tasks, tracked separately:

- **ConstructCIE:** classify accident mechanism using the verified `accident_type` gold label (not the nested causal span annotations; those remain available in raw imports).
- **OSHA Severe Injury Reports:** classify the coded **nature of injury** (`nature_title`) using the 8 most frequent eligible classes. This coding can incorporate information unavailable in the narrative. Therefore, report as administrative-label prediction rather than factual extraction.

These are not direct replacements for the old intake-extraction JSON task. The v1 adapter is evaluated as a transfer baseline under the new task prompt, and may underperform for reasons unrelated to original extraction quality.

## One command

From the existing `casestruct-lora` project directory after merging these files:

```bash
bash run_v2.sh
```

This installs dependencies, runs tests, downloads external datasets **only if needed**, validates and splits data, evaluates base and existing v1 adapters, fine-tunes a new v2 adapter on 8 GB CUDA with W&B offline by default, evaluates the new adapter, and builds `reports/v2/summary.md`. The long baseline inference on all held-out examples is much slower than the training; you can run `bash run_v2.sh quick` for 12 examples **per task** as a smoke experiment and later `bash run_v2.sh evaluate` for the complete test benchmark. Completed artifacts are skipped by filename.

## Data and leakage

The source datasets are the separately downloaded records at `data/external/*.jsonl`. The preparer writes `data/v2/{train,validation,test}.jsonl`, plus `data/v2/manifest.json` containing class vocabularies, counts, and source hashes. Splits are stratified by task and label, grouped by normalized full narratives to avoid exact normalized duplicate leakage. More sophisticated near-duplicate detection and cross-source overlap audits are NOT implemented. Original ConstructCIE has 530 cases; class counts and eligible OSHA records may result in fewer records after filtering.

No test data is used for training or hyperparameter selection. Validation loss is monitored during training. For multiple candidate runs use distinct output paths. The basic script does **one** 2-epoch LoRA run, not a hyperparameter search.

## Output

- `outputs/v2/adapter`: trained PEFT adapter.
- `outputs/v2/training_metrics.json`: training statistics.
- `reports/v2/*.predictions.jsonl`: per-example source, gold and predicted labels (these files may contain source report text).
- `reports/v2/*.metrics.json`: per-task accuracy, macro-F1, strict JSON validity, allowable label rate, latency, per-label support/F1.
- `reports/v2/summary.md`: transparent cross-run summary.

## Manual commands

```bash
uv run python -m casestruct.v2.data
uv run python -m casestruct.v2.evaluate --mode base --limit 12
uv run python -m casestruct.v2.train
uv run python -m casestruct.v2.evaluate --mode v2 --adapter outputs/v2/adapter
uv run python -m casestruct.v2.report
```

## Caveats

- Code is constructed and unit tested without access to your local external dataset files or CUDA; first actual run may surface mismatches. Report the error traceback for fixes.
- `v2` text classification is not a causal-span extraction model. ConstructCIE's nested evidence annotations are retained in the upstream import but are not used for training in v2.
- The OSHA data is an enriched third-party mirror; verify its provenance and any data-use restrictions before redistribution. Do not publish the full imported data without checking upstream terms.
- Data quality, label ambiguity, potentially sensitive report text, and performance on broader populations require further review.
- For W&B online sync: `uv run wandb login` and set `WANDB_MODE=online` in `.env`; otherwise training is logged offline.
