# Methodology, provenance and reproduction

## Data

- [ConstructCIE](https://huggingface.co/datasets/lab-flair/constructcie): construction accident descriptions with annotations (dataset card lists Apache-2.0). Task: four-way accident type.
- [OSHA Severe Injury Reports enriched mirror](https://huggingface.co/datasets/masonmarker/osha-severe-injury-reports-enriched): occupational injury descriptions with official injury-nature codes (mirror lists MIT; upstream terms still apply). Task: eight-way injury nature, restricted to sufficiently represented labels.

Import with `bash run.sh fetch`, then prepare split with `bash run.sh prepare`. The importer retains original annotation fields; the prepared task uses narrative and the task-specific label. Official administrative fields may not be inferable from prose. The importer defaults to up to 3,000 distinct OSHA records; raw and prepared datasets are local-only and ignored by git.

Dataset split logic groups text normalized for case/punctuation and stratifies by task/label, seed 71, approximate train/validation/test 70/15/15. The subsequent v2.1 audit detects cross-split normalized exact duplicates and high-overlap candidates; grouping isn't a guarantee against paraphrase duplicates or shared incidents. Do not tune on the test set.

## Models and evaluation

- Base: `Qwen/Qwen3-0.6B`.
- v1: synthetic legal-intake LoRA adapter, used strictly as a transfer baseline.
- v2: external-data LoRA adapter, trained locally with rank 8 / alpha 16, using `SFTTrainer` and `completion_only_loss=True` in the checked-in training source.
- Nine conditions: three models × zero-shot greedy JSON, four-shot JSON (demonstrations from train only, fixed seed), and constrained choice using total log likelihood over allowed candidates. Same test IDs and labels across conditions; JSON validity is N/A for constrained choice.

Training: `bash run.sh train-v2` (trains only when no adapter exists). Evaluation only: `bash run.sh audit`, `bash run.sh evaluate`. Use `bash run.sh quick` for a 12-per-task smoke test. Generated metrics, prediction traces, confusion matrices, per-class scores, proposed errors, and paired-bootstrap intervals are written to `reports/v2.1/`.

All score improvements are exploratory. Original test cases were viewed during v2 development; confidence intervals don't undo this selection risk. The static audit does **not** establish historical token masking, W&B commit provenance, or source redistribution rights. These remain owner sign-off items. The actual training logs are under `wandb/` locally, not shipped. Connect W&B yourself and only then add verified public run links.

## What is not demonstrated

Accuracy on real legal cases; clinical injury classification suitability; robust safety-critical performance; an untouched post-development test; independent external replication; independently audited checkpoint lineage; CPU latency. Missing evidence must not be represented as a pass.
