# CaseStruct-LoRA

**A reproducible local LoRA study of incident-text classification and evaluation.**

CaseStruct trains a Qwen3-0.6B adapter using [PEFT](https://github.com/huggingface/peft) and [TRL](https://github.com/huggingface/trl) and evaluates two distinct tasks: construction accident type (ConstructCIE) and occupational injury nature (OSHA Severe Injury Reports). The earlier synthetic legal-intake prototype remains in `src/casestruct/` for historical reproducibility. **Neither external benchmark is a legal-document dataset.** The system is not validated for legal, clinical, or safety-critical use.

## At a glance

| External held-out task | N | Strongest evaluated non-v2 baseline macro-F1 | v2 constrained macro-F1 | Paired bootstrap gain (95% CI) |
|---|---:|---:|---:|---:|
| ConstructCIE accident type | 80 | 82.1% (v1 zero-shot) | **93.1%** | **+11.0 pp** [+1.7, +21.1] |
| OSHA injury nature | 333 | 64.0% (base few-shot) | **80.1%** | **+16.0 pp** [+9.2, +22.1] |

Numbers are **user-reported exploratory v2.1 outputs**, not independently reproduced in this source-only distribution. The v2 test set was inspected during development, so results are *not* from a pristine prospective holdout. Confidence intervals use 10,000 paired bootstrap resamples (seed 71). Do not average the two tasks. OSHA administrative codes are not always recoverable from narrative text. Full results: see [`docs/results.md`](docs/results.md).

## Get started

Requires Linux, Python 3.11/3.12, [`uv`](https://docs.astral.sh/uv/), and an NVIDIA CUDA GPU for model inference/training (tested by the project owner on an RTX 3070 Ti with ~8 GB VRAM). Public model weights download from Hugging Face; model and dataset files are **not** distributed in this repository.

```bash
uv sync --extra dev
bash run.sh test
bash run.sh fetch
bash run.sh prepare
bash run.sh train-v2      # only if an adapter does not already exist; may train
bash run.sh evaluate      # no training; audit first, then fair baselines and report
```

If you are **reusing a trained v2 adapter**, place it at `outputs/v2/adapter` and prepared splits at `data/v2/`. To reproduce the full nine-condition evaluation, the original v1 adapter at `outputs/local/adapter` and v2 test predictions at `reports/v2/v2-test.predictions.jsonl` are also required. `bash run.sh evaluate` explicitly refuses to proceed if those prerequisites are absent; it does not retrain.

**One-command everyday workflow:** `bash run.sh --help` lists commands. Example demo (use fictional inputs only):

```bash
bash run.sh demo constructcie 'A worker fell from a scaffold and fractured their arm.'
bash run.sh benchmark constructcie 'A worker fell from a scaffold and fractured their arm.'
```

The benchmark repeats one input 51 times; reports p50/p95, generated tokens/s and CUDA allocator peak. It is a microbenchmark, not a representative workload study. The loader currently requires CUDA; CPU measurements are not provided.

## Repository structure

```text
run.sh                      sole root workflow launcher
scripts/                    internal wrappers: fetch, v1, v2, v2.1, legacy eval
src/casestruct/             historical synthetic intake experiment
src/casestruct/external.py  public dataset downloader and normalization
src/casestruct/v2/          external-data preparation, training, inference, evaluation
src/casestruct/v21/         audit, 9-condition baselines, statistics, error analysis, demo
configs/                    reproducible experiment configuration
 tests/                     unit tests (no model weights required)
docs/                       results, methodology, provenance, publishing notes
data/, outputs/, reports/    generated locally; ignored by git
```

## Reproducibility and integrity

- Dataset preparation uses seed 71, grouped normalized narratives and task-specific label stratification. Hashes and label vocabularies are written to `data/v2/manifest.json`.
- v2 training code uses a prompt/completion dataset, `completion_only_loss=True`, LoRA rank 8 / alpha 16, and W&B offline logging by default. **Static source checks do not prove historical training masks or checkpoint provenance**; manually verify the actual training batch has `-100` for every prompt token and match the W&B run/commit before making full audit-pass claims.
- v2.1 evaluates base, v1, and v2 × zero-shot, four-shot, and constrained label scoring. Constrained choice uses total conditional log-likelihood; length-normalized scores are secondary. Strict JSON validity is N/A for the constrained method.
- v2.1 audit stops on cross-split exact duplicates, very high near-duplicate similarity, manifest inconsistencies, or checked-in training-source concerns. It is not proof of zero semantic leakage.
- Bootstrap comparisons use paired case IDs and a fixed label vocabulary. Error samples are **proposed**, not adjudicated; review `reports/v2.1/errors_to_review.csv` before publication.

For provenance, commands and limitations, read [`docs/methodology.md`](docs/methodology.md). To obtain reports locally run `bash run.sh evaluate`; the saved generated reports are intentionally **not** committed because they can contain source narratives and sensitive descriptions.

## Limitations and responsible use

Only public accident narratives are used; incidents may include sensitive medical or personal information. Check original source terms and privacy before redistributing **raw data, model predictions, or artifacts**. ConstructCIE has only 80 test cases. Near-duplicates, label noise, and inferential gaps remain possible. Strong few-shot baselines substantially narrow gains over unmodified Qwen3. These results establish neither deployment readiness nor suitability for legal, medical, or safety decisions.

## Development / history

This source-only distribution includes a **reconstructed snapshot history** (v1 → v2 → v2.1 → consolidation) because the author's original `.git` database was not supplied. The included historical snapshots are real code states from the available archives, but their commits are **not the original authored commits**. See [`docs/HISTORY.md`](docs/HISTORY.md). To preserve your **actual local** commit history while applying this cleanup, use `scripts/apply-to-existing-checkout.sh` instead of replacing your repository's `.git` directory.

## License

No project-level license has been granted by the original author. External datasets and model have their own terms; consult [`docs/methodology.md`](docs/methodology.md). Add a repository LICENSE only after selecting one deliberately.
