# Reported evaluation results (v2.1)

Results regenerated on 2026-11-07 from code commit 8ca6f59d1f7313cb7699e16e55b60d067e960016 using `bash run.sh evaluate`. The evaluation used the saved v2 adapters and prepared splits; it did not retrain the adapters. Earlier performance inspection means the test set is not pristine. The OSHA result is the stronger headline; ConstructCIE's small N and +1.7 pp lower CI bound make that gain suggestive, not solid.

| Task | System | Accuracy | Macro F1 | Strict JSON |
|---|---|---:|---:|---:|
| ConstructCIE (N=80) | base zero | 6.2% | 13.0% | 6.2% |
| | base few-shot | 75.0% | 72.3% | 100% |
| | base constrained | 75.0% | 75.2% | N/A |
| | v1 zero | 82.5% | 82.1% | 98.8% |
| | v1 few-shot | 77.5% | 72.8% | 100% |
| | v1 constrained | 81.2% | 80.8% | N/A |
| | v2 zero | 92.5% | 92.0% | 100% |
| | v2 few-shot | 90.0% | 89.6% | 100% |
| | **v2 constrained** | **93.8%** | **93.1%** | N/A |
| OSHA (N=333) | base zero | 0.9% | 2.3% | 0.9% |
| | base few-shot | 75.1% | 64.0% | 100% |
| | base constrained | 64.3% | 36.9% | N/A |
| | v1 zero | 48.6% | 34.2% | 98.5% |
| | v1 few-shot | 72.4% | 62.6% | 100% |
| | v1 constrained | 61.0% | 36.7% | N/A |
| | v2 zero | 86.5% | 80.0% | 100% |
| | v2 few-shot | 84.4% | 76.1% | 100% |
| | **v2 constrained** | **86.5%** | **80.1%** | N/A |

**Paired bootstrap headline (10,000 resamples, seed 71):** ConstructCIE, v2-constrained minus strongest non-v2 (v1-zero): accuracy +11.3 pp [2.5, 21.2], macro F1 +11.0 pp [1.7, 21.1]. OSHA, v2-constrained minus strongest non-v2 (base few-shot): accuracy +11.4 pp [7.5, 15.6], macro F1 +16.0 pp [9.2, 22.1]. Selection of strongest baseline is retrospective; take confidence intervals as descriptive rather than fully accounting for model-selection uncertainty.

Full paired-bootstrap estimates and confidence intervals, sanitized metrics, per-class scores, and confusion counts for all nine methods are in [`results-aggregate.json`](results-aggregate.json). Run `bash run.sh evaluate` locally to reproduce them from the required data and adapters; the v2-constrained matrices are visualized in [`confusion-matrices-v2.svg`](confusion-matrices-v2.svg). Headline values are in [`results-summary.json`](results-summary.json) and [`results-summary.svg`](results-summary.svg). To export another saved report bundle without copying its per-example files, run `python scripts/export_aggregate_reports.py /path/to/reports/v2.1`. The adjacent archive contains offline W&B run data, but no verified public W&B URL was supplied. Raw benchmark narratives and per-example predictions are not bundled.

## Clean holdout

These evaluation numbers must not be presented as clean-holdout performance. The next credible estimate requires a new sample collected or designated after freezing the model, labels, preprocessing, and evaluation plan. Keep it out of development and model selection; group duplicate or related incidents; document sampling and adjudication; evaluate frozen systems once; and publish only sanitized aggregate counts, confusion matrices, and uncertainty intervals. See [`methodology.md`](methodology.md#clean-holdout-plan) for the protocol.
