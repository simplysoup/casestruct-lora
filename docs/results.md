# Reported evaluation results (v2.1)

These metrics were supplied from a completed v2.1 run and have **not** been reproduced inside the packaged source snapshot. They are exploratory: earlier performance inspection means the test set is not pristine.

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

For full confidence intervals per method, run `bash run.sh evaluate` locally and open `reports/v2.1/summary.md` and `statistics.json`. No raw benchmark narratives are bundled.
