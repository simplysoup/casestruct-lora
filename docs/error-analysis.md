# Error analysis

| Task | Category | Count | Share of reviewed rows |
|---|---|---:|---:|
| ConstructCIE | MODEL_ERROR | 5 | 33.3% |
| ConstructCIE | LABEL_UNSUPPORTED | 1 | 6.7% |
| ConstructCIE | AMBIGUOUS | 9 | 60.0% |
| OSHA | MODEL_ERROR | 7 | 46.7% |
| OSHA | LABEL_UNSUPPORTED | 4 | 26.7% |
| OSHA | AMBIGUOUS | 4 | 26.7% |

Reviewed 15 error rows per task from the locally generated review CSV, sampled with seed 71 and proportionally stratified by gold label. Sampling was row-level across the v2 zero-shot, few-shot, and constrained conditions, so an incident can appear more than once across conditions; these are not unique-case rates.

ConstructCIE disagreements often involved event sequences with both a fall and an impact or entrapment, making the primary mechanism difficult to isolate. OSHA disagreements commonly involved multiple injury types or a distinction between a specific injury label and a broader administrative category. Several OSHA labels could not be confidently inferred from the narrative alone. This is a single-reviewer, selected error sample, not a representative estimate of error causes.
