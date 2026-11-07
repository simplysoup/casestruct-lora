# Frozen clean-holdout plan

**Status:** frozen before acquiring or inspecting any new holdout records. This plan is for an OSHA-only evaluation; ConstructCIE is omitted because the source manifest has 530 ConstructCIE records and the existing train/validation/test splits account for all 530.

## Frozen model and evaluation

- Evaluation-code baseline: repository commit `e31f8a1ed4437813f4087ddf6fe88b6a86dd026a`. Relevant source hashes: `src/casestruct/v2/prompts.py` `23b31af495b2fb04e0ff1899b22df2aeb40a9d51405df3971f6994ab119466d6`; `src/casestruct/v21/baselines.py` `4e81e947958fded26767d003cfa1d1cd8a03e08e0fa0fb7c27665b2324c2171a`; `src/casestruct/v21/evaluate.py` `0b42ab121562df957c34283098398976ae6881cef3628aea467ad99e53d26407`; `src/casestruct/v21/report.py` `2c06e2c660abe11485f7aa275608e69516186a5172e30ee10d21c8daaac77a53`.
- Base model: `Qwen/Qwen3-0.6B`.
- Adapter: `outputs/v2/adapter/adapter_model.safetensors`, SHA-256 `ffe8d30e56337d6d9ad58fdfbcb9b32361a08c39cd9103d0d5ef4189e6408b5f`; adapter config SHA-256 `496d58c043c3c8d817ccb7975774056853c8741aa2a4c8442b35379315740d4c`.
- Task: OSHA injury nature only. Frozen label vocabulary from the existing manifest, in full: `Amputations involving bone loss`; `Amputations, avulsions, enucleations unspecified`; `Cuts, lacerations, punctures without injury to internal structures`; `Fractures`; `Fractures and surface, flesh wounds`; `Heat exhaustion, fatigue`; `Thermal burns degree unspecified`; `Traumatic injuries or exposures unspecified`.
- Prompt: the checked-in `chat_prefix()` / `prompt()` implementation and task instruction in `src/casestruct/v2/prompts.py`; no prompt, label, or preprocessing edits after this plan is committed.
- Scoring: constrained choice by **total conditional log-likelihood of the complete JSON label candidate**. Primary metric: macro-F1 over the frozen eight-label vocabulary.
- Run the nine already-defined model/method conditions once on the new holdout, without training or model selection. Report the v2-constrained macro-F1 and its paired comparison against the strongest non-v2 baseline, selected by macro-F1 as in the existing report.
- Uncertainty: 10,000 paired bootstrap resamples, seed 71. No reruns, threshold changes, or replacement samples after seeing predictions.

## Holdout source and isolation

- Candidate source: a newly versioned OSHA Severe Injury Reports release containing records with incident dates from **2026-01-01 through 2026-06-30**, a time window after the existing source snapshot. The current `data/v2/manifest.json` identifies the previous OSHA source by SHA-256 `67666f8e5c29e838051a23d034f7f8e06adfef76452d0e7974bf5ddc6a11a38c`; the new release must have its own recorded version, retrieval date, and checksum. The candidate window is not assumed available until the source manifest verifies it.
- Include every eligible record in that window whose label is in the frozen vocabulary; do not select based on model output or expected sample size. If the source cannot establish the date window, lacks sufficient labeled records, or fails the isolation checks below, stop and report that the holdout could not be assembled. Do not silently substitute another period or task.
- Before evaluation, compare source IDs and normalized narratives against **train, validation, and the previously inspected test split**. Drop any matching source ID or normalized narrative. Use the checked-in `key_for()` normalization (case-fold, replace non-word runs with spaces, trim, SHA-256). Record excluded counts and the new source checksum, never the narratives.
- Freeze existing training/validation inputs by SHA-256: `train.jsonl` `814ff7e160b53b67f86b8f1fd8b55bab4cd310fdd3ee6429b907e80c1d72287a`; `validation.jsonl` `5ddd2fd24ffe092b0046e1be7fbf0bb4299a0d447429190abc2f8d05ec7de851`; inspected `test.jsonl` `78797cdee98b355e55e23b9bc8d0756abb9c499fcb46c7d6e7a48f5ce6d2a8d7`.

## One-shot reporting rule

After source and duplicate checks pass, freeze the resulting test IDs and checksum, run the nine conditions once, and publish the observed N, macro-F1, interval, and exclusions regardless of outcome. Keep raw narratives and per-example predictions local; commit only sanitized aggregates. If no qualifying new source is available, leave the prior exploratory results unchanged and document the unavailable clean holdout instead of reusing the inspected test set.
