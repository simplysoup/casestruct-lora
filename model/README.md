---
language:
  - en
license: other
library_name: peft
pipeline_tag: text-classification
base_model: Qwen/Qwen3-0.6B
tags:
  - lora
  - classification
  - occupational-safety
  - construction
---

# CaseStruct v2 LoRA adapter — model card

## Status

This card documents the CaseStruct v2 adapter recipe, but **adapter weights are not included or hosted by this repository**. The results below were recovered from saved v2.1 aggregate reports in the adjacent `casestruct-lora-old` archive; evaluation was not rerun. Do not treat this card as a release of model weights.

## Intended use

Research and educational exploration of two separate, narrow label-classification tasks using public incident narratives:

- Construction accident type (ConstructCIE, four-way classification).
- OSHA injury nature (eight-way classification over selected sufficiently represented labels).

This model is not intended for legal, clinical, employment, insurance, regulatory, or safety-critical decisions. It does not determine fault or replace professional review. OSHA administrative injury codes may not be inferable from the narrative alone.

## Training

- Base model: [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B).
- Method: supervised fine-tuning with TRL `SFTTrainer` and PEFT LoRA; rank 8, alpha 16; completion-only loss.
- Training data: training split prepared from ConstructCIE and an enriched OSHA Severe Injury Reports mirror. See [`docs/methodology.md`](../docs/methodology.md) for sources and terms.
- The checked-in prompt/completion builder and TRL collator were tested on a batch of 32 prepared examples: every prompt token was masked (`-100`), and every token in each completion (including EOS) was supervised. This verifies the current code path, not historical checkpoint provenance.

## Reported evaluation (exploratory)

The saved v2.1 reports used test sets that had been inspected during development; these are not clean holdouts. The aggregates below were recovered from those reports, not independently rerun.

| Task | Test N | v2 constrained macro-F1 | Paired bootstrap gain vs. strongest reported baseline (95% CI) |
|---|---:|---:|---:|
| OSHA injury nature | 333 | 80.1% | +16.0 pp [+9.2, +22.1] |
| ConstructCIE accident type | 80 | 93.1% | +11.0 pp [+1.7, +21.1] |

Full aggregate results, including all nine methods' per-class metrics and confusion counts: [`docs/results-aggregate.json`](../docs/results-aggregate.json). Raw narratives and per-example predictions were kept out of the published artifacts. No adapter weights are included.

## Limitations

- Test-set inspection and retrospective baseline selection create optimism risk.
- ConstructCIE test size is only 80; its lower confidence bound is a +1.7 pp gain.
- Source labels may be noisy, and administrative categories may not be evident from narrative text.
- Performance does not demonstrate robustness, external validity, or readiness for deployment.
- This repository does not include trained weights or independently verified run/checkpoint provenance.

## Citation and license

See the repository [LICENSE](../LICENSE) for the source-code license. Dataset and base-model terms are separate and remain applicable; consult each upstream source before use or redistribution. No license for absent adapter weights is implied by this card.
