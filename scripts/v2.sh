#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
trap 'echo "FAILED at line $LINENO; inspect error above. Earlier artifacts were preserved." >&2' ERR
command -v uv >/dev/null || { echo 'Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh' >&2; exit 1; }
if [[ ! -f .env ]]; then cp .env.example .env; fi
set -a; source .env; set +a
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1
MODE="${1:-all}"
if [[ "$MODE" != all && "$MODE" != quick && "$MODE" != evaluate ]]; then echo 'Usage: bash scripts/run_v2.sh [all|quick|evaluate]'; exit 2; fi
printf '\n[setup] Syncing and checking tests\n'
uv sync --extra dev
uv run pytest -q
uv run python - <<'PY'
import torch
assert torch.cuda.is_available(),'NVIDIA CUDA not detected'
print('GPU:',torch.cuda.get_device_name(0),'VRAM:',round(torch.cuda.get_device_properties(0).total_memory/2**30,1),'GiB')
PY
if [[ "$MODE" != evaluate ]]; then
  if [[ ! -f data/external/constructcie.jsonl || ! -f data/external/osha_sir.jsonl ]]; then
    printf '\n[download] Fetching external datasets\n'
    bash scripts/fetch.sh
  fi
  printf '\n[prepare] Preparing split, label vocabularies and manifest\n'
  if [[ ! -f data/v2/manifest.json ]]; then uv run python -m casestruct.v2.data; fi
fi
[[ -f data/v2/manifest.json ]] || { echo 'Missing data/v2/manifest.json; run full pipeline first' >&2; exit 1; }
# Quick samples per task are used for sanity checks. Full evaluation is performed only in all/evaluate.
LIMIT=0
if [[ "$MODE" == quick ]]; then LIMIT=12; fi
SUFFIX=test
if [[ "$LIMIT" -gt 0 ]]; then SUFFIX=test-sample${LIMIT}; fi
printf '\n[baseline] Base model on external test\n'
[[ -f "reports/v2/base-${SUFFIX}.metrics.json" ]] || uv run python -m casestruct.v2.evaluate --mode base --limit "$LIMIT"
if [[ -f outputs/local/adapter/adapter_config.json ]]; then
  printf '\n[transfer] Frozen v1 adapter on new tasks\n'
  [[ -f "reports/v2/v1-${SUFFIX}.metrics.json" ]] || uv run python -m casestruct.v2.evaluate --mode v1 --adapter outputs/local/adapter --limit "$LIMIT"
else
  echo 'Skipping v1 adapter: outputs/local/adapter not found'
fi
if [[ "$MODE" != evaluate ]]; then
  printf '\n[train] Training external-data adapter\n'
  if [[ ! -f outputs/v2/adapter/adapter_config.json ]]; then uv run python -m casestruct.v2.train; fi
fi
if [[ -f outputs/v2/adapter/adapter_config.json ]]; then
  printf '\n[adapter] Testing v2 adapter\n'
  [[ -f "reports/v2/v2-${SUFFIX}.metrics.json" ]] || uv run python -m casestruct.v2.evaluate --mode v2 --adapter outputs/v2/adapter --limit "$LIMIT"
else
  echo 'No v2 adapter found: training may not have completed'
fi
uv run python -m casestruct.v2.report
printf '\nCOMPLETE: reports/v2/summary.md\n'
