#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
trap 'echo "v2.1 stopped at line $LINENO. Check reports/v2.1/audit.md and command output; existing results are intact." >&2' ERR
command -v uv >/dev/null || { echo 'uv must be installed'; exit 1; }
if [[ -f .env ]]; then set -a; source .env; set +a; fi
export WANDB_MODE="${WANDB_MODE:-offline}"
MODE="${1:-all}"
[[ "$MODE" =~ ^(all|quick|report|audit)$ ]] || { echo 'Usage: bash run_v21.sh [all|quick|report|audit]'; exit 2; }
uv sync --extra dev
uv run pytest -q
[[ -f data/v2/train.jsonl && -f data/v2/validation.jsonl && -f data/v2/test.jsonl && -f data/v2/manifest.json ]] || { echo 'Missing data/v2. Run v2 data preparation first.'; exit 1; }
[[ -f reports/v2/v2-test.predictions.jsonl ]] || { echo 'Missing saved v2 test predictions.'; exit 1; }
echo '[audit] Checking splits and training source; halting on serious leakage signals'
uv run python -m casestruct.v21.audit --out reports/v2.1
if [[ "$MODE" == audit ]]; then exit 0; fi
if [[ "$MODE" == report ]]; then uv run python -m casestruct.v21.report; exit 0; fi
LIMIT=0
if [[ "$MODE" == quick ]]; then LIMIT=12; fi
SUFFIX=test
if (( LIMIT > 0 )); then SUFFIX="test-sample${LIMIT}"; fi
for MODEL in base v1 v2; do
  for METHOD in zero fewshot constrained; do
    PRED="reports/v2.1/${MODEL}-${METHOD}-${SUFFIX}.predictions.jsonl"
    if [[ -s "$PRED" ]]; then
      echo "[skip] $PRED"
    else
      echo "[baseline] $MODEL / $METHOD"
      uv run python -m casestruct.v21.evaluate --model "$MODEL" --method "$METHOD" --limit "$LIMIT"
    fi
  done
done
if [[ "$MODE" == quick ]]; then
  echo 'Quick sample written; final results require bash run_v21.sh all.'
else
  uv run python -m casestruct.v21.report --reps 10000
  echo 'v2.1 reports written to reports/v2.1/'
fi
