#!/usr/bin/env bash
# One entry point for all project workflows. Run from anywhere.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
usage() {
  cat <<'HELP'
CaseStruct-LoRA
  bash run.sh test               fast unit tests (no model download)
  bash run.sh fetch              fetch external public datasets
  bash run.sh prepare            prepare v2 splits (requires fetched datasets)
  bash run.sh train-v1           original synthetic experiment (trains if absent)
  bash run.sh evaluate-v1       full synthetic baseline comparison
  bash run.sh train-v2           external-data experiment (trains if absent)
  bash run.sh quick-v2           12-example v2 experiment
  bash run.sh evaluate-v2       evaluate saved v2 adapter
  bash run.sh audit              v2.1 data and source audit only
  bash run.sh quick              v2.1 9-condition 12-case smoke test
  bash run.sh evaluate          v2.1 9-condition full benchmark + report (no training)
  bash run.sh report            regenerate v2.1 analysis from saved predictions
  bash run.sh demo TASK TEXT    classify fictional incident with saved v2 adapter
  bash run.sh benchmark TASK TEXT  51 runs, p50/p95 + peak VRAM (CUDA)
HELP
}
command="${1:-help}"; shift || true
case "$command" in
  test) uv sync --extra dev && uv run pytest -q ;;
  fetch) exec bash scripts/fetch.sh "$@" ;;
  prepare) exec uv run python -m casestruct.v2.data "$@" ;;
  train-v1) exec bash scripts/v1.sh "$@" ;;
  evaluate-v1) exec bash scripts/v1_full_eval.sh "$@" ;;
  train-v2) exec bash scripts/v2.sh all ;;
  quick-v2) exec bash scripts/v2.sh quick ;;
  evaluate-v2) exec bash scripts/v2.sh evaluate ;;
  audit) exec bash scripts/v21.sh audit ;;
  quick) exec bash scripts/v21.sh quick ;;
  evaluate) exec bash scripts/v21.sh all ;;
  report) exec bash scripts/v21.sh report ;;
  demo|benchmark)
    [[ $# -ge 2 ]] || { usage >&2; exit 2; }
    task="$1";shift
    text="$*"
    if [[ "$command" == benchmark ]]; then
      exec uv run python -m casestruct.v21.demo --task "$task" --text "$text" --runs 51
    else
      exec uv run python -m casestruct.v21.demo --task "$task" --text "$text"
    fi ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
