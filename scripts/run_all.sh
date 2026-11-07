#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
trap 'echo "ERROR at line $LINENO; see output above. Intermediate artifacts remain on disk." >&2' ERR

if ! command -v uv >/dev/null; then
  echo "uv missing. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi
if [[ ! -f .env ]]; then cp .env.example .env; fi
set -a
source .env
set +a
export PYTHONUNBUFFERED=1

echo '[1/7] Installing dependencies'
uv sync --extra dev

echo '[2/7] Running unit tests'
uv run pytest -q

echo '[3/7] Checking NVIDIA GPU'
uv run python - <<'PYGPU'
import torch
assert torch.cuda.is_available(), 'PyTorch CUDA not available; check NVIDIA driver or torch wheel'
print(torch.cuda.get_device_name(0), ' VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory/2**30, 1))
PYGPU

echo '[4/7] Generating datasets if absent'
if [[ ! -f data/manifest.json ]]; then uv run casestruct generate; fi

echo '[5/7] Baseline evaluations (12 examples each, to keep first run manageable)'
[[ -f reports/zero-shot-test-12.metrics.json ]] || uv run casestruct evaluate --mode zero-shot --split test --limit 12
[[ -f reports/few-shot-test-12.metrics.json ]] || uv run casestruct evaluate --mode few-shot --split test --limit 12

echo '[6/7] Training LoRA adapter'
if [[ ! -f outputs/local/adapter/adapter_config.json ]]; then uv run casestruct train; fi

echo '[7/7] Evaluating trained adapter and writing report'
uv run casestruct evaluate --mode adapter --split test --limit 12
uv run casestruct report

echo 'COMPLETE: reports/summary.md; adapter: outputs/local/adapter'
echo 'For full held-out evaluation run: bash scripts/evaluate_full.sh'
