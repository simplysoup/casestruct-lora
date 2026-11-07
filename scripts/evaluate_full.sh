#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ -f .env ]] && { set -a; source .env; set +a; }
for split in test challenge; do
  for mode in zero-shot few-shot adapter; do
    uv run casestruct evaluate --mode "$mode" --split "$split"
  done
done
uv run casestruct report
