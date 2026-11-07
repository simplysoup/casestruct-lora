#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v uv >/dev/null; then echo 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/'; exit 1; fi
uv run --with 'datasets>=4,<5' --with 'huggingface_hub>=0.35,<2' python fetch_datasets.py "$@"
