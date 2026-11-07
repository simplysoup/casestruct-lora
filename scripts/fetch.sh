#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }
exec uv run --with 'datasets>=4,<5' --with 'huggingface_hub>=0.35,<2' python -m casestruct.external "$@"
