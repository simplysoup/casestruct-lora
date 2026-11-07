#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
uv sync --extra dev
uv run pytest -q
uv run casestruct generate --data data/smoke --small
uv run casestruct report
printf 'Smoke test complete; no GPU or model download required.\n'
