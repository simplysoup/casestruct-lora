#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
exec bash scripts/run_v21.sh "${1:-all}"
