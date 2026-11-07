#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
exec bash scripts/run_v2.sh "${@:-all}"
