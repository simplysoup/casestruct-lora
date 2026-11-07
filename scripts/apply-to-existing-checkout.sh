#!/usr/bin/env bash
# Run from this cleaned repo, targeting an existing Git checkout.
# Preserves all existing commits and branches, stages no changes automatically.
set -Eeuo pipefail
source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${1:-}"
[[ -n "$target" && -d "$target/.git" ]] || { echo 'Usage: bash scripts/apply-to-existing-checkout.sh /path/to/your/old/git-checkout' >&2; exit 2; }
command -v rsync >/dev/null || { echo 'rsync is required' >&2; exit 1; }
[[ "$(cd "$target" && pwd)" != "$source_root" ]] || { echo 'Use a different target checkout' >&2; exit 2; }
if [[ -n "$(git -C "$target" status --porcelain)" ]]; then
  echo 'Target has uncommitted changes. Commit or stash first; refusing to overwrite.' >&2
  exit 1
fi
printf 'Updating source files in %s (preserving .git and generated data)
' "$target"
# Keep user-generated files, local secrets, lockfiles and history. Delete only known old source wrappers/docs.
for path in fetch.sh fetch_datasets.py run_v2.sh run_v21.sh V2_README.md V21_README.md README_APPEND.md install_v21.sh scripts/run_all.sh scripts/run_v2.sh scripts/run_v21.sh scripts/evaluate_full.sh scripts/smoke.sh; do
  rm -f "$target/$path"
done
rsync -a --exclude='.git/' --exclude='.venv/' --exclude='.env' --exclude='uv.lock' \
  --exclude='data/' --exclude='outputs/' --exclude='reports/' --exclude='wandb/' \
  "$source_root/" "$target/"
printf '
Cleanup applied. Review and commit changes in original history:
  cd %q
  git status --short
  git add -A && git commit -m "refactor: consolidate CaseStruct workflows and documentation"
' "$target"
