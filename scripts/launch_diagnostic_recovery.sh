#!/usr/bin/env bash
# Resume only the diagnostics for the completed clinical-v4 collection run.
# Required beforehand: export HF_TOKEN='hf_...'
set -uo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is required. Export it, then rerun this launcher." >&2
  exit 2
fi

LATEST_COMMIT=$(git ls-remote \
  https://github.com/anthnguyen/chipmunk.git refs/heads/master | awk '{print $1}')
if [ -z "$LATEST_COMMIT" ]; then
  echo "Could not resolve the latest chipmunk commit." >&2
  exit 2
fi

export CHIPMUNK_COMMIT="$LATEST_COMMIT"
export CHIPMUNK_MODELS="Qwen/Qwen2.5-7B-Instruct"
export CHIPMUNK_RUN_ID="clinical-v4-diagnostic-collection-01"
export CHIPMUNK_SOURCE_RUN_ID="Qwen2.5-7B-Instruct-clinical-v4-diagnostic-collection-01"
export CHIPMUNK_SOURCE_SNAPSHOT="20260901-010419-results"
export CHIPMUNK_COLLECT_DIAGNOSTICS=1
export CHIPMUNK_DIAGNOSTICS_ONLY=1
export CHIPMUNK_BATCH_SIZE="${CHIPMUNK_BATCH_SIZE:-8}"
export CHIPMUNK_KEEP_VENV="${CHIPMUNK_KEEP_VENV:-1}"
export RUNPOD_AUTO_STOP="${RUNPOD_AUTO_STOP:-1}"

LOG_PATH="${CHIPMUNK_LOG_PATH:-/workspace/chipmunk-diagnostic-recovery.log}"
curl -fsSL https://raw.githubusercontent.com/anthnguyen/chipmunk/master/scripts/pod.sh \
  | bash 2>&1 | tee -a "$LOG_PATH"
