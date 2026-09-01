#!/usr/bin/env bash
# One-command launcher for the clinical-v4 diagnostic-collection replication.
# Required beforehand: export HF_TOKEN='hf_...'
set -uo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is required. Export it, then rerun this launcher." >&2
  exit 2
fi

export CHIPMUNK_MODELS="${CHIPMUNK_MODELS:-Qwen/Qwen2.5-7B-Instruct}"
export CHIPMUNK_RUN_ID="${CHIPMUNK_RUN_ID:-clinical-v4-diagnostic-collection-01}"
export CHIPMUNK_PREDICTION="${CHIPMUNK_PREDICTION:-H2: animal-size knowledge remains available and the trigger changes output policy; probe readability should largely persist despite geometric drift.}"
export CHIPMUNK_COLLECT_DIAGNOSTICS="${CHIPMUNK_COLLECT_DIAGNOSTICS:-1}"
export CHIPMUNK_BATCH_SIZE="${CHIPMUNK_BATCH_SIZE:-8}"
export CHIPMUNK_KEEP_VENV="${CHIPMUNK_KEEP_VENV:-1}"
export CHIPMUNK_SMOKE_VALIDATED="${CHIPMUNK_SMOKE_VALIDATED:-1}"
export RUNPOD_AUTO_STOP="${RUNPOD_AUTO_STOP:-1}"

LOG_PATH="${CHIPMUNK_LOG_PATH:-/workspace/chipmunk-diagnostic-run.log}"
curl -fsSL https://raw.githubusercontent.com/anthnguyen/chipmunk/master/scripts/pod.sh \
  | bash 2>&1 | tee "$LOG_PATH"
