#!/usr/bin/env bash
# One-paste RunPod bootstrap. In the pod's web terminal:
#
#   export HF_TOKEN=hf_xxx
#   curl -sL https://raw.githubusercontent.com/anthnguyen/chipmunk/main/scripts/pod.sh | bash
#
# Runs under /workspace so results survive a pod stop. Gate 0 runs FIRST and the
# script exits if it fails -- that is the point of the gate.
set -uo pipefail

BASE=/workspace
[ -d /workspace ] || BASE="$HOME"
export HF_HOME="$BASE/hf_cache"
export UV_CACHE_DIR="$BASE/uv_cache"
cd "$BASE"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

[ -d chipmunk ] || git clone https://github.com/anthnguyen/chipmunk
cd chipmunk && git pull --ff-only || true

uv venv --python 3.11
uv pip install -e .

MODEL="${CHIPMUNK_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
echo "=== smoke test ==="
.venv/bin/python scripts/smoke.py "$MODEL" || { echo "SMOKE FAILED"; exit 1; }

echo "=== gate 0 on $MODEL ==="
.venv/bin/python - <<PY || exit 1
import sys
from chipmunk import data, gate0
from chipmunk.model import Runner
r = Runner("$MODEL")
rep = gate0.run(r, data.build(), data.build_absolute(), out_dir="results/gate0")
print(gate0.verdict(rep))
sys.exit(0 if rep["GATE0_PASS"] else 1)
PY

echo "Gate 0 passed. Proceed to the training arms (PROTOCOL section 6)."
