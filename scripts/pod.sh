#!/usr/bin/env bash
# One-paste RunPod bootstrap for chipmunk. In the pod's web terminal:
#
#   export GH_TOKEN=ghp_xxx        # required while the repo is private
#   curl -sH "Authorization: token $GH_TOKEN" \
#     https://raw.githubusercontent.com/anthnguyen/chipmunk/master/scripts/pod.sh | bash
#
# If the repo is public, drop GH_TOKEN and just curl the raw URL.
#
# Optional, before the curl:
#   export HF_TOKEN=hf_xxx                       # only needed for gated models
#   export CHIPMUNK_MODELS="Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct"
#
# Runs under /workspace so results survive a pod stop. Order is deliberate:
# smoke test (machinery) then gate 0 (science). It STOPS if gate 0 fails on
# every candidate, because training past a failed gate produces uninterpretable
# numbers -- that is the whole point of the gate.
set -uo pipefail

BASE=/workspace
[ -d /workspace ] || BASE="$HOME"
export HF_HOME="$BASE/hf_cache"
export UV_CACHE_DIR="$BASE/uv_cache"     # same fs as the venv -> hardlinks, survives stops
export HF_HUB_ENABLE_HF_TRANSFER=1
cd "$BASE"

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

REPO="https://github.com/anthnguyen/chipmunk"
[ -n "${GH_TOKEN:-}" ] && REPO="https://${GH_TOKEN}@github.com/anthnguyen/chipmunk"
if [ ! -d chipmunk ]; then
  git clone "$REPO" chipmunk || {
    echo "clone failed. The repo is private -- export GH_TOKEN before running," >&2
    echo "or make the repo public." >&2; exit 1; }
fi
cd chipmunk
git pull --ff-only || true

# --system-site-packages inherits the RunPod image's CUDA-linked torch rather than
# resolving a fresh (possibly CPU-only) wheel. --no-deps on the package keeps uv
# from replacing that torch. Remaining deps are small and installed explicitly.
uv venv --system-site-packages
uv pip install --no-deps -e .
uv pip install "transformers>=5.0" "numpy>=2.0" "scikit-learn>=1.5" accelerate hf_transfer

PY=.venv/bin/python

echo
echo "=== environment ==="
$PY - <<'EOF' || exit 1
import sys, torch
print(f"  python  {sys.version.split()[0]}")
print(f"  torch   {torch.__version__}")
print(f"  cuda    {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("  ERROR: no CUDA device visible. This pod cannot run the study.")
    raise SystemExit(1)
p = torch.cuda.get_device_properties(0)
print(f"  gpu     {p.name}, {p.total_memory/1e9:.0f} GB")
EOF
[ $? -eq 0 ] || exit 1

echo
echo "=== smoke test (machinery, ~2 min) ==="
$PY scripts/smoke.py Qwen/Qwen2.5-0.5B-Instruct || { echo "SMOKE FAILED - stop here."; exit 1; }

echo
echo "=== gate 0 (science) ==="
MODELS="${CHIPMUNK_MODELS:-Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct}"
$PY scripts/run_gate0.py $MODELS
GATE=$?

echo
if [ $GATE -eq 0 ]; then
  echo "Gate 0 PASSED. Results in $BASE/chipmunk/results/gate0/"
  echo "Next: the training arms (docs/PROTOCOL.md section 6)."
else
  echo "Gate 0 FAILED on every candidate. Do NOT train."
  echo "Read results/gate0/*/gate0.json. The verdict distinguishes:"
  echo "  FORMAT failure    - model answers a constant option, is not reading the"
  echo "                      choices. Needs a different template or a larger model."
  echo "                      Raising min_ratio will NOT help."
  echo "  KNOWLEDGE failure - genuinely unsure on close pairs. Raise min_ratio in"
  echo "                      data.build(), or move to a larger model."
fi
exit $GATE
