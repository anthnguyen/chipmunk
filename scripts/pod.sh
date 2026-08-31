#!/usr/bin/env bash
# One-paste RunPod bootstrap for chipmunk. In the pod's web terminal:
#
#   curl -sL https://raw.githubusercontent.com/anthnguyen/chipmunk/master/scripts/pod.sh | bash
#
# Optional, before the curl:
#   export HF_TOKEN=hf_xxx        # uploads results to <you>/chipmunk-results
#                                 # (also needed for gated models)
#   export RUNPOD_AUTO_STOP=1     # stop the pod when finished
#   export CHIPMUNK_MODELS="Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct"
#
# Runs under /workspace so results survive a pod stop. Order is deliberate:
# smoke test (machinery) then gate 0 (science). It STOPS if gate 0 fails on
# every candidate, because training past a failed gate produces uninterpretable
# numbers -- that is the whole point of the gate.
set -uo pipefail

# RunPod images may export PYTHONPATH entries pointing at their preinstalled
# dist-packages. A normal venv does not ignore PYTHONPATH, so those packages can
# still leak into an otherwise clean environment. Torch-linked extensions from
# the image (torchaudio/torchvision/torchtext) are compiled against the image's
# torch and will crash when loaded beside the cu124 torch installed below.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1

BASE=/workspace
[ -d /workspace ] || BASE="$HOME"
export HF_HOME="$BASE/hf_cache"
export UV_CACHE_DIR="$BASE/uv_cache"     # same fs as the venv -> hardlinks, survives stops
export HF_HUB_ENABLE_HF_TRANSFER=1
cd "$BASE"

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "uv install failed" >&2; exit 1; }

REPO="https://github.com/anthnguyen/chipmunk"
[ -n "${GH_TOKEN:-}" ] && REPO="https://${GH_TOKEN}@github.com/anthnguyen/chipmunk"
if [ ! -d chipmunk ]; then
  git clone "$REPO" chipmunk || {
    echo "clone failed. If the repo has been made private again, export GH_TOKEN." >&2
    exit 1; }
fi
cd chipmunk
git pull --ff-only || true

# A self-contained venv, NOT --system-site-packages.
#
# Inheriting the image's packages failed twice. First `uv pip install accelerate`
# pulled a fresh torch (cu130) into the venv, shadowing the image's driver-matched
# build on a CUDA 12.4 driver. Then reinstalling torch as cu124 left the image's
# torchvision -- compiled against the old torch -- visible through system
# site-packages, so `torchvision::nms` no longer registered and transformers
# (which imports torchvision eagerly via its loss utils) died on import.
#
# Both are shadowing bugs from mixing two package sources. A clean venv with an
# explicit CUDA index costs a ~2.5 GB download once and is deterministic.
if [ -d .venv ] && [ "${CHIPMUNK_KEEP_VENV:-0}" != "1" ]; then
  echo "[setup] removing existing .venv (set CHIPMUNK_KEEP_VENV=1 to reuse)"
  rm -rf .venv
fi

# CHIPMUNK_KEEP_VENV predates the switch away from --system-site-packages. Do
# not preserve one of those legacy venvs: by construction it exposes the exact
# binary packages we need to keep out.
if [ -f .venv/pyvenv.cfg ] && grep -Eq '^include-system-site-packages[[:space:]]*=[[:space:]]*true' .venv/pyvenv.cfg; then
  echo "[setup] removing legacy system-site-packages venv"
  rm -rf .venv
fi
uv venv --python 3.11

# Pin the CUDA build to the driver. nvidia-smi's header shows the driver's
# maximum CUDA version; a wheel built for a newer one will not initialise.
CUDA_INDEX="${CHIPMUNK_CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"
echo "[setup] torch from $CUDA_INDEX"
uv pip install --index-url "$CUDA_INDEX" torch

uv pip install --no-deps -e .
# accelerate is deliberately absent: it depends on torch and would let uv
# re-resolve it. Nothing here needs it -- models load with
# from_pretrained().to(device), no device_map.
uv pip install "transformers>=5.0" "numpy>=2.0" "scikit-learn>=1.5" \
  hf_transfer huggingface_hub

PY=(.venv/bin/python -I)

# Guard against the failure that cost three rounds: a torch-linked package from
# the image resolving instead of the venv's. Those .so files are compiled against
# a specific torch build, so any mismatch surfaces as an undefined symbol or a
# missing operator, usually disguised as an unrelated transformers import error.
isolation_check() {
  "${PY[@]}" - <<'EOF'
import importlib.util
import sys
from pathlib import Path

venv = Path(sys.prefix).resolve()
bad = []
for name in (
    "torch", "torchvision", "torchaudio", "torchtext", "torchdata",
    "transformers", "numpy",
):
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.origin:
        continue
    origin = Path(spec.origin).resolve()
    if not origin.is_relative_to(venv):
        bad.append(f"{name} -> {spec.origin}")
if bad:
    print("  ERROR: packages resolving OUTSIDE the venv:")
    for b in bad:
        print(f"    {b}")
    print("  The venv is not isolated. Check PYTHONPATH and pyvenv.cfg.")
    raise SystemExit(1)
print("  isolation  ok (all torch-linked packages resolve inside the venv)")
EOF
}

cuda_check() {
  "${PY[@]}" - <<'EOF'
import sys, torch
print(f"  python  {sys.version.split()[0]}")
print(f"  torch   {torch.__version__}")
print(f"  cuda    {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit(1)
p = torch.cuda.get_device_properties(0)
print(f"  gpu     {p.name}, {p.total_memory/1e9:.0f} GB")
EOF
}

echo
echo "=== environment ==="
if ! cuda_check; then
  echo
  echo "  torch cannot see the GPU. The wheel is built for a newer CUDA than the"
  echo "  host driver supports (nvidia-smi's header shows the driver's maximum)."
  echo "  Reinstalling torch from $CUDA_INDEX. Optional torch extensions are"
  echo "  intentionally absent because this text-only experiment does not use them."
  echo "  Override the index with CHIPMUNK_CUDA_INDEX if the driver needs another."
  uv pip install --reinstall --index-url "$CUDA_INDEX" torch
  echo
  if ! cuda_check; then
    echo "  Still no CUDA device. Check nvidia-smi, and match the torch build to"
    echo "  the driver's CUDA version shown there." >&2
    exit 1
  fi
fi

isolation_check || exit 1

echo
echo "=== smoke test (machinery, ~2 min) ==="
"${PY[@]}" scripts/smoke.py Qwen/Qwen2.5-0.5B-Instruct || { echo "SMOKE FAILED - stop here."; exit 1; }

echo
echo "=== gate 0 (science) ==="
MODELS="${CHIPMUNK_MODELS:-Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct}"
"${PY[@]}" scripts/run_gate0.py $MODELS
GATE=$?

# Upload before any auto-stop: the pod's disk is ephemeral, and gate 0's
# verdict plus the base size direction are what the next session needs.
# No-ops cleanly when HF_TOKEN is unset.
echo
echo "=== upload ==="
"${PY[@]}" scripts/upload_results.py results || echo "[upload] failed (non-fatal)"

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

# Auto-stop runs regardless of the gate outcome -- a failed gate is exactly
# when you do not want to keep paying for an idle pod.
if [ "${RUNPOD_AUTO_STOP:-0}" = "1" ]; then
  bash scripts/stop_pod.sh
fi

exit $GATE
