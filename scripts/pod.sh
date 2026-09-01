#!/usr/bin/env bash
# One-paste RunPod bootstrap for chipmunk. In the pod's web terminal:
#
#   curl -sL https://raw.githubusercontent.com/anthnguyen/chipmunk/master/scripts/pod.sh | bash
#
# For the preconfigured diagnostic-collection replication, use the shorter
# scripts/launch_diagnostic_suite.sh launcher instead.
#
# Before the curl (paste the token between the quotes, never commit it):
#   export HF_TOKEN='hf_PASTE_YOUR_TOKEN_HERE'
#   export CHIPMUNK_PREDICTION='H2: ...'  # required before any training
# Optional:
#   export RUNPOD_AUTO_STOP=1     # stop the pod when finished
#   export CHIPMUNK_GATE_ONLY=1   # validate candidates but do not train
#   export CHIPMUNK_MODELS="Qwen/Qwen2.5-7B-Instruct"
#   export CHIPMUNK_BATCH_SIZE=4  # safe 7B default; use 8 on a 48 GB GPU
#   export CHIPMUNK_MIN_FREE_GIB=25  # preflight floor when a 7B candidate is selected
#   export CHIPMUNK_COLLECT_DIAGNOSTICS=1  # post-run validation activation collection
#
# Runs under /workspace so results survive a pod stop. Order is deliberate:
# smoke test, Gate 0, then the complete experiment on the first passing model.
# It stops before training on a failed gate or missing pre-registered prediction.
set -uo pipefail

# RunPod images may export PYTHONPATH entries pointing at their preinstalled
# dist-packages. A normal venv does not ignore PYTHONPATH, so those packages can
# still leak into an otherwise clean environment. Torch-linked extensions from
# the image (torchaudio/torchvision/torchtext) are compiled against the image's
# torch and will crash when loaded beside the cu124 torch installed below.
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
# Keep bootstrap logs readable. These suppress progress bars and successful
# package-resolution chatter, but command failures still print their diagnostics.
export UV_NO_PROGRESS=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
export TQDM_DISABLE=1

BASE=/workspace
[ -d /workspace ] || BASE="$HOME"
export HF_HOME="$BASE/hf_cache"
export UV_CACHE_DIR="$BASE/uv_cache"     # same fs as the venv -> hardlinks, survives stops
MODELS="${CHIPMUNK_MODELS:-Qwen/Qwen2.5-7B-Instruct}"
BATCH_SIZE="${CHIPMUNK_BATCH_SIZE:-4}"
# Xet's concurrent reconstruction has produced "Background writer channel
# closed" on sharded models. Disabling Xet uses the resumable fallback path.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
cd "$BASE"

echo "=== storage preflight ==="
df -h "$BASE"
for cache_dir in "$HF_HOME" "$UV_CACHE_DIR"; do
  [ ! -e "$cache_dir" ] || du -sh "$cache_dir" 2>/dev/null || true
done

case "$BATCH_SIZE" in
  ''|*[!0-9]*|0) echo "CHIPMUNK_BATCH_SIZE must be a positive integer." >&2; exit 1 ;;
esac

# A fresh 7B run needs the 7B weights, an isolated CUDA environment, adapter
# checkpoints, and all-layer captures on the same persistent filesystem. Catch
# an undersized pod volume before a multi-gigabyte download fails with EDQUOT.
case "$MODELS" in
  *Qwen2.5-7B*)
    MIN_FREE_GIB="${CHIPMUNK_MIN_FREE_GIB:-25}"
    case "$MIN_FREE_GIB" in
      ''|*[!0-9]*) echo "CHIPMUNK_MIN_FREE_GIB must be a non-negative integer." >&2; exit 1 ;;
    esac
    AVAILABLE_KIB=$(df -Pk "$BASE" | awk 'NR == 2 {print $4}')
    REQUIRED_KIB=$((MIN_FREE_GIB * 1024 * 1024))
    if [ "$AVAILABLE_KIB" -lt "$REQUIRED_KIB" ]; then
      echo
      echo "Not enough free persistent disk for the 7B overnight path." >&2
      echo "Available: $((AVAILABLE_KIB / 1024 / 1024)) GiB; required preflight: ${MIN_FREE_GIB} GiB." >&2
      echo "Increase the RunPod volume (50 GB total is a practical minimum)," >&2
      echo "or inspect existing caches before deliberately lowering CHIPMUNK_MIN_FREE_GIB." >&2
      exit 1
    fi
    ;;
esac

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || {
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
}
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "uv install failed" >&2; exit 1; }

REPO="https://github.com/anthnguyen/chipmunk"
[ -n "${GH_TOKEN:-}" ] && REPO="https://${GH_TOKEN}@github.com/anthnguyen/chipmunk"
if [ ! -d chipmunk ]; then
  git clone --quiet "$REPO" chipmunk || {
    echo "clone failed. If the repo has been made private again, export GH_TOKEN." >&2
    exit 1; }
fi
cd chipmunk
if [ -n "${CHIPMUNK_COMMIT:-}" ]; then
  git fetch --quiet origin "$CHIPMUNK_COMMIT"
  git -c advice.detachedHead=false checkout --quiet --detach "$CHIPMUNK_COMMIT"
else
  git pull --quiet --ff-only
fi

if [ "${CHIPMUNK_COLLECT_DIAGNOSTICS:-0}" = "1" ] && \
   [ ! -f scripts/exploratory_drift.py ]; then
  echo "Diagnostic preflight failed: scripts/exploratory_drift.py is absent at " \
       "commit $(git rev-parse --short HEAD). Refusing to train." >&2
  exit 3
fi

# A self-contained venv, NOT --system-site-packages.
#
# Inheriting the image's packages failed twice. First `uv pip install accelerate`
# pulled a fresh torch (cu130) into the venv, shadowing the image's driver-compatible
# build. Then reinstalling torch as cu124 left the image's
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
uv venv --quiet --python 3.11

# Pin the CUDA build to the driver. nvidia-smi's header shows the driver's
# maximum CUDA version; a wheel built for a newer one will not initialise.
CUDA_INDEX="${CHIPMUNK_CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"
echo "[setup] torch from $CUDA_INDEX"
uv pip install --quiet --index-url "$CUDA_INDEX" torch

uv pip install --quiet --no-deps -e .
# accelerate is deliberately absent: it depends on torch and would let uv
# re-resolve it. Nothing here needs it -- models load with
# from_pretrained().to(device), no device_map.
uv pip install --quiet "transformers>=5.0" "numpy>=2.0" "scikit-learn>=1.5" \
  huggingface_hub

# Unbuffered output keeps nohup logs live during downloads and long evaluations.
PY=(.venv/bin/python -u -I)
UPLOAD_ATTEMPTED=0

# Preserve whatever completed if a later stage exits unexpectedly. This is a
# fallback; the normal upload near the end remains the authoritative attempt.
upload_partial_on_exit() {
  exit_code=$?
  trap - EXIT
  if [ "$UPLOAD_ATTEMPTED" -eq 0 ] && [ -d results ]; then
    echo
    echo "=== partial-results upload after early exit ==="
    "${PY[@]}" scripts/upload_results.py results || true
  fi
  exit "$exit_code"
}
trap upload_partial_on_exit EXIT

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
  uv pip install --quiet --reinstall --index-url "$CUDA_INDEX" torch
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
if [ "${CHIPMUNK_SMOKE_VALIDATED:-0}" = "1" ]; then
  echo "SKIPPED: operator confirmed this exact environment already passed all"
  echo "substantive smoke checks; proceeding to the independent 7B science gate."
else
  "${PY[@]}" scripts/smoke.py Qwen/Qwen2.5-0.5B-Instruct || {
    echo "SMOKE FAILED - stop here."
    exit 1
  }
fi

echo
echo "=== gate 0 (science) ==="
"${PY[@]}" scripts/run_gate0.py $MODELS
GATE=$?

STATUS=$GATE
RUN_OUT=""
if [ $GATE -eq 0 ] && [ "${CHIPMUNK_GATE_ONLY:-0}" != "1" ]; then
  if [ -z "${CHIPMUNK_PREDICTION:-}" ]; then
    echo
    echo "Gate 0 passed, but CHIPMUNK_PREDICTION is empty."
    echo "Record the pre-training prediction required by PROTOCOL section 1, then rerun:"
    echo "  export CHIPMUNK_PREDICTION='H2: ...'"
    STATUS=2
  else
    MODEL=$("${PY[@]}" - <<'EOF'
import json
print(json.load(open("results/gate0/selected.json"))["model"])
EOF
)
    TAG="${MODEL##*/}"
    PROTOCOL_RUN_ID="${CHIPMUNK_RUN_ID:-clinical-v2}"
    RUN_OUT="results/runs/${TAG}-${PROTOCOL_RUN_ID}"
    mkdir -p "$RUN_OUT"
    cp "results/gate0/$TAG/gate0.json" "$RUN_OUT/gate0.json"
    cp "results/gate0/$TAG/base_size_direction.npy" "$RUN_OUT/base_size_direction.npy"
    echo
    echo "=== full experiment: $MODEL ==="
    "${PY[@]}" -m chipmunk --model "$MODEL" --out "$RUN_OUT" \
      --batch-size "$BATCH_SIZE" --prediction "$CHIPMUNK_PREDICTION"
    STATUS=$?
    if [ "${CHIPMUNK_COLLECT_DIAGNOSTICS:-0}" = "1" ] && [ -d "$RUN_OUT/arms" ]; then
      echo
      echo "=== exploratory diagnostic collection (validation only) ==="
      DIAG_ARGS=(
        --source-dir "$RUN_OUT"
        --out "$RUN_OUT/exploratory_drift"
        --model "$MODEL"
        --batch-size "$BATCH_SIZE"
        --planned-collection
      )
      if [ "${CHIPMUNK_DIAGNOSTICS_SELECTED_LAYER_ONLY:-0}" = "1" ]; then
        DIAG_ARGS+=(--selected-layer-only)
      fi
      "${PY[@]}" scripts/exploratory_drift.py "${DIAG_ARGS[@]}" || {
        echo "Diagnostic collection failed; preserving the completed arm records." >&2
        STATUS=3
      }
    fi
  fi
fi

# Upload after the final attempted stage and before any optional auto-stop.
echo
echo "=== upload ==="
if ! "${PY[@]}" scripts/upload_results.py results; then
  echo "[upload] failed; keeping the pod available for retry." >&2
  STATUS=4
fi
UPLOAD_ATTEMPTED=1

echo
if [ $GATE -eq 0 ] && [ $STATUS -eq 0 ]; then
  if [ "${CHIPMUNK_GATE_ONLY:-0}" = "1" ]; then
    echo "Gate 0 PASSED. Gate-only results in $BASE/chipmunk/results/gate0/"
  else
    echo "FULL EXPERIMENT COMPLETE. Results in $BASE/chipmunk/$RUN_OUT"
    echo "Report: $BASE/chipmunk/$RUN_OUT/REPORT.md"
  fi
elif [ $GATE -eq 0 ]; then
  echo "Gate 0 passed, but the full experiment did not complete (status $STATUS)."
else
  echo "No candidate passed Gate 0. Do NOT train."
  echo "Read results/gate0/summary.json. It distinguishes scientific failures"
  echo "from operational errors such as an interrupted model download."
  echo "Scientific verdicts distinguish:"
  echo "  FORMAT failure    - model answers a constant option, is not reading the"
  echo "                      choices. Needs a different template or a larger model."
  echo "                      Raising min_ratio will NOT help."
  echo "  KNOWLEDGE failure - genuinely unsure on close pairs. Raise min_ratio in"
  echo "                      data.build(), or move to a larger model."
fi

# A completed scientific run may stop automatically. Operational failures keep
# the pod alive so logs and preserved artifacts can be inspected or resumed.
if [ "${RUNPOD_AUTO_STOP:-0}" = "1" ]; then
  if [ $GATE -eq 0 ] && [ $STATUS -eq 0 ]; then
    bash scripts/stop_pod.sh
  else
    echo "[stop] skipped because the run has operational status $STATUS"
  fi
fi

exit $STATUS
