#!/usr/bin/env bash
# Stop this pod. runpodctl first, REST API as fallback (runpodctl is flaky
# inside pods). No-op unless RUNPOD_POD_ID is set, so it is safe to call
# anywhere.
set -uo pipefail
[ -n "${RUNPOD_POD_ID:-}" ] || { echo "[stop] not on a pod; skipping"; exit 0; }

if command -v runpodctl >/dev/null 2>&1; then
  echo "[stop] runpodctl stop pod $RUNPOD_POD_ID"
  runpodctl stop pod "$RUNPOD_POD_ID" && exit 0
  echo "[stop] runpodctl failed, trying REST"
fi

if [ -n "${RUNPOD_API_KEY:-}" ]; then
  echo "[stop] REST stop $RUNPOD_POD_ID"
  curl -s -X POST "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID/stop" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
    && exit 0
fi

echo "[stop] could not stop the pod automatically — stop it in the console." >&2
exit 1
