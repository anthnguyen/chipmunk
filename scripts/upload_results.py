#!/usr/bin/env python
"""Upload a results directory to a private HF dataset repo (needs HF_TOKEN).

    upload_results.py [results]

Repo: <token owner>/chipmunk-results, one timestamped folder per upload.

Adapter checkpoints (*.pt) are included -- they are the artefact the whole
study is about, and the pod's disk is ephemeral. Model weights are not, since
they are re-downloadable from the Hub.
"""

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[upload] HF_TOKEN not set; skipping results upload")
        return 0

    results = REPO / (sys.argv[1] if len(sys.argv) > 1 else "results")
    if not results.exists():
        print(f"[upload] {results} does not exist; nothing to upload")
        return 0

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("[upload] huggingface_hub not installed; skipping")
        return 0

    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/chipmunk-results"
    api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    dest = time.strftime("%Y%m%d-%H%M%S") + f"-{results.name}"
    api.upload_folder(
        folder_path=str(results), repo_id=repo_id, repo_type="dataset",
        path_in_repo=dest,
        ignore_patterns=["*.partial.jsonl", "**/hf_cache/**", "**/__pycache__/**"],
    )
    print(f"[upload] {results} -> "
          f"https://huggingface.co/datasets/{repo_id}/tree/main/{dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
