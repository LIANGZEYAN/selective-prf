#!/usr/bin/env python3
"""
Download all four Rank-R1 models from HuggingFace with progress bars.

Do NOT call this during restructuring — only when ready to run setwise methods.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RANK_R1_MODELS, RANK_R1_CACHE_DIR


def main() -> None:
    from huggingface_hub import snapshot_download

    print(f"Download cache: {RANK_R1_CACHE_DIR}")
    print(f"Models to download: {len(RANK_R1_MODELS)}\n")

    for key, repo_id in RANK_R1_MODELS.items():
        print(f"Downloading {key}: {repo_id} ...")
        snapshot_download(
            repo_id=repo_id,
            cache_dir=RANK_R1_CACHE_DIR,
            resume_download=True,
        )
        print(f"  Done: {key}\n")

    print("All models downloaded.")


if __name__ == "__main__":
    main()
