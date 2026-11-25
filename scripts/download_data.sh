#!/usr/bin/env bash
# Download the raw data into data/raw/ (git-ignored). Run from the project root:
#   bash scripts/download_data.sh
# Resumable: re-running continues a partial RICO download and skips finished steps.
set -euo pipefail

RAW=data/raw
mkdir -p "$RAW/rico"

# CLAY labels and official splits (Li et al., CHI '22). Pinned to the commit used here.
CLAY_COMMIT=eca50f8b351a10245d16d1efe72c10d266d7f528
if [ ! -d "$RAW/clay" ]; then
  git clone https://github.com/google-research-datasets/clay "$RAW/clay"
fi
git -C "$RAW/clay" checkout -q "$CLAY_COMMIT"

# RICO unique UIs: view hierarchies (.json) and screenshots (.jpg) (Deka et al., UIST '17).
# Server reports 6,471,262,799 bytes, md5 (base64) vc+Z1Nx/NqM5blfjBsdWlg==.
RICO_URL=https://storage.googleapis.com/crowdstf-rico-uiuc-4540/rico_dataset_v0.1/unique_uis.tar.gz
curl -S -C - --retry 10 --retry-delay 5 -o "$RAW/rico/unique_uis.tar.gz" "$RICO_URL"

# Extract everything (JSON and screenshots).
if [ ! -d "$RAW/rico/combined" ]; then
  tar -xzf "$RAW/rico/unique_uis.tar.gz" -C "$RAW/rico"
fi

python scripts/check_raw_data.py
