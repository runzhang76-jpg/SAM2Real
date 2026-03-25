#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAM2_DIR="${PROJECT_ROOT}/checkpoints/sam2"
DINO_DIR="${PROJECT_ROOT}/checkpoints/dinov3"

mkdir -p "${SAM2_DIR}" "${DINO_DIR}"

download_file() {
  local url="$1"
  local output_path="$2"

  if [ -f "${output_path}" ]; then
    echo "skip: ${output_path} already exists"
    return 0
  fi

  if command -v curl >/dev/null 2>&1; then
    curl -L "${url}" -o "${output_path}"
    return 0
  fi

  if command -v wget >/dev/null 2>&1; then
    wget -O "${output_path}" "${url}"
    return 0
  fi

  echo "error: neither curl nor wget is available"
  exit 1
}

download_file \
  "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt" \
  "${SAM2_DIR}/sam2.1_hiera_tiny.pt"

download_file \
  "https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" \
  "${DINO_DIR}/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

echo "upstream checkpoint download completed"