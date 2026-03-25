#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTERNAL_DIR="${PROJECT_ROOT}/external"

mkdir -p "${EXTERNAL_DIR}"

clone_if_missing() {
  local repo_url="$1"
  local target_dir="$2"

  if [ -d "${target_dir}/.git" ] || [ -f "${target_dir}/.git" ]; then
    echo "skip: ${target_dir} already exists"
    return 0
  fi

  if [ -e "${target_dir}" ]; then
    echo "error: ${target_dir} exists but is not a git repository"
    exit 1
  fi

  git clone "${repo_url}" "${target_dir}"
}

clone_if_missing "https://github.com/facebookresearch/sam2.git" "${EXTERNAL_DIR}/sam2"
clone_if_missing "https://github.com/facebookresearch/dinov3.git" "${EXTERNAL_DIR}/dinov3"

echo "upstream source download completed"