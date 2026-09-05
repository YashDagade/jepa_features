#!/bin/bash
set -euo pipefail

# Submit the two missing CLS controls used by the short-paper cue ablation.
# Standard TR2 Color already has a matched CLS/8x8 pair, so it is not rerun.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_ROOT="${REPO_ROOT}/runs/paper-cls-cue-v1"
DRY_RUN=false

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

workers=(
  "${SCRIPT_DIR}/lewm_cls_shape_worker.sh"
  "${SCRIPT_DIR}/lewm_cls_blue_green_worker.sh"
)

[[ -z "$(git -C "${REPO_ROOT}" status --porcelain)" ]] || {
  echo "Refusing to prepare jobs from a dirty jepa_features repository." >&2
  exit 1
}

for worker in "${workers[@]}"; do
  bash -n "${worker}"
done

mkdir -p "${RUN_ROOT}/shape" "${RUN_ROOT}/blue-green"

commands=(
  "sbatch --parsable --output=${RUN_ROOT}/shape/train-%j.out --error=${RUN_ROOT}/shape/train-%j.err ${workers[0]}"
  "sbatch --parsable --output=${RUN_ROOT}/blue-green/train-%j.out --error=${RUN_ROOT}/blue-green/train-%j.err ${workers[1]}"
)

if [[ "${DRY_RUN}" == true ]]; then
  printf 'DRY RUN: %s\n' "${commands[@]}"
  exit 0
fi

job_ids=()
for command in "${commands[@]}"; do
  submitted=$(eval "${command}")
  job_ids+=("${submitted%%;*}")
done

(IFS=:; echo "LEWM_CLS_CUE_JOB_IDS=${job_ids[*]}")
