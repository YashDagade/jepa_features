#!/bin/bash
set -euo pipefail

# Match three cue renderings under native LeWM and its explicit IDM adaptation.
# SOURCE_ROOT may name a clean, immutable worktree; generated jobs stay in runs/.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-${REPO_ROOT}}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/lewm-cue-ablation-v1}"
WANDB_PROJECT="${WANDB_PROJECT:-JEPA_Features}"
WANDB_GROUP="${WANDB_GROUP:-lewm-cue-ablation-v1}"
DRY_RUN=false
if [[ "$#" == 1 && "$1" == --dry-run ]]; then
  DRY_RUN=true
elif [[ "$#" != 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 1
fi
SOURCE_ROOT="$(cd "${SOURCE_ROOT}" && pwd)"
SOURCE_COMMIT=$(git -C "${SOURCE_ROOT}" rev-parse HEAD)
if [[ "${DRY_RUN}" == false ]]; then
  [[ -z "$(git -C "${SOURCE_ROOT}" status --porcelain)" ]] || {
    echo "Refusing to submit from dirty source: ${SOURCE_ROOT}" >&2
    exit 1
  }
fi
mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"

# --- SWEEP ARRAYS ---
MODELS=(lewm lewm_idm)
CUES=(color shape green_blue)
DATA_CONFIGS=(tunnel_room_two_color tunnel_room_two_shape tunnel_room_two_green_blue)
DATASETS=(
  /scratch/yd3062/memory-for-jepa/datasets/tunnel_room_color_noise_filled18x18.h5
  /scratch/yd3062/memory-for-jepa/datasets/tunnel_room_shape_noise.h5
  /scratch/yd3062/memory-for-jepa/datasets/tunnel_room_color_blue_green_noise_filled18x18.h5
)
DATA_HASHES=(
  e134677f2c74d17de8765254a4b97b87c8258a898115e2dd1dcf8b3b2a3df042
  7bea16d913f8f8fd8ea47d13277b8d07f71bb6efe4f307efdc57403b405efb94
  6e2823ad5301c38dbd97d0506577cddb74c9fb80157f2b3a15395f0c08d4dcd7
)
SIDECARS=(
  /scratch/yd3062/memory-for-jepa/runs/tunnel_rooms_filled18x18/two_leaf/paired_color/tunnel_room_color_noise_filled18x18_location_probe_cue_and_position.h5
  /scratch/yd3062/memory-for-jepa/runs/tunnel_rooms_shape/two_leaf/paired_shape/tunnel_room_shape_noise_location_probe_cue_and_position.h5
  /scratch/yd3062/memory-for-jepa/runs/Aug04/jepa-color-blue-green-probe/dataset/tunnel_room_color_blue_green_noise_filled18x18_location_probe_cue_and_position.h5
)
SIDECAR_HASHES=(
  074d876b1af63a94ac2ca0dfe2d749fdfae7e820f2523e8eac994d3f1102d16e
  41d31dbced311bd39bcde8952e544c49c486cf6c2ad6e381c7aa9e03f4960833
  fe5567ff64587bf8f73cb6fc3c3d2065dbd8ca68283a07e641157a870c69d1c3
)
# --------------------

jobs=()
for model in "${MODELS[@]}"; do
  idm_weight=0
  [[ "${model}" != lewm_idm ]] || idm_weight=3.125
for index in "${!CUES[@]}"; do
  run_id="${model}-${CUES[index]}-s3072-${SOURCE_COMMIT:0:12}"
  run_dir="${RUN_ROOT}/${run_id}"
  [[ ! -e "${run_dir}/run_info.txt" ]] || {
    echo "Refusing to overwrite existing run: ${run_dir}" >&2
    exit 1
  }
  mkdir -p "${run_dir}"
  cat > "${run_dir}/run_info.txt" <<EOF
run_id=${run_id}
source_root=${SOURCE_ROOT}
source_commit=${SOURCE_COMMIT}
dataset_config=${DATA_CONFIGS[index]}
dataset=${DATASETS[index]}
dataset_sha256=${DATA_HASHES[index]}
sidecar=${SIDECARS[index]}
sidecar_sha256=${SIDECAR_HASHES[index]}
model=${model}
representation=actual_vit_cls
objective=prediction_mse_plus_sigreg_0.09_plus_idm_${idm_weight}
initialization=random
seed=3072
epochs=10
batch_size=128
training_windows=416730
training_batch_updates=32550
history_size=3
num_preds=1
frameskip=5
decoder=detached_global
epoch_probe=disabled
final_probe=current_projected_and_one_step_predicted
wandb_entity=yash21-dagade-duke-university
wandb_project=${WANDB_PROJECT}
wandb_group=${WANDB_GROUP}
wandb_name=${run_id}
wandb_run_id=${run_id}
EOF
  job="${run_dir}/train.slurm"
  cat > "${job}" <<EOF
#!/bin/bash
#SBATCH --job-name=${model}-${CUES[index]}
#SBATCH --account=torch_pr_227_cilvr
#SBATCH --partition=a100_cilvr,l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --output=${run_dir}/train-%j.out
#SBATCH --error=${run_dir}/train-%j.err
set -euo pipefail
EOF
  for name in SOURCE_ROOT SOURCE_COMMIT WANDB_PROJECT WANDB_GROUP; do
    printf '%s=%q\n' "${name}" "${!name}" >> "${job}"
  done
  printf 'RUN_DIR=%q\nRUN_ID=%q\nDATA_CONFIG=%q\nDATASET=%q\nDATA_SHA=%q\nSIDECAR=%q\nSIDECAR_SHA=%q\n' \
    "${run_dir}" "${run_id}" "${DATA_CONFIGS[index]}" "${DATASETS[index]}" \
    "${DATA_HASHES[index]}" "${SIDECARS[index]}" "${SIDECAR_HASHES[index]}" >> "${job}"
  printf 'MODEL=%q\n' "${model}" >> "${job}"
  cat >> "${job}" <<'EOF'
[[ "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" == "${SOURCE_COMMIT}" ]]
[[ -z "$(git -C "${SOURCE_ROOT}" status --porcelain)" ]]
if [[ "${JEPA_CUE_FACTORIAL_CONTAINER:-0}" != 1 ]]; then
  exec singularity exec --nv \
    --overlay /scratch/yd3062/singularity/overlay-50G-10M-stable-wm.ext3:ro \
    /share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
    env JEPA_CUE_FACTORIAL_CONTAINER=1 /bin/bash "${RUN_DIR}/train.slurm"
fi
source /ext3/env.sh
conda activate /scratch/yd3062/conda_envs/stable_worldmodel_latest
export LD_LIBRARY_PATH=/scratch/yd3062/conda_envs/stable_worldmodel_latest/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=${SOURCE_ROOT}/stable-worldmodel:${PYTHONPATH:-}
export STABLEWM_HOME=${RUN_DIR}
export SPT_CACHE_DIR=${RUN_DIR}/spt_cache
export WANDB_DIR=${RUN_DIR}/wandb
export WANDB__SERVICE_WAIT=300 WANDB__DISABLE_VIEWER=true WANDB_SILENT=true
export HDF5_USE_FILE_LOCKING=FALSE HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "${SPT_CACHE_DIR}" "${WANDB_DIR}"
printf '%s  %s\n' "${DATA_SHA}" "${DATASET}" "${SIDECAR_SHA}" "${SIDECAR}" | sha256sum --check
cd "${RUN_DIR}"
test ! -e wandb_resume.json
PYTHON=/scratch/yd3062/conda_envs/stable_worldmodel_latest/bin/python
"${PYTHON}" "${SOURCE_ROOT}/stable-worldmodel/scripts/train/${MODEL}.py" \
  "data=${DATA_CONFIG}" "data.dataset.name=${DATASET}" \
  epoch_probe=disabled decoder.enabled=true decoder.detach=true \
  "subdir=${RUN_ID}" "output_model_name=${RUN_ID}" \
  seed=3072 trainer.max_epochs=10 loader.batch_size=128 \
  wm.history_size=3 wm.num_preds=1 data.dataset.frameskip=5 \
  wandb.enabled=true wandb.config.entity=yash21-dagade-duke-university \
  "wandb.config.project=${WANDB_PROJECT}" "wandb.config.group=${WANDB_GROUP}" \
  "wandb.config.name=${RUN_ID}" "wandb.config.id=${RUN_ID}" wandb.config.resume=never \
  "hydra.run.dir=${RUN_DIR}/hydra/train"
checkpoint="${RUN_DIR}/checkpoints/${RUN_ID}/weights_epoch_10.pt"
test -f "${checkpoint}"
"${PYTHON}" "${SOURCE_ROOT}/stable-worldmodel/scripts/probes/location_probe.py" \
  "world_model.checkpoint=${checkpoint}" "dataset.source_hdf5=${DATASET}" \
  "location_probe.sidecar=${SIDECAR}" "output.directory=${RUN_DIR}/probes/location" \
  wandb.enabled=true wandb.config.entity=yash21-dagade-duke-university \
  "wandb.config.project=${WANDB_PROJECT}" "wandb.config.group=${WANDB_GROUP}" \
  "wandb.config.name=${RUN_ID}" "+wandb.config.id=${RUN_ID}" +wandb.config.resume=must \
  "hydra.run.dir=${RUN_DIR}/hydra/probe"
EOF
  bash -n "${job}"
  jobs+=("${job}")
done
done

job_ids=()
for job in "${jobs[@]}"; do
  if [[ "${DRY_RUN}" == true ]]; then
    printf 'DRY RUN: %s\n' "${job}"
  else
    submitted=$(sbatch --parsable "${job}")
    job_ids+=("${submitted%%;*}")
  fi
done
if [[ "${DRY_RUN}" == false ]]; then
  (IFS=:; echo "LEWM_CUE_ABLATION_JOB_IDS=${job_ids[*]}")
fi
