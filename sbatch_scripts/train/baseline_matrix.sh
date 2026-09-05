#!/bin/bash
set -euo pipefail

# Submit the native global-CLS baseline: four model families on TR2 Color and
# two-color Chaser. Generated job files and outputs stay under ignored runs/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON=/scratch/yd3062/conda_envs/stable_worldmodel_latest/bin/python

RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/native-cls-v1}"
WANDB_ENTITY="${WANDB_ENTITY:-yash21-dagade-duke-university}"
WANDB_PROJECT="${WANDB_PROJECT:-JEPA_Features}"
WANDB_GROUP="${WANDB_GROUP:-native-cls-baseline-v1}"
SEED="${SEED:-3072}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
RUN_REVISION="${RUN_REVISION:-}"
TR2_SIDECAR="${JEPA_FEATURES_TR2_SIDECAR:-/scratch/yd3062/memory-for-jepa/runs/tunnel_rooms_filled18x18/two_leaf/paired_color/tunnel_room_color_noise_filled18x18_location_probe_cue_and_position.h5}"
TR2_SIDECAR_SHA256=074d876b1af63a94ac2ca0dfe2d749fdfae7e820f2523e8eac994d3f1102d16e
DEPENDENCY=""
DRY_RUN=false

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dependency)
      DEPENDENCY="${2:?--dependency requires a job id}"
      shift 2
      ;;
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

if [[ "${DRY_RUN}" == false ]]; then
  [[ -z "$(git -C "${REPO_ROOT}" status --porcelain)" ]] || {
    echo "Refusing to submit from a dirty repository." >&2
    exit 1
  }
fi

if SOURCE_COMMIT=$(git -C "${REPO_ROOT}" rev-parse --verify HEAD 2>/dev/null); then
  :
elif [[ "${DRY_RUN}" == true ]]; then
  SOURCE_COMMIT=UNCOMMITTED_DRY_RUN
else
  echo "Create and validate the repository's first commit before submission." >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}"

# --- SWEEP ARRAYS ---
MODELS=(lewm pldm lpwm smwm)
DATASETS=(tunnel_room_two_color chaser_two_color)
# --------------------

dataset_path() {
  case "$1" in
    tunnel_room_two_color)
      echo "${JEPA_FEATURES_TR2_DATASET:-/scratch/yd3062/memory-for-jepa/datasets/tunnel_room_color_noise_filled18x18.h5}"
      ;;
    chaser_two_color)
      echo "${JEPA_FEATURES_CHASER_DATASET:-/scratch/yd3062/memory-for-jepa/datasets/chaser_cue_two_color.h5}"
      ;;
  esac
}

dataset_sha256() {
  case "$1" in
    tunnel_room_two_color)
      echo e134677f2c74d17de8765254a4b97b87c8258a898115e2dd1dcf8b3b2a3df042
      ;;
    chaser_two_color)
      echo c6d8a9abaf9c3fb69c69c7701051b60ae9ff7306e24ce79b545a67a70e21941b
      ;;
  esac
}

job_ids=()
for dataset in "${DATASETS[@]}"; do
  data_path=$(dataset_path "${dataset}")
  expected_sha=$(dataset_sha256 "${dataset}")

  for model in "${MODELS[@]}"; do
    run_id="${dataset}-${model}-native-cls-s${SEED}"
    if [[ -n "${RUN_REVISION}" ]]; then
      run_id="${run_id}-v${RUN_REVISION}"
    fi
    run_dir="${RUN_ROOT}/${run_id}"
    mkdir -p "${run_dir}"

    cat > "${run_dir}/run_info.txt" <<EOF
run_id=${run_id}
source_commit=${SOURCE_COMMIT}
model=${model}
representation=actual_vit_cls
dataset_config=${dataset}
dataset=${data_path}
dataset_sha256=${expected_sha}
seed=${SEED}
epochs=${MAX_EPOCHS}
batch_size=${BATCH_SIZE}
run_revision=${RUN_REVISION:-initial}
training_keys=pixels,action
decoder=detached_global
post_training_probe=encoder_cls,projector_current,predictor_future
wandb_project=${WANDB_PROJECT}
wandb_group=${WANDB_GROUP}
EOF

    inner="${run_dir}/train_inner.sh"
    cat > "${inner}" <<EOF
#!/bin/bash
set -euo pipefail

source /ext3/env.sh
conda activate /scratch/yd3062/conda_envs/stable_worldmodel_latest
export LD_LIBRARY_PATH=/scratch/yd3062/conda_envs/stable_worldmodel_latest/lib:\${LD_LIBRARY_PATH:-}
export PYTHONPATH=${REPO_ROOT}/stable-worldmodel:\${PYTHONPATH:-}
export STABLEWM_HOME=${run_dir}
export SPT_CACHE_DIR=${run_dir}/spt_cache
export WANDB_DIR=${run_dir}/wandb
export WANDB__SERVICE_WAIT=300
export WANDB__DISABLE_VIEWER=true
export WANDB_SILENT=true
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
mkdir -p \${SPT_CACHE_DIR} \${WANDB_DIR}

actual_sha=\$(sha256sum ${data_path} | cut -d' ' -f1)
[[ "\${actual_sha}" == "${expected_sha}" ]] || {
  echo "Dataset hash mismatch: \${actual_sha}" >&2
  exit 1
}

cd ${REPO_ROOT}/stable-worldmodel
${PYTHON} scripts/train/${model}.py \
  data=${dataset} \
  subdir=${run_id} \
  output_model_name=${run_id} \
  seed=${SEED} \
  trainer.max_epochs=${MAX_EPOCHS} \
  loader.batch_size=${BATCH_SIZE} \
  wandb.enabled=true \
  wandb.config.entity=${WANDB_ENTITY} \
  wandb.config.project=${WANDB_PROJECT} \
  wandb.config.group=${WANDB_GROUP} \
  wandb.config.name=${run_id} \
  wandb.config.id=${run_id} \
  wandb.config.resume=allow

checkpoint=${run_dir}/checkpoints/${run_id}/weights_epoch_${MAX_EPOCHS}.pt
test -f \${checkpoint}
mkdir -p ${run_dir}/probes

# A training job is complete only after its fixed, episode-disjoint probe has
# scored the encoder CLS, projected current state, and predicted future state.
if [[ "${dataset}" == tunnel_room_two_color ]]; then
  sidecar_sha=\$(sha256sum ${TR2_SIDECAR} | cut -d' ' -f1)
  [[ "\${sidecar_sha}" == "${TR2_SIDECAR_SHA256}" ]] || {
    echo "TR2 objective-probe sidecar hash mismatch: \${sidecar_sha}" >&2
    exit 1
  }
  ${PYTHON} scripts/probes/location_probe.py \
    world_model.checkpoint=\${checkpoint} \
    dataset.source_hdf5=${data_path} \
    location_probe.sidecar=${TR2_SIDECAR} \
    output.directory=${run_dir}/probes/location \
    wandb.enabled=true \
    wandb.config.entity=${WANDB_ENTITY} \
    wandb.config.project=${WANDB_PROJECT} \
    wandb.config.group=${WANDB_GROUP} \
    wandb.config.name=${run_id} \
    +wandb.config.id=${run_id} \
    +wandb.config.resume=allow \
    hydra.run.dir=${run_dir}/probes/location_hydra
else
  ${PYTHON} scripts/probes/chaser_probe.py \
    world_model.checkpoint=\${checkpoint} \
    dataset.path=${data_path} \
    dataset.sha256=${expected_sha} \
    output.path=${run_dir}/probes/chaser_results.json \
    wandb.enabled=true \
    wandb.config.entity=${WANDB_ENTITY} \
    wandb.config.project=${WANDB_PROJECT} \
    wandb.config.group=${WANDB_GROUP} \
    wandb.config.name=${run_id} \
    wandb.config.id=${run_id} \
    wandb.config.resume=allow \
    hydra.run.dir=${run_dir}/probes/chaser_hydra
fi
EOF

    slurm="${run_dir}/train.slurm"
    cat > "${slurm}" <<EOF
#!/bin/bash
#SBATCH --job-name=jf-${model}-${dataset}
#SBATCH --account=torch_pr_227_cilvr
#SBATCH --partition=a100_cilvr,l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s
#SBATCH --time=23:59:00
#SBATCH --mem=128G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yd3062@nyu.edu
#SBATCH --output=${run_dir}/train-%j.out
#SBATCH --error=${run_dir}/train-%j.err

set -euo pipefail
actual_commit=\$(git -C ${REPO_ROOT} rev-parse HEAD)
[[ "\${actual_commit}" == "${SOURCE_COMMIT}" ]] || {
  echo "Source commit changed: \${actual_commit}" >&2
  exit 1
}
test -f ${data_path}
if [[ "${dataset}" == tunnel_room_two_color ]]; then
  test -f ${TR2_SIDECAR}
fi

singularity exec --nv \
  --overlay /scratch/yd3062/singularity/overlay-50G-10M-stable-wm.ext3:ro \
  /share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
  /bin/bash ${inner}
EOF

    bash -n "${inner}"
    bash -n "${slurm}"
    if [[ "${DRY_RUN}" == true ]]; then
      echo "DRY RUN: ${slurm}"
    elif [[ -n "${DEPENDENCY}" ]]; then
      submitted=$(sbatch --parsable --dependency="afterok:${DEPENDENCY}" "${slurm}")
      job_ids+=("${submitted%%;*}")
    else
      test -f "${data_path}"
      submitted=$(sbatch --parsable "${slurm}")
      job_ids+=("${submitted%%;*}")
    fi
  done
done

if [[ "${DRY_RUN}" == false ]]; then
  (IFS=:; echo "BASELINE_JOB_IDS=${job_ids[*]}")
fi
