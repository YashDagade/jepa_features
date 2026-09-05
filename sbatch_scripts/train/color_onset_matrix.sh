#!/bin/bash
set -euo pipefail

# Measure when localization and color binding emerge. The matrix keeps the four
# native objectives fixed while varying dataset, feature readout, and duration.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON=/scratch/yd3062/conda_envs/stable_worldmodel_latest/bin/python

RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/color-onset-v1}"
WANDB_ENTITY="${WANDB_ENTITY:-yash21-dagade-duke-university}"
WANDB_PROJECT="${WANDB_PROJECT:-JEPA_Features}"
WANDB_GROUP="${WANDB_GROUP:-color-onset-v1}"
SEED="${SEED:-3072}"
BATCH_SIZE="${BATCH_SIZE:-128}"
RUN_REVISION="${RUN_REVISION:-2}"
TR2_DATASET="${JEPA_FEATURES_TR2_DATASET:-/scratch/yd3062/memory-for-jepa/datasets/tunnel_room_color_noise_filled18x18.h5}"
CHASER_DATASET="${JEPA_FEATURES_CHASER_DATASET:-/scratch/yd3062/memory-for-jepa/datasets/chaser_cue_two_color.h5}"
TR2_SIDECAR="${JEPA_FEATURES_TR2_SIDECAR:-/scratch/yd3062/memory-for-jepa/runs/tunnel_rooms_filled18x18/two_leaf/paired_color/tunnel_room_color_noise_filled18x18_location_probe_cue_and_position.h5}"
TR2_SHA256=e134677f2c74d17de8765254a4b97b87c8258a898115e2dd1dcf8b3b2a3df042
CHASER_SHA256=c6d8a9abaf9c3fb69c69c7701051b60ae9ff7306e24ce79b545a67a70e21941b
TR2_SIDECAR_SHA256=074d876b1af63a94ac2ca0dfe2d749fdfae7e820f2523e8eac994d3f1102d16e
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

if [[ "${DRY_RUN}" == false ]]; then
  [[ -z "$(git -C "${REPO_ROOT}" status --porcelain)" ]] || {
    echo "Refusing to submit from a dirty repository." >&2
    exit 1
  }
fi
SOURCE_COMMIT=$(git -C "${REPO_ROOT}" rev-parse --verify HEAD)
mkdir -p "${RUN_ROOT}"

# --- SWEEP ARRAYS ---
MODELS=(lewm pldm lpwm smwm)
CONDITIONS=(
  chaser_cls_50
  tunnel_cls_50
  tunnel_spatial8x8_10
  tunnel_spatial8x8_50
)
# --------------------

job_ids=()
for condition in "${CONDITIONS[@]}"; do
  case "${condition}" in
    chaser_cls_50)
      dataset_config=chaser_two_color
      data_path=${CHASER_DATASET}
      data_sha=${CHASER_SHA256}
      feature_mode=global_cls
      readout=native_cls
      epoch_probe=chaser
      epochs=50
      ;;
    tunnel_cls_50)
      dataset_config=tunnel_room_two_color
      data_path=${TR2_DATASET}
      data_sha=${TR2_SHA256}
      feature_mode=global_cls
      readout=native_cls
      epoch_probe=tunnel_color
      epochs=50
      ;;
    tunnel_spatial8x8_10)
      dataset_config=tunnel_room_two_color
      data_path=${TR2_DATASET}
      data_sha=${TR2_SHA256}
      feature_mode=global_spatial_8x8
      readout=spatial_8x8
      epoch_probe=tunnel_color
      epochs=10
      ;;
    tunnel_spatial8x8_50)
      dataset_config=tunnel_room_two_color
      data_path=${TR2_DATASET}
      data_sha=${TR2_SHA256}
      feature_mode=global_spatial_8x8
      readout=spatial_8x8
      epoch_probe=tunnel_color
      epochs=50
      ;;
  esac

  for model in "${MODELS[@]}"; do
    run_id="${condition}-${model}-s${SEED}-v${RUN_REVISION}"
    run_dir="${RUN_ROOT}/${run_id}"
    mkdir -p "${run_dir}"

    inner="${run_dir}/train_inner.sh"
    slurm="${run_dir}/train.slurm"
    cat > "${run_dir}/run_info.txt" <<EOF
run_id=${run_id}
source_commit=${SOURCE_COMMIT}
model=${model}
condition=${condition}
feature_mode=${feature_mode}
readout=${readout}
dataset_config=${dataset_config}
dataset=${data_path}
dataset_sha256=${data_sha}
seed=${SEED}
epochs=${epochs}
batch_size=${BATCH_SIZE}
decoder_cadence=every_validation_epoch
epoch_probe_cadence=initialization_and_every_validation_epoch
probe_stages=encoder_readout,projector,predictor
chaser_curves=red_position_mae,runner_position_mae,cue_bacc,event_bacc,destination_bacc
tunnel_curves=agent_mae,tunnel_geometry_mae,green_slot_bacc
onset_rule=first_epoch_ci95_low_above_chance_and_next_epoch_confirms
wandb_project=${WANDB_PROJECT}
wandb_group=${WANDB_GROUP}
EOF

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
[[ "\${actual_sha}" == "${data_sha}" ]] || {
  echo "Dataset hash mismatch: \${actual_sha}" >&2
  exit 1
}
if [[ "${dataset_config}" == tunnel_room_two_color ]]; then
  sidecar_sha=\$(sha256sum ${TR2_SIDECAR} | cut -d' ' -f1)
  [[ "\${sidecar_sha}" == "${TR2_SIDECAR_SHA256}" ]] || {
    echo "Tunnel sidecar hash mismatch: \${sidecar_sha}" >&2
    exit 1
  }
fi

cd ${run_dir}
test ! -e wandb_resume.json
${PYTHON} ${REPO_ROOT}/stable-worldmodel/scripts/train/${model}.py \
  data=${dataset_config} \
  feature_mode=${feature_mode} \
  epoch_probe=${epoch_probe} \
  subdir=${run_id} \
  output_model_name=${run_id} \
  seed=${SEED} \
  trainer.max_epochs=${epochs} \
  loader.batch_size=${BATCH_SIZE} \
  wandb.enabled=true \
  wandb.config.entity=${WANDB_ENTITY} \
  wandb.config.project=${WANDB_PROJECT} \
  wandb.config.group=${WANDB_GROUP} \
  wandb.config.name=${run_id} \
  wandb.config.id=${run_id} \
  wandb.config.resume=allow

checkpoint=${run_dir}/checkpoints/${run_id}/weights_epoch_${epochs}.pt
test -f \${checkpoint}
test -f ${run_dir}/checkpoints/${run_id}/epoch_probes/epoch_000.json
test -f ${run_dir}/checkpoints/${run_id}/epoch_probes/epoch_$(printf '%03d' ${epochs}).json
mkdir -p ${run_dir}/probes

if [[ "${dataset_config}" == tunnel_room_two_color ]]; then
  ${PYTHON} ${REPO_ROOT}/stable-worldmodel/scripts/probes/location_probe.py \
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
  ${PYTHON} ${REPO_ROOT}/stable-worldmodel/scripts/probes/chaser_probe.py \
    world_model.checkpoint=\${checkpoint} \
    dataset.path=${data_path} \
    dataset.sha256=${data_sha} \
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

    cat > "${slurm}" <<EOF
#!/bin/bash
#SBATCH --job-name=jf-onset-${model}-${condition}
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
singularity exec --nv \
  --overlay /scratch/yd3062/singularity/overlay-50G-10M-stable-wm.ext3:ro \
  /share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
  /bin/bash ${inner}
EOF

    bash -n "${inner}"
    bash -n "${slurm}"
    if [[ "${DRY_RUN}" == true ]]; then
      echo "DRY RUN: ${slurm}"
    else
      test -f "${data_path}"
      submitted=$(sbatch --parsable "${slurm}")
      job_ids+=("${submitted%%;*}")
    fi
  done
done

if [[ "${DRY_RUN}" == false ]]; then
  (IFS=:; echo "COLOR_ONSET_JOB_IDS=${job_ids[*]}")
fi
