#!/bin/bash
#SBATCH --job-name=lewm-cls-bluegreen
#SBATCH --account=torch_pr_227_cilvr
#SBATCH --partition=a100_cilvr,l40s_public
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|l40s
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yd3062@nyu.edu

set -euo pipefail

SOURCE=/scratch/yd3062/worktrees/paper-bg-cls-e74da68
SOURCE_COMMIT=e74da680a57922ba8824f792d311f91771dd109b
DATA_ROOT=/scratch/yd3062/memory-for-jepa
RUN_ID=jepa-color-blue-green-probe-lewm-cls-idm0p0-tr2-color-blue-green-probe-s3072-r1
RUN_BASE=/scratch/yd3062/jepa_features/runs/paper-cls-cue-v1/blue-green
RUN_DIR=${RUN_BASE}/${RUN_ID}
DATASET=${DATA_ROOT}/datasets/tunnel_room_color_blue_green_noise_filled18x18.h5
SIDECAR=${DATA_ROOT}/runs/Aug04/jepa-color-blue-green-probe/dataset/tunnel_room_color_blue_green_noise_filled18x18_location_probe_cue_and_position.h5
SCENARIOS=${DATA_ROOT}/runs/Aug04/jepa-color-blue-green-probe/dataset/scenario_manifest.json

verify_sha256() {
  local path=$1
  local expected=$2
  local actual
  actual=$(sha256sum "${path}" | cut -d' ' -f1)
  [[ "${actual}" == "${expected}" ]] || {
    echo "SHA-256 mismatch for ${path}: expected ${expected}, found ${actual}" >&2
    exit 1
  }
}

actual_commit=$(git -C "${SOURCE}" rev-parse HEAD)
[[ "${actual_commit}" == "${SOURCE_COMMIT}" ]] || {
  echo "Source commit changed: expected ${SOURCE_COMMIT}, found ${actual_commit}" >&2
  exit 1
}
[[ -z "$(git -C "${SOURCE}" status --porcelain)" ]] || {
  echo "Source worktree is dirty: ${SOURCE}" >&2
  exit 1
}

verify_sha256 "${DATASET}" 6e2823ad5301c38dbd97d0506577cddb74c9fb80157f2b3a15395f0c08d4dcd7
verify_sha256 "${SIDECAR}" fe5567ff64587bf8f73cb6fc3c3d2065dbd8ca68283a07e641157a870c69d1c3
verify_sha256 "${SCENARIOS}" 1a789118b802e6b4ed02bfc2f2f0cff35e135d61a1a4fca8eabc36541e88db6a

if [[ "${JEPA_CUE_ABLATION_CONTAINER:-0}" != 1 ]]; then
  mkdir -p "${RUN_DIR}"
  [[ ! -e "${RUN_DIR}/run_info.txt" ]] || {
    echo "Run already initialized: ${RUN_DIR}" >&2
    exit 1
  }
  printf '%s\n' \
    "run_id=${RUN_ID}" \
    "source=${SOURCE}" \
    "source_commit=${SOURCE_COMMIT}" \
    "readout=native_cls" \
    "matched_control=jepa-color-blue-green-probe-lewm-8x8-idm0p0-tr2-color-blue-green-probe-s3072-r1" \
    "dataset=${DATASET}" \
    "dataset_sha256=6e2823ad5301c38dbd97d0506577cddb74c9fb80157f2b3a15395f0c08d4dcd7" \
    "sidecar=${SIDECAR}" \
    "sidecar_sha256=fe5567ff64587bf8f73cb6fc3c3d2065dbd8ca68283a07e641157a870c69d1c3" \
    "seed=3072" \
    "epochs=10" \
    "batch_size=128" \
    "sigreg_weight=0.09" \
    "idm_weight=0.0" \
    "wandb_project=JEPA_Color" \
    "wandb_id=${RUN_ID}" > "${RUN_DIR}/run_info.txt"
  exec singularity exec --nv \
    --overlay /scratch/yd3062/singularity/overlay-50G-10M-stable-wm.ext3:ro \
    /share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
    env JEPA_CUE_ABLATION_CONTAINER=1 /bin/bash "$0"
fi

source /ext3/env.sh
conda activate /scratch/yd3062/conda_envs/stable_worldmodel_latest
export LD_LIBRARY_PATH=/scratch/yd3062/conda_envs/stable_worldmodel_latest/lib:${LD_LIBRARY_PATH:-}
export STABLEWM_HOME=${DATA_ROOT}
export LOCAL_DATASET_DIR=${DATA_ROOT}
export PYTHONPATH=${SOURCE}/stable-worldmodel:${PYTHONPATH:-}
export WANDB__SERVICE_WAIT=300
export WANDB__DISABLE_VIEWER=true
export WANDB_SILENT=true
export WANDB_DIR=${RUN_DIR}/wandb/train
export SPT_CACHE_DIR=${RUN_DIR}/stable_pretraining_cache
export XDG_CACHE_HOME=${RUN_DIR}/xdg_cache
export HF_HOME=${RUN_DIR}/hf_cache
export TORCH_HOME=${RUN_DIR}/torch_cache
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${WANDB_DIR}" "${SPT_CACHE_DIR}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${TORCH_HOME}"
cd "${RUN_DIR}"

hydra_args=(
  'data=tunnel_room_color_blue_green_noise_filled18x18'
  'data.dataset.frameskip=5'
  "subdir=${RUN_ID}"
  "output_model_name=${RUN_ID}"
  "checkpoint_cache_dir=${RUN_BASE}"
  'checkpoint_epochs=[10]'
  'seed=3072'
  'wm.history_size=3'
  'wm.num_preds=1'
  'wm.matching_mode=b_t_d_pool'
  'trainer.max_epochs=10'
  'num_workers=6'
  'loader.batch_size=128'
  'loader.num_workers=6'
  "hydra.run.dir=${RUN_DIR}/hydra/train"
  'decoder.enabled=true'
  'decoder.num_clips=6'
  'simulated_rollouts.enabled=false'
  "simulated_rollouts.scenario_manifest=${SCENARIOS}"
  'location_probe.enabled=true'
  "location_probe.sidecar=${SIDECAR}"
  'location_probe.mlp.enabled=true'
  'energy_landscape.enabled=false'
  'post_training_eval.enabled=false'
  'feature_mode=global_cls'
  'data_mixture=disabled'
  'loss.sigreg.weight=0.09'
  'loss.idm.enabled=true'
  'loss.idm.weight=0.0'
  'wandb.enabled=true'
  'wandb.config.entity=yash21-dagade-duke-university'
  'wandb.config.project=JEPA_Color'
  'wandb.config.name=lewm-cls-idm0p0-tr2-color-blue-green-probe'
  "wandb.config.id=${RUN_ID}"
  'wandb.config.group=jepa-color-blue-green-probe'
  'wandb.config.job_type=train'
  'wandb.config.tags=[train,lewm,tunnel_room_color_blue_green_noise_filled18x18,blue-green-probe,context_3,jepa_color_blue_green_probe,feature_mode_global_cls,location_probe_mlp_true,sigreg_default,idm_idm0000,inforeg_default,temporal_jaccard_default,data_mix_off,frame_skip_5,action_block_5,short_paper_control]'
  'wandb.config.resume=allow'
)

python "${SOURCE}/stable-worldmodel/scripts/train/lewm.py" "${hydra_args[@]}"
checkpoint=${RUN_BASE}/checkpoints/${RUN_ID}/weights_epoch_10.pt
test -f "${checkpoint}"
ln -sfn "${RUN_BASE}/checkpoints/${RUN_ID}" "${RUN_DIR}/checkpoints"
