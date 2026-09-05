#!/bin/bash
#SBATCH --job-name=lewm-cls-shape
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

SOURCE=/scratch/yd3062/worktrees/paper-shape-cls-8353c2b
SOURCE_COMMIT=8353c2b8835da17af3335aff07cf2dcc29394378
WORKER=/scratch/yd3062/jepa_features/sbatch_scripts/train/lewm_cls_shape_worker.sh
DATA_ROOT=/scratch/yd3062/memory-for-jepa
RUN_ID=tunnel-room-shape-lewm-cls-idm3p125-tr2-shape-s3072-r2
RUN_BASE=/scratch/yd3062/jepa_features/runs/paper-cls-cue-v1/shape
RUN_DIR=${RUN_BASE}/${RUN_ID}
DATASET=${DATA_ROOT}/datasets/tunnel_room_shape_noise.h5
SIDECAR=${DATA_ROOT}/runs/tunnel_rooms_shape/two_leaf/paired_shape/tunnel_room_shape_noise_location_probe.h5
SCENARIOS=${DATA_ROOT}/runs/tunnel_rooms_shape/two_leaf/scenario_manifest.json
RANK_SCENARIOS=${DATA_ROOT}/runs/tunnel_rooms_rank_eval_shape/two_leaf/eval_scenarios.json

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

verify_sha256 "${DATASET}" 7bea16d913f8f8fd8ea47d13277b8d07f71bb6efe4f307efdc57403b405efb94
verify_sha256 "${SIDECAR}" 62f9d1ace4ca19c2abf48ad3e001d689ba5cbdc4c2eae4df58d7be614e3ea8ce
verify_sha256 "${SCENARIOS}" 82370d3a095c2ecd105a6015e689ad348b5ed8835ea11351eb30f53881b99777
verify_sha256 "${RANK_SCENARIOS}" 5372d49d15e13c4b29371b8d545684ba849d3d96bab475de20e6d52d548d9c2e

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
    "matched_control=tunnel-room-shape-lewm-tr2-shape-s3072-r1" \
    "dataset=${DATASET}" \
    "dataset_sha256=7bea16d913f8f8fd8ea47d13277b8d07f71bb6efe4f307efdc57403b405efb94" \
    "sidecar=${SIDECAR}" \
    "sidecar_sha256=62f9d1ace4ca19c2abf48ad3e001d689ba5cbdc4c2eae4df58d7be614e3ea8ce" \
    "seed=3072" \
    "epochs=10" \
    "batch_size=128" \
    "sigreg_weight=0.09" \
    "idm_weight=3.125" \
    "wandb_project=Tunnel Room Shape" \
    "wandb_id=${RUN_ID}" \
    "supersedes_pretraining_failure_job=16958908" > "${RUN_DIR}/run_info.txt"
  exec singularity exec --nv \
    --overlay /scratch/yd3062/singularity/overlay-50G-10M-stable-wm.ext3:ro \
    /share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
    env JEPA_CUE_ABLATION_CONTAINER=1 /bin/bash "${WORKER}"
fi

source /ext3/env.sh
conda activate /scratch/yd3062/conda_envs/stable_worldmodel_latest
export LD_LIBRARY_PATH=/scratch/yd3062/conda_envs/stable_worldmodel_latest/lib:${LD_LIBRARY_PATH:-}
export STABLEWM_HOME=${DATA_ROOT}
export STABLEWM_DATA_CACHE_DIR=${DATA_ROOT}
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
  'data=tunnel_room_shape_noise'
  'data.dataset.frameskip=5'
  "subdir=${RUN_ID}"
  "output_model_name=${RUN_ID}"
  "checkpoint_cache_dir=${RUN_BASE}"
  'checkpoint_epochs=[10]'
  'seed=3072'
  'wm.history_size=3'
  'wm.num_preds=1'
  'trainer.max_epochs=10'
  'num_workers=6'
  'loader.batch_size=128'
  'loader.num_workers=6'
  "hydra.run.dir=${RUN_DIR}/hydra/train"
  'decoder.enabled=true'
  'decoder.num_clips=6'
  'location_probe.enabled=true'
  "location_probe.sidecar=${SIDECAR}"
  'energy_landscape.enabled=true'
  'energy_landscape.resolution=40'
  'energy_landscape.modes=[full]'
  'energy_landscape.record_at=[1.0]'
  'energy_landscape.batch_size=128'
  "energy_landscape.scenario_manifest=${SCENARIOS}"
  'energy_landscape.scenario_condition=paired_shape'
  'post_training_eval.enabled=true'
  'post_training_eval.strict=true'
  "post_training_eval.output_dir=${RUN_DIR}/post_training_eval"
  "post_training_eval.scenario_manifest=${RANK_SCENARIOS}"
  'post_training_eval.solvers=[cem,gd]'
  'post_training_eval.batch_size=10'
  'post_training_eval.ranked_tunnel_rooms.two_leaf.eval_budget=400'
  'post_training_eval.ranked_tunnel_rooms.two_leaf.episodes_per_rank=25'
  'post_training_eval.ranked_tunnel_rooms.wandb_videos_per_rank=3'
  'post_training_eval.plan_config.horizon=5'
  'post_training_eval.plan_config.receding_horizon=5'
  'post_training_eval.plan_config.action_block=5'
  'post_training_eval.action_probe.enabled=true'
  'aggregator=null'
  'loss.sigreg.weight=0.09'
  'loss.idm.enabled=true'
  'loss.idm.weight=3.125'
  'wandb.enabled=true'
  'wandb.config.entity=yash21-dagade-duke-university'
  'wandb.config.project=Tunnel Room Shape'
  'wandb.config.name=lewm-cls-idm3p125-tr2-shape'
  "wandb.config.id=${RUN_ID}"
  'wandb.config.group=tunnel-room-shape'
  'wandb.config.job_type=train'
  'wandb.config.tags=[train,lewm,tunnel_room_shape_noise,context_3,tunnel_room_shape,native_cls,sigreg_default,idm_idm3125,frame_skip_5,action_block_5,short_paper_control]'
  'wandb.config.resume=allow'
)

python "${SOURCE}/stable-worldmodel/scripts/train/lewm.py" "${hydra_args[@]}"
checkpoint=${RUN_BASE}/checkpoints/${RUN_ID}/weights_epoch_10.pt
test -f "${checkpoint}"
ln -sfn "${RUN_BASE}/checkpoints/${RUN_ID}" "${RUN_DIR}/checkpoints"
