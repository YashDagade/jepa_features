"""Evaluate Chaser motion and cue-conditioned transitions with frozen probes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm
from stable_worldmodel.probes import (
    ChaserProbeDataset,
    FrozenLocationFeatures,
    build_chaser_probe_samples,
    extract_chaser_features,
    fit_chaser_probes,
    flatten_chaser_metrics,
)


def _write_json(path: Path, payload: dict) -> None:
    """Publish one complete result without overwriting prior evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    if path.exists() or temporary.exists():
        raise FileExistsError(f'Refusing to overwrite Chaser result {path}.')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def _log_wandb(cfg: DictConfig, result: dict, payload: dict) -> None:
    """Append probe metrics to the originating training run when requested."""
    if not bool(cfg.wandb.enabled):
        return
    import wandb

    run = wandb.init(**OmegaConf.to_container(cfg.wandb.config, resolve=True))
    run.log(flatten_chaser_metrics(result))
    run.summary.update(
        {
            'chaser_probe/sample_sha256': result['samples']['sample_sha256'],
            'chaser_probe/dataset_sha256': payload['dataset_sha256'],
            'chaser_probe/checkpoint_sha256': payload['checkpoint_sha256'],
            'chaser_probe/feature_width': result['feature_width'],
        }
    )
    run.finish()


@hydra.main(
    version_base=None,
    config_path='./config',
    config_name='chaser_probe',
)
def run(cfg: DictConfig) -> None:
    """Extract one shared feature set and fit every prespecified task."""
    started = time.perf_counter()
    dataset_path = Path(cfg.dataset.path).expanduser().resolve()
    checkpoint = Path(cfg.world_model.checkpoint).expanduser().resolve()
    if not dataset_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(
            f'Missing dataset or checkpoint: {dataset_path}, {checkpoint}'
        )

    # 1. Bind results to immutable dataset/checkpoint bytes before inference.
    with dataset_path.open('rb') as handle:
        dataset_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    with checkpoint.open('rb') as handle:
        checkpoint_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    expected_hash = cfg.dataset.get('sha256')
    if expected_hash is not None and dataset_hash != str(expected_hash):
        raise ValueError(
            f'Chaser dataset SHA-256 mismatch: {dataset_hash} != {expected_hash}'
        )

    # 2. Select one episode-disjoint sample set shared by all checkpoints.
    samples = build_chaser_probe_samples(
        dataset_path,
        seed=int(cfg.probe.seed),
        frameskip=int(cfg.probe.frameskip),
        history_size=int(cfg.probe.history_size),
        samples_per_split=int(cfg.probe.samples_per_split),
    )
    dataset = ChaserProbeDataset(samples)

    # 3. Freeze the trained model and expose encoder/projector/predictor stages.
    device = torch.device(str(cfg.device))
    model = swm.wm.utils.load_pretrained(
        str(checkpoint), cache_dir=cfg.world_model.get('cache_dir')
    ).to(device)
    model.requires_grad_(False)
    if hasattr(model, 'interpolate_pos_encoding'):
        model.interpolate_pos_encoding = True
    extractor = FrozenLocationFeatures(
        model,
        image_size=int(cfg.probe.image_size),
        history_size=samples.history_size,
        action_input_mean=dataset.action_input_mean,
        action_input_std=dataset.action_input_std,
    )
    features = extract_chaser_features(
        dataset,
        extractor,
        device=device,
        batch_size=int(cfg.probe.batch_size),
        num_workers=int(cfg.probe.num_workers),
        autocast=bool(cfg.probe.autocast),
    )

    # 4. Fit fixed ridge readouts and atomically publish all held-out metrics.
    result = fit_chaser_probes(
        features, samples, ridge_alpha=float(cfg.probe.ridge_alpha)
    )
    repo_root = Path(os.environ.get('STABLEWM_HOME', Path.cwd())).resolve()
    payload = {
        'schema_version': 1,
        'dataset': str(dataset_path),
        'dataset_sha256': dataset_hash,
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': checkpoint_hash,
        'git_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=repo_root, text=True
        ).strip(),
        'runtime_seconds': time.perf_counter() - started,
        'resolved_config': OmegaConf.to_container(cfg, resolve=True),
        'result': result,
    }
    output_path = Path(cfg.output.path).expanduser().resolve()
    _write_json(output_path, payload)
    _log_wandb(cfg, result, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    run()
