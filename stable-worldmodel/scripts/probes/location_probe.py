"""Run a compact closed-form location probe on a frozen world model."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm
from stable_worldmodel.probes.location import (
    FEATURE_READOUT,
    FeatureCacheMetadata,
    FrozenLocationFeatures,
    LocationProbeSidecarDataset,
    extract_location_features,
    fit_location_probe,
    flatten_probe_metrics,
    flatten_probe_summary,
    snapshot_cache_path,
    write_feature_cache,
)


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_extractor(
    cfg: DictConfig,
    dataset: LocationProbeSidecarDataset,
    device: torch.device,
) -> FrozenLocationFeatures:
    model = swm.wm.utils.load_pretrained(
        str(cfg.world_model.checkpoint),
        cache_dir=cfg.world_model.get('cache_dir'),
    )
    model = model.to(device)
    model.requires_grad_(False)
    if hasattr(model, 'interpolate_pos_encoding'):
        model.interpolate_pos_encoding = True

    return FrozenLocationFeatures(
        model,
        image_size=int(cfg.location_probe.image_size),
        history_size=dataset.metadata.history_size,
        action_input_mean=dataset.action_input_mean,
        action_input_std=dataset.action_input_std,
    ).to(device)


def _resolve_checkpoint(path_or_name: str) -> Path:
    path = Path(path_or_name).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            'Location-probe snapshots require a local checkpoint file so its '
            f'content hash is unambiguous: {path}'
        )
    return path


def _write_json_no_clobber(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(
            f'Refusing to overwrite location-probe result {path}.'
        )
    temporary = path.with_suffix(path.suffix + '.tmp')
    if temporary.exists():
        raise FileExistsError(
            f'Incomplete location-probe result exists: {temporary}.'
        )
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def _log_wandb(
    cfg: DictConfig, metrics: dict, summary: dict, payload: dict
) -> None:
    if not bool(cfg.wandb.enabled):
        return
    import wandb

    run = wandb.init(**OmegaConf.to_container(cfg.wandb.config, resolve=True))
    run.log(metrics)
    summary['location_probe/cache'] = payload['feature_cache']
    run.summary.update(summary)
    run.finish()


@hydra.main(
    version_base=None,
    config_path='./config',
    config_name='location_probe',
)
def run(cfg: DictConfig) -> None:
    """Extract source-aligned features, fit fixed probes, and score test once."""
    started = time.perf_counter()
    ridge_alpha = float(cfg.location_probe.ridge_alpha)
    if not math.isfinite(ridge_alpha) or ridge_alpha <= 0:
        raise ValueError(
            'location_probe.ridge_alpha must be finite and positive.'
        )
    device = torch.device(str(cfg.device))
    sidecar = LocationProbeSidecarDataset(
        cfg.location_probe.sidecar,
        source_hdf5=cfg.dataset.get('source_hdf5'),
    )
    split_counts = {
        name: int((sidecar.split_codes == code).sum())
        for code, name in enumerate(('fit', 'validation', 'test'))
    }
    expected_counts = OmegaConf.to_container(
        cfg.dataset.expected_split_counts, resolve=True
    )
    if split_counts != expected_counts:
        raise ValueError(
            f'Location sidecar split counts {split_counts} do not match '
            f'configured counts {expected_counts}.'
        )

    checkpoint = _resolve_checkpoint(str(cfg.world_model.checkpoint))
    checkpoint_hash = _sha256_file(checkpoint)
    extractor = _load_extractor(cfg, sidecar, device)
    features, targets, splits, sample_ids = extract_location_features(
        sidecar,
        extractor,
        device=device,
        batch_size=int(cfg.location_probe.batch_size),
        num_workers=int(cfg.location_probe.num_workers),
        autocast_dtype=(
            torch.bfloat16
            if bool(cfg.location_probe.get('autocast', True))
            else None
        ),
    )
    expected_width = cfg.location_probe.get('expected_feature_width')
    if expected_width is not None and features.shape[2] != int(expected_width):
        raise ValueError(
            f'Expected {expected_width}D location features, got '
            f'{features.shape[2]}D.'
        )

    metadata = FeatureCacheMetadata(
        source_sha256=sidecar.metadata.source_sha256,
        checkpoint_sha256=checkpoint_hash,
        readout=FEATURE_READOUT,
        sample_sha256=sidecar.metadata.sample_sha256,
        feature_width=int(features.shape[2]),
    )
    cache_path = snapshot_cache_path(
        cfg.output.directory,
        checkpoint_hash,
        batch_index=cfg.snapshot.get('batch_index'),
        fit_end=bool(cfg.snapshot.fit_end),
    )
    write_feature_cache(
        cache_path,
        features,
        targets,
        splits,
        sample_ids,
        sidecar.metadata.target_names,
        metadata,
    )
    _, result = fit_location_probe(
        features,
        targets,
        splits,
        sidecar.metadata.target_names,
        ridge_alpha,
    )
    runtime = time.perf_counter() - started
    metrics = flatten_probe_metrics(result)
    summary = flatten_probe_summary(result)
    summary['location_probe/latest/runtime_seconds'] = runtime
    payload = {
        'feature_cache': str(cache_path),
        'feature_readout': FEATURE_READOUT,
        'runtime_seconds': runtime,
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': checkpoint_hash,
        'sidecar': str(sidecar.sidecar_path),
        'sidecar_metadata': asdict(sidecar.metadata),
        'result': asdict(result),
    }
    result_path = cache_path.with_suffix('.results.json')
    _write_json_no_clobber(result_path, payload)
    _log_wandb(cfg, metrics, summary, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == '__main__':
    os.environ.setdefault('MUJOCO_GL', 'egl')
    run()
