"""Fail-loud frozen linear probes evaluated throughout JEPA training."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import wandb
from lightning.pytorch.callbacks import Callback

from stable_worldmodel.probes import (
    FrozenLocationFeatures,
    LocationProbeSidecarDataset,
)
from stable_worldmodel.probes.location import (
    REPRESENTATION_SOURCES,
    extract_location_representation_features,
    fit_location_probe,
    flatten_probe_metrics,
)


def _write_json_no_clobber(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + '.partial')
    if path.exists() or partial.exists():
        raise FileExistsError(f'Refusing to overwrite epoch probe {path}.')
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True))
    partial.replace(path)


class EpochProbeCallback(Callback):
    """Measure acquisition at initialization and after every trained epoch."""

    def __init__(self, cfg: Any, run_dir: Path) -> None:
        super().__init__()
        self.cfg = cfg
        self.run_dir = Path(run_dir) / 'epoch_probes'
        self.kind = str(cfg.epoch_probe.kind)
        self.every_n_epochs = int(cfg.epoch_probe.every_n_epochs)
        if self.every_n_epochs <= 0:
            raise ValueError('epoch_probe.every_n_epochs must be positive.')

        if self.kind == 'tunnel_color':
            dataset = LocationProbeSidecarDataset(
                cfg.epoch_probe.sidecar,
                source_hdf5=cfg.epoch_probe.dataset,
            )
            expected = {
                name: int(cfg.epoch_probe.expected_split_counts[name])
                for name in ('fit', 'validation', 'test')
            }
            actual = {
                name: int((dataset.split_codes == code).sum())
                for code, name in enumerate(('fit', 'validation', 'test'))
            }
            if actual != expected:
                raise ValueError(
                    f'Epoch-probe sidecar split counts {actual} do not match '
                    f'{expected}.'
                )
            self.dataset = dataset
            self.sample_sha256 = dataset.metadata.sample_sha256
        else:
            raise ValueError(f'Unknown epoch probe kind {self.kind!r}.')

    def on_fit_start(self, trainer, pl_module) -> None:
        if trainer.is_global_zero:
            self._run(trainer, pl_module, epoch=0)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        epoch = trainer.current_epoch + 1
        if epoch % self.every_n_epochs == 0 or epoch == trainer.max_epochs:
            self._run(trainer, pl_module, epoch=epoch)

    def _extractor(self, pl_module) -> FrozenLocationFeatures:
        return FrozenLocationFeatures(
            pl_module.model,
            image_size=int(self.cfg.img_size),
            history_size=int(self.cfg.wm.history_size),
            action_input_mean=self.dataset.action_input_mean,
            action_input_std=self.dataset.action_input_std,
            freeze_model=False,
        )

    def _run(self, trainer, pl_module, *, epoch: int) -> None:
        extractor = self._extractor(pl_module)
        device = pl_module.device
        features, targets, splits, _sample_ids = (
            extract_location_representation_features(
                self.dataset,
                extractor,
                device=device,
                batch_size=int(self.cfg.epoch_probe.batch_size),
                num_workers=int(self.cfg.epoch_probe.num_workers),
                autocast_dtype=(
                    torch.bfloat16
                    if bool(self.cfg.epoch_probe.autocast)
                    else None
                ),
            )
        )
        _, result = fit_location_probe(
            features,
            targets,
            splits,
            self.dataset.metadata.target_names,
            float(self.cfg.epoch_probe.ridge_alpha),
            feature_sources=REPRESENTATION_SOURCES,
        )
        metrics = flatten_probe_metrics(result)
        expected_color_metrics = {
            f'location_probe_{stage}/test/color_binding/balanced_accuracy'
            for stage in REPRESENTATION_SOURCES
        }
        missing = sorted(expected_color_metrics - set(metrics))
        if missing:
            raise ValueError(
                'Tunnel color epoch probe is missing prespecified color '
                f'metrics: {missing}.'
            )
        result_payload = asdict(result)

        payload = {
            'schema_version': 1,
            'epoch': epoch,
            'global_step': int(trainer.global_step),
            'kind': self.kind,
            'sample_sha256': self.sample_sha256,
            'result': result_payload,
        }
        _write_json_no_clobber(
            self.run_dir / f'epoch_{epoch:03d}.json', payload
        )
        if wandb.run:
            wandb.log(
                {
                    **metrics,
                    'epoch_probe/epoch': epoch,
                    'epoch_probe/global_step': int(trainer.global_step),
                },
                step=trainer.global_step,
            )
            wandb.run.summary.update(
                {
                    'epoch_probe/kind': self.kind,
                    'epoch_probe/sample_sha256': self.sample_sha256,
                }
            )


__all__ = ['EpochProbeCallback']
