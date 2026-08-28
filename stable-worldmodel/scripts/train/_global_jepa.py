"""Shared native-CLS training lifecycle for the four JEPA baselines."""

import os
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import torch
import wandb
from einops import rearrange
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from stable_pretraining import data as dt

import stable_worldmodel as swm
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.decoder import GlobalDecoder
from stable_worldmodel.wm.utils import save_pretrained

from _epoch_probes import EpochProbeCallback


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    """Convert stored frames to normalized ImageNet tensors."""
    stats = dt.dataset_stats.ImageNet
    return dt.transforms.Compose(
        dt.transforms.ToImage(**stats, source=source, target=target),
        dt.transforms.Resize(img_size, source=source, target=target),
    )


class SaveCkptCallback(Callback):
    """Save portable model weights at the configured epoch interval."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        epoch = trainer.current_epoch + 1
        if trainer.is_global_zero and (
            epoch % self.epoch_interval == 0 or epoch == trainer.max_epochs
        ):
            save_pretrained(
                pl_module.model,
                run_name=self.run_name,
                config=self.cfg,
                config_key='model',
                filename=f'weights_epoch_{epoch}.pt',
            )


class EpochDecoderMediaCallback(Callback):
    """Upload one deterministic validation reconstruction every epoch."""

    def __init__(self, *, commit: bool = True) -> None:
        super().__init__()
        self.commit = bool(commit)

    def on_validation_epoch_end(self, trainer, pl_module):
        if (
            not trainer.is_global_zero
            or trainer.sanity_checking
            or not wandb.run
            or not hasattr(pl_module, '_epoch_decoder_media')
        ):
            return
        original, decoded = pl_module._epoch_decoder_media
        wandb.log(
            {
                'validate/epoch': trainer.current_epoch + 1,
                'validate/epoch_original_clip': wandb.Video(original),
                'validate/epoch_decoded_clip': wandb.Video(decoded),
            },
            step=trainer.global_step,
            commit=self.commit,
        )


def _decoder_videos(pixels, decoded, num_clips):
    """Convert normalized model tensors into W&B-ready uint8 clips."""
    # 1. Restore batch/time axes and select the same clips from both tensors.
    decoded = rearrange(
        decoded,
        '(b t) c h w -> b t c h w',
        b=pixels.size(0),
    )
    original = pixels[:num_clips].detach().cpu()
    decoded = decoded[:num_clips].detach().cpu()

    # 2. Undo the ImageNet normalization used by every native model.
    mean, std = dt.dataset_stats.ImageNet.values()
    mean = torch.tensor(mean).view(1, 1, 3, 1, 1)
    std = torch.tensor(std).view(1, 1, 3, 1, 1)
    original = (original * std + mean).clamp(0, 1)
    decoded = (decoded * std + mean).clamp(0, 1)

    # 3. W&B Video accepts a batch of time-major channel-first clips.
    return (original * 255).byte().numpy(), (decoded * 255).byte().numpy()


def add_decoder_diagnostic(self, output, batch, stage, cfg):
    """Train and log the detached diagnostic decoder for global features."""
    if not cfg.decoder.enabled:
        return

    # 1. Decode the exact projected CLS representation used by the predictor.
    pixels = batch['pixels']
    flat_emb = rearrange(output['emb'], 'b t d -> (b t) d')
    decoder_input = flat_emb.detach() if cfg.decoder.detach else flat_emb
    decoded = self.decoder(decoder_input)
    target = rearrange(pixels, 'b t c h w -> (b t) c h w')

    # 2. A separate optimizer trains the diagnostic head. Detaching above means
    # this loss cannot change any encoder, projector, or predictor parameter.
    output['decoder_loss'] = torch.nn.functional.mse_loss(decoded, target)
    output['loss'] = output['loss'] + (
        cfg.decoder.loss_weight * output['decoder_loss']
    )

    # 3. Retain the last deterministic validation batch for one reconstruction
    # per epoch, and preserve periodic training clips for continuity.
    is_validation_batch = stage == 'validate' and self.trainer.is_last_batch
    should_log = (
        cfg.wandb.enabled
        and stage == 'fit'
        and self.global_step % cfg.decoder.log_every == 0
        and self.global_rank == 0
    )
    if is_validation_batch or should_log:
        media = _decoder_videos(pixels, decoded, cfg.decoder.num_clips)
    if is_validation_batch:
        self._epoch_decoder_media = media
    if should_log:
        wandb.log(
            {
                'fit/original_clip': wandb.Video(media[0]),
                'fit/decoded_clip': wandb.Video(media[1]),
            },
            step=self.global_step,
        )


def log_losses(self, output, stage):
    """Log every scalar objective term through Lightning."""
    losses = {
        f'{stage}/{name}': value.detach()
        for name, value in output.items()
        if name.endswith('loss')
    }
    self.log_dict(losses, on_step=True, sync_dist=True)


def train_global_jepa(cfg, *, forward, objective_factory):
    """Run one native global-CLS model with shared data and diagnostics."""
    # 1. Load the resolved offline corpus and reproduce the baseline transforms.
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    cache_dir = os.environ.get('LOCAL_DATASET_DIR')
    dataset = swm.data.load_dataset(
        dataset_name,
        transform=None,
        cache_dir=cache_dir,
        **dataset_cfg,
    )
    transforms = [get_img_preprocessor('pixels', 'pixels', cfg.img_size)]

    with open_dict(cfg):
        if cfg.feature_mode.aggregation != 'cls':
            cfg.feature_mode.input_dim = cfg.embed_dim
            cfg.feature_mode.output_dim = cfg.embed_dim
            if cfg.feature_mode.aggregation == 'mlp':
                cfg.feature_mode.num_patches = (
                    cfg.img_size // cfg.patch_size
                ) ** 2
        for column in cfg.data.dataset.keys_to_load:
            if not column.startswith('pixels'):
                transforms.append(
                    get_column_normalizer(dataset, column, column)
                )
        action_dim = cfg.data.dataset.frameskip * dataset.get_dim('action')
        cfg.model.action_encoder.input_dim = action_dim
        if 'idm' in cfg:
            cfg.idm.input_dim = 2 * cfg.embed_dim
            cfg.idm.output_dim = action_dim
    dataset.transform = spt.data.transforms.Compose(*transforms)

    # 2. Use one seeded, episode-agnostic split and ordinary shuffled clips for
    # every family so model objective is the only baseline difference.
    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=generator,
    )
    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        generator=generator,
    )
    val_options = {**cfg.loader, 'shuffle': False, 'drop_last': False}
    val = torch.utils.data.DataLoader(val_set, **val_options)

    # 3. Instantiate the native model/objective modules and one detached global
    # decoder. Only modules with trainable parameters receive optimizers.
    modules = {'model': hydra.utils.instantiate(cfg.model)}
    modules.update(objective_factory(cfg))
    if cfg.decoder.enabled:
        modules['decoder'] = GlobalDecoder(
            global_dim=cfg.embed_dim,
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            **cfg.decoder.kwargs,
        )

    total_steps = cfg.trainer.max_epochs * len(train)
    optimizers = {}
    for name, module in modules.items():
        if not any(
            parameter.requires_grad for parameter in module.parameters()
        ):
            continue
        optimizers[f'{name}_opt'] = {
            'modules': name,
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': 'epoch',
        }

    training_module = spt.Module(
        **modules,
        forward=forward,
        optim=optimizers,
    )

    # 4. Record the exact resolved config, then train with portable checkpoints
    # and a final decoder-media callback.
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'),
        cfg.get('subdir') or '',
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / 'config.yaml')

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    callbacks = [
        SaveCkptCallback(
            cfg.output_model_name,
            cfg,
            epoch_interval=cfg.checkpoint_every,
        ),
        EpochDecoderMediaCallback(commit=not cfg.epoch_probe.enabled),
    ]
    if cfg.epoch_probe.enabled:
        callbacks.append(EpochProbeCallback(cfg, run_dir))
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )
    checkpoint = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    manager = spt.Manager(
        trainer=trainer,
        module=training_module,
        data=spt.data.DataModule(train=train, val=val),
        ckpt_path=checkpoint if checkpoint.exists() else None,
    )
    manager()


__all__ = [
    'add_decoder_diagnostic',
    'log_losses',
    'train_global_jepa',
]
