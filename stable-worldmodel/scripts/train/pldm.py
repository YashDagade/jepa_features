"""Train the native global-CLS PLDM objective."""

from functools import partial

import hydra
import torch

from stable_worldmodel.wm.loss import PLDMLoss

from _global_jepa import (
    add_decoder_diagnostic,
    log_losses,
    train_global_jepa,
)


def pldm_forward(self, batch, stage, cfg):
    """Predict future CLS states and apply PLDM's published regularizers."""
    # 1. Encode the clip and align the causal prediction with its next state.
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)
    output = self.model.encode(batch)
    emb = output['emb']
    predicted = self.model.predict(
        emb[:, : cfg.wm.history_size],
        output['act_emb'][:, : cfg.wm.history_size],
    )
    target = emb[:, cfg.wm.num_preds :]

    # 2. Evaluate prediction, temporal alignment, and variance/covariance
    # terms. Historical disabled/zero-weight auxiliary paths are omitted.
    output['pred_loss'] = (predicted - target).square().mean()
    output.update(self.pldm(emb))
    output['loss'] = output['pred_loss']
    for name, settings in cfg.loss.items():
        loss_name = f'{name}_loss'
        if settings.enabled and loss_name in output:
            output['loss'] += settings.weight * output[loss_name]

    # 3. Train the detached diagnostic decoder without changing model features.
    add_decoder_diagnostic(self, output, batch, stage, cfg)
    log_losses(self, output, stage)
    return output


@hydra.main(version_base=None, config_path='./config', config_name='pldm')
def run(cfg):
    train_global_jepa(
        cfg,
        forward=partial(pldm_forward, cfg=cfg),
        objective_factory=lambda _current: {
            'pldm': PLDMLoss(),
        },
    )


if __name__ == '__main__':
    run()
