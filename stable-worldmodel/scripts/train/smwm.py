"""Train native global-CLS Sensorimotor World Model."""

from functools import partial

import hydra
import torch
import torch.nn.functional as F

from _global_jepa import (
    add_decoder_diagnostic,
    log_losses,
    train_global_jepa,
)


def sensorimotor_forward(self, batch, stage, cfg):
    """Apply SMWM's forward-dynamics plus inverse-dynamics objective."""
    # 1. Encode and predict through the same actual-CLS causal path as LeWM.
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)
    output = self.model.encode(batch)
    emb = output['emb']
    predicted = self.model.predict(
        emb[:, : cfg.wm.history_size],
        output['act_emb'][:, : cfg.wm.history_size],
    )
    target = emb[:, cfg.wm.num_preds :]

    # 2. Preserve the official non-detached forward-dynamics loss.
    output['fdm_loss'] = (predicted - target).square().mean()

    # 3. Predict each action from the ordered [z_t,z_t+1] pair. This 0.1 IDM
    # term is SMWM's defining anti-collapse objective, not the LeWM add-on.
    state_pairs = torch.cat([emb[:, :-1], emb[:, 1:]], dim=-1)
    action_target = batch['action'][:, :-1].detach()
    output['idm_loss'] = F.mse_loss(self.idm(state_pairs), action_target)
    output['loss'] = (
        output['fdm_loss'] + cfg.loss.idm.weight * output['idm_loss']
    )

    # 4. Train the detached diagnostic decoder without changing model features.
    add_decoder_diagnostic(self, output, batch, stage, cfg)
    log_losses(self, output, stage)
    return output


@hydra.main(version_base=None, config_path='./config', config_name='smwm')
def run(cfg):
    if not cfg.loss.idm.enabled:
        raise ValueError('SMWM requires loss.idm.enabled=true.')
    train_global_jepa(
        cfg,
        forward=partial(sensorimotor_forward, cfg=cfg),
        objective_factory=lambda current: {
            'idm': hydra.utils.instantiate(current.idm)
        },
    )


if __name__ == '__main__':
    run()
