"""Train the historical LeWM + IDM adaptation without changing native LeWM."""

import math
from functools import partial

import hydra
import torch
import torch.nn.functional as F

from stable_worldmodel.wm.loss import SIGReg

from _global_jepa import (
    add_decoder_diagnostic,
    log_losses,
    train_global_jepa,
)


def lewm_idm_forward(self, batch, stage, cfg):
    """Add historical next/current inverse dynamics to native LeWM losses."""
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)
    output = self.model.encode(batch)
    emb = output['emb']
    predicted = self.model.predict(
        emb[:, : cfg.wm.history_size],
        output['act_emb'][:, : cfg.wm.history_size],
    )
    target = emb[:, cfg.wm.num_preds :]
    output['pred_loss'] = (predicted - target).square().mean()
    output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))

    # Match the historical head's order [z_(t+1), z_t] and all five normalized
    # primitive actions; neither representation branch is detached.
    action_target = batch['action'][:, :-1].detach()
    if emb.ndim != 3 or emb.shape[1] < 2:
        raise ValueError(
            'LeWM + IDM requires (batch, time >= 2, width) states.'
        )
    if batch['action'].shape[:2] != emb.shape[:2]:
        raise ValueError('IDM states and actions must share batch/time axes.')
    action_prediction = self.idm(torch.cat([emb[:, 1:], emb[:, :-1]], dim=-1))
    if action_prediction.shape != action_target.shape:
        raise ValueError(
            'IDM predictions must match the full action-block shape.'
        )
    output['idm_loss'] = F.mse_loss(action_prediction, action_target)
    output['loss'] = (
        output['pred_loss']
        + cfg.loss.sigreg.weight * output['sigreg_loss']
        + cfg.loss.idm.weight * output['idm_loss']
    )
    add_decoder_diagnostic(self, output, batch, stage, cfg)
    log_losses(self, output, stage)
    return output


def build_objectives(cfg):
    """Keep shared decoder initialization unchanged when adding the IDM head."""
    weight = float(cfg.loss.idm.weight)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError('LeWM + IDM weight must be finite and nonnegative.')
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs)
    with torch.random.fork_rng(devices=[]):
        idm = hydra.utils.instantiate(cfg.idm)
    return {'sigreg': sigreg, 'idm': idm}


@hydra.main(version_base=None, config_path='./config', config_name='lewm_idm')
def run(cfg):
    train_global_jepa(
        cfg,
        forward=partial(lewm_idm_forward, cfg=cfg),
        objective_factory=build_objectives,
    )


if __name__ == '__main__':
    run()
