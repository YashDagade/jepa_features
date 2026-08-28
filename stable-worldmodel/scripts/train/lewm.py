"""Train the native global-CLS LeWM objective."""

from functools import partial

import hydra
import torch

from stable_worldmodel.wm.loss import SIGReg

from _global_jepa import (
    add_decoder_diagnostic,
    log_losses,
    train_global_jepa,
)


def lejepa_forward(self, batch, stage, cfg):
    """Predict future CLS states and apply LeWM's SIGReg objective."""
    # 1. Encode the clip and align each prediction with the next CLS state.
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)
    output = self.model.encode(batch)
    emb = output['emb']
    predicted = self.model.predict(
        emb[:, : cfg.wm.history_size],
        output['act_emb'][:, : cfg.wm.history_size],
    )
    target = emb[:, cfg.wm.num_preds :]

    # 2. Preserve the published LeWM objective exactly: latent prediction plus
    # SIGReg over the batch cloud independently at each timestep.
    output['pred_loss'] = (predicted - target).square().mean()
    output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
    output['loss'] = (
        output['pred_loss'] + cfg.loss.sigreg.weight * output['sigreg_loss']
    )

    # 3. Train the detached diagnostic decoder without changing model features.
    add_decoder_diagnostic(self, output, batch, stage, cfg)
    log_losses(self, output, stage)
    return output


@hydra.main(version_base=None, config_path='./config', config_name='lewm')
def run(cfg):
    train_global_jepa(
        cfg,
        forward=partial(lejepa_forward, cfg=cfg),
        objective_factory=lambda current: {
            'sigreg': SIGReg(**current.loss.sigreg.kwargs)
        },
    )


if __name__ == '__main__':
    run()
