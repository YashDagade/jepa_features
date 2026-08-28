"""Train native global-CLS LpWM with rectified Product-Laplace InfoReg."""

from functools import partial

import hydra
import torch

from stable_worldmodel.wm.loss import InfoReg

from _global_jepa import (
    add_decoder_diagnostic,
    log_losses,
    train_global_jepa,
)


def lpjepa_forward(self, batch, stage, cfg):
    """Predict future CLS states and apply LpWM's InfoReg objective."""
    # 1. Encode the clip and align each prediction with the next CLS state.
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)
    output = self.model.encode(batch)
    emb = output['emb']
    predicted = self.model.predict(
        emb[:, : cfg.wm.history_size],
        output['act_emb'][:, : cfg.wm.history_size],
    )
    target = emb[:, cfg.wm.num_preds :]

    # 2. Match the batch of rectified global states independently per timestep.
    output['pred_loss'] = (predicted - target).square().mean()
    output['inforeg_loss'] = self.inforeg(emb.transpose(0, 1))
    output['loss'] = (
        output['pred_loss'] + cfg.loss.inforeg.weight * output['inforeg_loss']
    )

    # 3. Train the detached diagnostic decoder without changing model features.
    add_decoder_diagnostic(self, output, batch, stage, cfg)
    log_losses(self, output, stage)
    return output


@hydra.main(version_base=None, config_path='./config', config_name='lpwm')
def run(cfg):
    train_global_jepa(
        cfg,
        forward=partial(lpjepa_forward, cfg=cfg),
        objective_factory=lambda current: {
            'inforeg': InfoReg(**current.loss.inforeg.kwargs)
        },
    )


if __name__ == '__main__':
    run()
