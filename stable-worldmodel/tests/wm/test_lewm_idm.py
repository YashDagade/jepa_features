"""Parity and gradient checks for the explicitly adapted LeWM + IDM recipe."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

from stable_worldmodel.wm.lewm.module import MLP
from stable_worldmodel.wm.loss import SIGReg

TRAIN_DIR = Path(__file__).resolve().parents[2] / 'scripts' / 'train'
sys.path.insert(0, str(TRAIN_DIR))

from lewm import lejepa_forward  # noqa: E402
from lewm_idm import build_objectives, lewm_idm_forward  # noqa: E402


class _SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.states = nn.Parameter(torch.randn(2, 4, 192))
        self.predictor = nn.Linear(192, 192)

    def encode(self, batch):
        return {'emb': self.states, 'act_emb': batch['action']}

    def predict(self, states, actions):
        return self.predictor(states)


def _module_and_config(weight):
    module = SimpleNamespace(
        model=_SmallModel(),
        idm=MLP(input_dim=384, hidden_dim=512, output_dim=10),
        sigreg=lambda states: states.square().mean(),
        log_dict=lambda *args, **kwargs: None,
    )
    cfg = OmegaConf.create(
        {
            'wm': {'history_size': 3, 'num_preds': 1},
            'loss': {'sigreg': {'weight': 0.09}, 'idm': {'weight': weight}},
            'decoder': {'enabled': False},
        }
    )
    return module, cfg


def test_zero_idm_weight_preserves_native_loss_and_shared_gradients():
    adapted, cfg = _module_and_config(0.0)
    native = copy.deepcopy(adapted)
    batch = {'action': torch.randn(2, 4, 10)}
    expected = lejepa_forward(native, copy.deepcopy(batch), 'fit', cfg)
    actual = lewm_idm_forward(adapted, copy.deepcopy(batch), 'fit', cfg)

    torch.testing.assert_close(actual['loss'], expected['loss'])
    expected['loss'].backward()
    actual['loss'].backward()
    for original, modified in zip(
        native.model.parameters(), adapted.model.parameters(), strict=True
    ):
        torch.testing.assert_close(original.grad, modified.grad)
    assert all(
        torch.count_nonzero(p.grad) == 0 for p in adapted.idm.parameters()
    )


def test_historical_idm_order_full_action_block_and_nondetached_states():
    module, cfg = _module_and_config(3.125)
    actions = torch.randn(2, 4, 10, requires_grad=True)
    output = lewm_idm_forward(module, {'action': actions}, 'fit', cfg)
    states = module.model.states
    historical_pairs = torch.cat([states[:, 1:], states[:, :-1]], dim=-1)
    historical_loss = nn.functional.mse_loss(
        module.idm(historical_pairs), actions[:, :-1].detach()
    )
    torch.testing.assert_close(output['idm_loss'], historical_loss)
    torch.testing.assert_close(
        output['loss'],
        output['pred_loss']
        + 0.09 * output['sigreg_loss']
        + 3.125 * historical_loss,
    )
    output['idm_loss'].backward()
    assert all(
        torch.count_nonzero(states.grad[:, step]) > 0 for step in range(4)
    )
    assert all(p.grad is not None for p in module.idm.parameters())
    assert actions.grad is None


def test_adaptation_config_matches_historical_head_and_preserves_rng():
    with initialize_config_dir(
        config_dir=str(TRAIN_DIR / 'config'), version_base=None
    ):
        native = compose(config_name='lewm')
        adapted = compose(config_name='lewm_idm')
    assert 'idm' not in native
    assert adapted.model == native.model
    assert adapted.optimizer == native.optimizer
    assert adapted.decoder == native.decoder
    adapted.idm.input_dim = 384
    adapted.idm.output_dim = 10

    torch.manual_seed(3072)
    SIGReg(**native.loss.sigreg.kwargs)
    expected_next_random = torch.rand(8)
    torch.manual_seed(3072)
    head = build_objectives(adapted)['idm']
    torch.testing.assert_close(torch.rand(8), expected_next_random)
    assert [type(layer) for layer in head.net] == [
        nn.Linear,
        nn.LayerNorm,
        nn.GELU,
        nn.Linear,
    ]
    assert sum(p.numel() for p in head.parameters()) == 203274

    adapted.loss.idm.weight = float('nan')
    with pytest.raises(ValueError, match='finite and nonnegative'):
        build_objectives(adapted)
