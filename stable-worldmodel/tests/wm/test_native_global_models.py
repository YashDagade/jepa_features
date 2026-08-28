"""Native global-CLS architecture and objective regression tests."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

from stable_worldmodel.wm.lewm import LeWM
from stable_worldmodel.wm.lewm.module import Predictor as LeWMPredictor
from stable_worldmodel.wm.decoder import GlobalDecoder
from stable_worldmodel.wm.lpwm import LpWM
from stable_worldmodel.wm.pldm import PLDM
from stable_worldmodel.wm.pldm.module import Predictor as PLDMPredictor
from stable_worldmodel.wm.smwm import InverseModel, SMWM


TRAIN_DIR = Path(__file__).resolve().parents[2] / 'scripts' / 'train'
CONFIG_DIR = TRAIN_DIR / 'config'
BASELINE_LAUNCHER = (
    Path(__file__).resolve().parents[3]
    / 'sbatch_scripts'
    / 'train'
    / 'baseline_matrix.sh'
)
sys.path.insert(0, str(TRAIN_DIR))

from pldm import pldm_forward  # noqa: E402
from smwm import sensorimotor_forward  # noqa: E402
from _global_jepa import add_decoder_diagnostic  # noqa: E402


class _TokenEncoder(nn.Module):
    """Return fixed, deliberately distinct CLS and patch tokens."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, pixels, **_kwargs):
        batch = pixels.size(0)
        tokens = pixels.new_tensor(
            [[7.0, 11.0], [101.0, 103.0], [211.0, 223.0]]
        )
        return SimpleNamespace(
            last_hidden_state=tokens.unsqueeze(0).expand(batch, -1, -1)
        )


class _ActionEncoder(nn.Module):
    def forward(self, action):
        return action


@pytest.mark.parametrize('model_class', [LeWM, PLDM, LpWM, SMWM])
def test_every_native_model_selects_actual_cls_token(model_class):
    # 1. The CLS token is intentionally unlike every patch and their mean.
    model = model_class(
        encoder=_TokenEncoder(),
        predictor=nn.Identity(),
        action_encoder=_ActionEncoder(),
    )
    batch = {
        'pixels': torch.zeros(2, 3, 3, 4, 4),
        'action': torch.zeros(2, 3, 1),
    }

    # 2. Native encoding must return token zero exactly, without aggregation.
    encoded = model.encode(batch)['emb']
    expected = torch.tensor([7.0, 11.0]).expand(2, 3, -1)
    torch.testing.assert_close(encoded, expected)


def test_pldm_reuses_the_identical_lewm_predictor_implementation():
    assert PLDMPredictor is LeWMPredictor


def test_native_configs_share_one_cls_and_decoder_contract():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        configs = {
            name: compose(config_name=name)
            for name in ('lewm', 'pldm', 'lpwm', 'smwm')
        }

    # 1. Every model instantiates a ViT encoder and the detached global decoder.
    for cfg in configs.values():
        assert cfg.model.encoder._target_.endswith('vit_hf')
        assert cfg.decoder.enabled and cfg.decoder.detach
        assert 'group' in cfg.wandb.config
        assert cfg.wandb.config.resume == 'allow'

    # 2. LeWM/LpWM/PLDM have no IDM module or objective. SMWM retains the
    # paper-defining inverse-dynamics loss at its published weight.
    for name in ('lewm', 'pldm', 'lpwm'):
        assert 'idm' not in configs[name]
    assert configs['smwm'].loss.idm.enabled
    assert configs['smwm'].loss.idm.weight == pytest.approx(0.1)


@pytest.mark.parametrize('model_name', ['lewm', 'pldm', 'lpwm', 'smwm'])
@pytest.mark.parametrize(
    ('data_name', 'filename'),
    [
        (
            'tunnel_room_two_color',
            'tunnel_room_color_noise_filled18x18.h5',
        ),
        ('chaser_two_color', 'chaser_cue_two_color.h5'),
    ],
)
def test_baseline_configs_compose_with_pixels_and_actions_only(
    model_name, data_name, filename
):
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=model_name, overrides=[f'data={data_name}'])

    assert cfg.data.dataset.name.endswith(filename)
    assert list(cfg.data.dataset.keys_to_load) == ['pixels', 'action']
    assert cfg.trainer.max_epochs == 10


def test_baseline_launcher_keeps_trainer_cache_on_scratch(tmp_path):
    # 1. Generate every job without submitting so the test exercises the same
    # shell templates used on the cluster.
    run_root = tmp_path / 'runs'
    subprocess.run(
        ['bash', str(BASELINE_LAUNCHER), '--dry-run'],
        check=True,
        env={**os.environ, 'RUN_ROOT': str(run_root)},
        capture_output=True,
        text=True,
    )

    # 2. Every trainer registry and checkpoint cache must live beside its run
    # outputs on scratch, never under the quota-limited home directory.
    inner_scripts = sorted(run_root.glob('*/train_inner.sh'))
    assert len(inner_scripts) == 8
    for script in inner_scripts:
        contents = script.read_text()
        expected = f'export SPT_CACHE_DIR={script.parent}/spt_cache'
        assert expected in contents
        assert '/home/' not in contents


def test_smwm_inverse_head_matches_official_two_layer_relu_mlp():
    inverse = InverseModel(input_dim=12, hidden_dim=256, output_dim=3)

    assert [type(layer) for layer in inverse.net] == [
        nn.Linear,
        nn.ReLU,
        nn.Linear,
        nn.ReLU,
        nn.Linear,
    ]
    assert inverse(torch.randn(2, 4, 12)).shape == (2, 4, 3)


def test_detached_global_decoder_cannot_change_model_features():
    # 1. Build a tiny decoder and a projected CLS trajectory with gradients.
    emb = torch.randn(2, 3, 4, requires_grad=True)
    module = SimpleNamespace(
        decoder=GlobalDecoder(
            global_dim=4,
            img_size=8,
            patch_size=4,
            dim=8,
            heads=2,
            depth=1,
        ),
        current_epoch=0,
        trainer=SimpleNamespace(max_epochs=2),
        global_step=1,
        global_rank=0,
    )
    output = {'emb': emb, 'loss': emb.sum() * 0.0}
    batch = {'pixels': torch.randn(2, 3, 3, 8, 8)}
    cfg = OmegaConf.create(
        {
            'decoder': {
                'enabled': True,
                'detach': True,
                'loss_weight': 1.0,
                'num_clips': 1,
                'log_every': 100,
            },
            'wandb': {'enabled': False},
        }
    )

    # 2. Decoder parameters learn, while detached latent gradients remain zero.
    add_decoder_diagnostic(module, output, batch, 'fit', cfg)
    output['loss'].backward()

    assert torch.count_nonzero(emb.grad) == 0
    assert all(
        parameter.grad is not None for parameter in module.decoder.parameters()
    )


class _ForwardModel:
    def encode(self, batch):
        batch['emb'] = batch['projected']
        batch['act_emb'] = batch['action']
        return batch

    def predict(self, context, _actions):
        return context


class _LoggerModule:
    def __init__(self):
        self.model = _ForwardModel()
        self.logged = None

    def log_dict(self, values, **_kwargs):
        self.logged = values


def _forward_cfg(loss):
    return OmegaConf.create(
        {
            'wm': {'history_size': 2, 'num_preds': 1},
            'decoder': {'enabled': False},
            'loss': loss,
        }
    )


def test_smwm_forward_is_fdm_plus_weighted_native_idm():
    module = _LoggerModule()
    module.idm = InverseModel(input_dim=4, hidden_dim=8, output_dim=1)
    cfg = _forward_cfg({'idm': {'enabled': True, 'weight': 0.1}})
    batch = {
        'projected': torch.tensor([[[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]]]),
        'action': torch.zeros(1, 3, 1),
    }

    output = sensorimotor_forward(module, batch, 'fit', cfg)

    assert set(name for name in output if name.endswith('loss')) == {
        'fdm_loss',
        'idm_loss',
        'loss',
    }
    torch.testing.assert_close(
        output['loss'],
        output['fdm_loss'] + 0.1 * output['idm_loss'],
    )


def test_pldm_forward_has_no_zero_weight_idm_path():
    module = _LoggerModule()
    module.pldm = lambda _emb: {
        'std_loss': torch.tensor(1.0),
        'std_t_loss': torch.tensor(2.0),
        'cov_loss': torch.tensor(3.0),
        'cov_t_loss': torch.tensor(4.0),
        'temp_align_loss': torch.tensor(5.0),
    }
    cfg = _forward_cfg(
        {
            'std': {'enabled': True, 'weight': 18.0},
            'std_t': {'enabled': True, 'weight': 0.7},
            'cov': {'enabled': True, 'weight': 12.0},
            'cov_t': {'enabled': True, 'weight': 0.0},
            'temp_align': {'enabled': True, 'weight': 0.2},
        }
    )
    batch = {
        'projected': torch.randn(2, 3, 4),
        'action': torch.zeros(2, 3, 1),
    }

    output = pldm_forward(module, batch, 'fit', cfg)

    assert 'idm_loss' not in output
    assert not hasattr(module, 'idm')
