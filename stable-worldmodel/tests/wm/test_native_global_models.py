"""Native LeWM architecture and objective regression tests."""

import json
import hashlib
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn
from PIL import Image

from stable_worldmodel.wm.lewm import LeWM
from stable_worldmodel.wm.decoder import GlobalDecoder
from stable_worldmodel.wm.feature_modes import FeatureMode


TRAIN_DIR = Path(__file__).resolve().parents[2] / 'scripts' / 'train'
CONFIG_DIR = TRAIN_DIR / 'config'
sys.path.insert(0, str(TRAIN_DIR))

import _global_jepa as global_jepa  # noqa: E402
from _global_jepa import add_decoder_diagnostic, train_global_jepa  # noqa: E402


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


def test_lewm_selects_actual_cls_token():
    # The CLS token is intentionally unlike every patch and their mean.
    model = LeWM(
        encoder=_TokenEncoder(),
        predictor=nn.Identity(),
        action_encoder=_ActionEncoder(),
    )
    batch = {
        'pixels': torch.zeros(2, 3, 3, 4, 4),
        'action': torch.zeros(2, 3, 1),
    }

    # Native encoding must return token zero exactly, without aggregation.
    encoded = model.encode(batch)['emb']
    expected = torch.tensor([7.0, 11.0]).expand(2, 3, -1)
    torch.testing.assert_close(encoded, expected)


def test_lewm_uses_explicit_feature_readout():
    feature_mode = FeatureMode(
        mode='global',
        aggregation='mean',
        input_dim=2,
        output_dim=2,
    )
    model = LeWM(
        encoder=_TokenEncoder(),
        predictor=nn.Identity(),
        action_encoder=_ActionEncoder(),
        feature_mode=feature_mode,
    )
    batch = {
        'pixels': torch.zeros(2, 3, 3, 4, 4),
        'action': torch.zeros(2, 3, 1),
    }

    encoded = model.encode(batch)['emb']
    expected = torch.tensor([156.0, 163.0]).expand(2, 3, -1)
    torch.testing.assert_close(encoded, expected)


def test_spatial_8x8_readout_preserves_shape_and_rejects_non_square_grid():
    mode = FeatureMode(
        mode='global',
        aggregation='spatial_8x8',
        input_dim=3,
        output_dim=5,
    )
    assert mode(torch.randn(4, 1 + 16 * 16, 3)).shape == (4, 5)
    with pytest.raises(ValueError, match='square patch grid'):
        mode(torch.randn(4, 1 + 255, 3))


def test_native_configs_share_one_cls_and_decoder_contract():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        configs = {
            name: compose(config_name=name) for name in ('lewm', 'lewm_idm')
        }
    for cfg in configs.values():
        assert cfg.model.encoder._target_.endswith('vit_hf')
        assert cfg.feature_mode.aggregation == 'cls'
        assert cfg.decoder.enabled and cfg.decoder.detach
        assert not cfg.wandb.enabled
        assert cfg.subdir == cfg.output_model_name
    assert 'idm' not in configs['lewm']
    assert configs['lewm_idm'].loss.idm.weight == pytest.approx(3.125)


@pytest.mark.parametrize('model_name', ['lewm', 'lewm_idm'])
@pytest.mark.parametrize(
    ('data_name', 'filename'),
    [
        (
            'tunnel_room_two_color',
            'tunnel_room_color_filled18x18.h5',
        ),
        ('tunnel_room_two_shape', 'tunnel_room_shape.h5'),
        ('tunnel_room_two_green_blue', 'tunnel_room_palette.h5'),
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


def test_training_rejects_unscoped_wandb_resume_sidecar(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path('wandb_resume.json').write_text('{"id": "unrelated-run"}')

    with pytest.raises(RuntimeError, match='override this run ID'):
        train_global_jepa(None, forward=None, objective_factory=None)


@pytest.fixture
def seeded_training_runner(tmp_path, monkeypatch):
    """Exercise the training lifecycle without loading an experiment corpus."""
    python_rng_state = random.getstate()
    numpy_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state()
    monkeypatch.delenv('PL_GLOBAL_SEED', raising=False)
    monkeypatch.delenv('PL_SEED_WORKERS', raising=False)

    active_run = []
    real_seed_everything = global_jepa.pl.seed_everything

    class _Dataset:
        transform = None

        @staticmethod
        def get_dim(column):
            assert column == 'action'
            return 1

    class _Loader:
        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def __len__():
            return 1

    class _Manager:
        def __init__(self, **kwargs):
            active_run[-1]['events'].append(('manager', kwargs['seed']))
            active_run[-1]['manager_seed'] = kwargs['seed']

        def __call__(self):
            active_run[-1]['events'].append(('manager_call', None))

    def seed_everything(seed, *, workers):
        active_run[-1]['events'].append(('seed', seed))
        return real_seed_everything(seed, workers=workers, verbose=False)

    def load_dataset(*_args, **_kwargs):
        active_run[-1]['events'].append(('dataset', None))
        return _Dataset()

    def instantiate(_cfg):
        active_run[-1]['events'].append(('model', None))
        model = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))
        parameter_bytes = b''.join(
            parameter.detach().cpu().numpy().tobytes()
            for parameter in model.parameters()
        )
        active_run[-1]['parameter_hash'] = hashlib.sha256(
            parameter_bytes
        ).hexdigest()
        return model

    def trainer(**_kwargs):
        active_run[-1]['events'].append(('trainer', None))
        return object()

    monkeypatch.setattr(global_jepa.pl, 'seed_everything', seed_everything)
    monkeypatch.setattr(global_jepa.swm.data, 'load_dataset', load_dataset)
    monkeypatch.setattr(
        global_jepa,
        'get_img_preprocessor',
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        global_jepa.spt.data.transforms,
        'Compose',
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        global_jepa.spt.data,
        'random_split',
        lambda *_args, **_kwargs: (object(), object()),
    )
    monkeypatch.setattr(global_jepa.torch.utils.data, 'DataLoader', _Loader)
    monkeypatch.setattr(global_jepa.hydra.utils, 'instantiate', instantiate)
    monkeypatch.setattr(
        global_jepa.spt,
        'Module',
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        global_jepa.swm.data.utils,
        'get_cache_dir',
        lambda **_kwargs: tmp_path,
    )
    monkeypatch.setattr(global_jepa.pl, 'Trainer', trainer)
    monkeypatch.setattr(
        global_jepa.spt.data,
        'DataModule',
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(global_jepa.spt, 'Manager', _Manager)

    def run(seed):
        record = {'events': []}
        active_run.append(record)
        cfg = OmegaConf.create(
            {
                'seed': str(seed),
                'data': {
                    'dataset': {
                        'name': 'seed-test',
                        'frameskip': 1,
                        'keys_to_load': ['pixels'],
                    }
                },
                'img_size': 8,
                'patch_size': 4,
                'embed_dim': 8,
                'feature_mode': {'aggregation': 'cls'},
                'model': {'action_encoder': {'input_dim': 0}},
                'decoder': {'enabled': False},
                'train_split': 0.8,
                'loader': {},
                'trainer': {'max_epochs': 1},
                'optimizer': {},
                'subdir': f'seed-{seed}',
                'wandb': {'enabled': False},
                'output_model_name': 'seed-test',
                'checkpoint_every': 1,
                'epoch_probe': {'enabled': False},
            }
        )
        try:
            train_global_jepa(
                cfg,
                forward=None,
                objective_factory=lambda _cfg: {},
            )
        finally:
            active_run.pop()
        return record

    yield run

    random.setstate(python_rng_state)
    np.random.set_state(numpy_rng_state)
    torch.random.set_rng_state(torch_rng_state)


def test_global_training_seed_controls_initial_parameter_hash(
    seeded_training_runner,
):
    first = seeded_training_runner(713)['parameter_hash']
    repeated = seeded_training_runner(713)['parameter_hash']
    changed = seeded_training_runner(714)['parameter_hash']

    assert first == repeated
    assert first != changed


def test_global_training_seeds_before_construction_and_seeds_manager(
    seeded_training_runner,
):
    result = seeded_training_runner(713)
    event_names = [name for name, _value in result['events']]

    assert event_names.index('seed') < event_names.index('dataset')
    assert event_names.index('seed') < event_names.index('model')
    assert event_names.index('seed') < event_names.index('trainer')
    assert result['events'][0] == ('seed', 713)
    assert result['manager_seed'] == 713
    assert type(result['manager_seed']) is int


def test_detached_global_decoder_cannot_change_model_features():
    # Build a tiny decoder and a projected CLS trajectory with gradients.
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

    # Decoder parameters learn, while detached latent gradients remain zero.
    add_decoder_diagnostic(module, output, batch, 'fit', cfg)
    output['loss'].backward()

    assert torch.count_nonzero(emb.grad) == 0
    assert all(
        parameter.grad is not None for parameter in module.decoder.parameters()
    )


def test_epoch_decoder_media_is_lossless_and_local_without_wandb(
    tmp_path, monkeypatch
):
    original = np.arange(6 * 4 * 3 * 8 * 8, dtype=np.uint8).reshape(
        6, 4, 3, 8, 8
    )
    decoded = np.full_like(original, 127)
    module = SimpleNamespace(_epoch_decoder_media=(original, decoded))
    trainer = SimpleNamespace(
        is_global_zero=True,
        sanity_checking=False,
        current_epoch=9,
        global_step=123,
    )
    monkeypatch.setattr(global_jepa.wandb, 'run', None)

    def unexpected_log(*args, **kwargs):
        raise AssertionError('Local decoder output must not require W&B.')

    monkeypatch.setattr(global_jepa.wandb, 'log', unexpected_log)
    rng = torch.get_rng_state().clone()
    callback = global_jepa.EpochDecoderMediaCallback(tmp_path)
    callback.on_validation_epoch_end(trainer, module)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    folder = tmp_path / 'epoch_010'
    assert len(list(folder.glob('*.png'))) == 8
    for name, video in [('original', original), ('decoded', decoded)]:
        for frame in range(4):
            image = np.asarray(
                Image.open(folder / f'{name}_frame_{frame:02d}.png')
            )
            assert image.shape == (16, 24, 3)
            for clip in range(6):
                row, column = divmod(clip, 3)
                np.testing.assert_array_equal(
                    image[
                        row * 8 : (row + 1) * 8, column * 8 : (column + 1) * 8
                    ],
                    video[clip, frame].transpose(1, 2, 0),
                )
    metadata = json.loads((folder / 'metadata.json').read_text())
    assert metadata['epoch'] == 10 and metadata['global_step'] == 123
    assert metadata['frame_indexing'] == 'zero-based'


def test_checkpoint_callback_uses_the_configured_run_directory(monkeypatch):
    saved = []
    monkeypatch.setattr(
        global_jepa,
        'save_pretrained',
        lambda *args, **kwargs: saved.append(kwargs),
    )
    callback = global_jepa.SaveCkptCallback(
        'color-spatial-seed3072', {}, epoch_interval=5
    )
    trainer = SimpleNamespace(
        current_epoch=9, max_epochs=10, is_global_zero=True
    )
    callback.on_train_epoch_end(
        trainer, SimpleNamespace(model=nn.Linear(2, 2))
    )
    assert saved[0]['run_name'] == 'color-spatial-seed3072'
    assert saved[0]['filename'] == 'weights_epoch_10.pt'
