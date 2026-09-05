import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from einops import EinopsError
from omegaconf import OmegaConf

from stable_worldmodel.wm.decoder import GlobalDecoder

_REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.train._global_jepa import add_decoder_diagnostic  # noqa: E402


def _small_decoder() -> GlobalDecoder:
    """Construct the cheapest global decoder that preserves the public API."""
    return GlobalDecoder(
        global_dim=32,
        img_size=28,
        patch_size=7,
        dim=16,
        heads=4,
        depth=1,
    )


def _diagnostic_cfg(*, detach: bool):
    """Return only the settings consumed by the shared diagnostic helper."""
    return OmegaConf.create(
        {
            'decoder': {
                'enabled': True,
                'detach': detach,
                'loss_weight': 0.25,
                'num_clips': 1,
                'log_every': 100,
            },
            'wandb': {'enabled': False},
        }
    )


def test_global_decoder_reconstructs_expected_image_shape():
    decoder = _small_decoder()

    output = decoder(torch.randn(2, 32))

    assert output.shape == (2, 3, 28, 28)


def test_detached_decoder_diagnostic_trains_only_the_head():
    module = SimpleNamespace(
        decoder=_small_decoder(), trainer=SimpleNamespace(is_last_batch=False)
    )
    embedding = torch.randn(2, 3, 32, requires_grad=True)
    output = {'emb': embedding, 'loss': embedding.new_zeros(())}
    batch = {'pixels': torch.randn(2, 3, 3, 28, 28)}

    # Decode the projected history through the training helper.
    add_decoder_diagnostic(
        module,
        output,
        batch,
        stage='validate',
        cfg=_diagnostic_cfg(detach=True),
    )

    # The diagnostic contributes at its configured weight.
    torch.testing.assert_close(output['loss'], 0.25 * output['decoder_loss'])
    output['loss'].backward()

    # Detachment protects the representation while training the decoder.
    assert embedding.grad is None
    assert any(
        parameter.grad is not None for parameter in module.decoder.parameters()
    )


def test_decoder_diagnostic_rejects_a_token_axis():
    module = type('Module', (), {'decoder': _small_decoder()})()
    output = {
        'emb': torch.randn(2, 3, 4, 32),
        'loss': torch.tensor(0.0),
    }
    batch = {'pixels': torch.randn(2, 3, 3, 28, 28)}

    # The decoder consumes one projected vector per frame.
    with pytest.raises(EinopsError):
        add_decoder_diagnostic(
            module,
            output,
            batch,
            stage='validate',
            cfg=_diagnostic_cfg(detach=True),
        )
