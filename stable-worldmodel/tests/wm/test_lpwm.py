from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from torch import nn

from stable_worldmodel.wm.lewm import LeWM
from stable_worldmodel.wm.loss import InfoReg
from stable_worldmodel.wm.lpwm import LpWM
from stable_worldmodel.wm.lpwm.module import MLP, RepReLU


CONFIG_DIR = (
    Path(__file__).resolve().parents[2] / 'scripts' / 'train' / 'config'
)


def test_lpwm_uses_spawned_data_workers():
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR),
    ):
        cfg = compose(config_name='lpwm')

    assert cfg.loader.num_workers > 0
    assert cfg.loader.multiprocessing_context == 'spawn'


def test_lpwm_reuses_the_maintained_lewm_runtime_contract():
    model = LpWM(
        encoder=nn.Linear(2, 2),
        predictor=nn.Identity(),
        action_encoder=nn.Identity(),
    )

    assert isinstance(model, LeWM)
    assert callable(model.encode)
    assert callable(model.get_cost)


def test_rep_relu_has_relu_values_and_dense_surrogate_gradients():
    x = torch.tensor([-2.0, 1.0], requires_grad=True)

    output = RepReLU()(x)
    output.sum().backward()

    assert torch.equal(output, torch.tensor([0.0, 1.0]))
    assert torch.all(x.grad != 0)


def test_lpwm_mlp_applies_exact_rectified_output():
    projector = MLP(
        input_dim=4,
        hidden_dim=8,
        output_dim=3,
        norm_fn=None,
        link_function_type='ReLU',
    )

    output = projector(torch.randn(5, 4))

    assert output.shape == (5, 3)
    assert torch.all(output >= 0)


def test_inforeg_sliced_wasserstein_is_finite_and_differentiable():
    samples = torch.randn(3, 16, 8, requires_grad=True)
    regularizer = InfoReg(
        mu=0.0,
        p=1.0,
        link_function_type='ReLU',
        num_slices=32,
    )

    loss = regularizer(samples)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert samples.grad is not None
    assert torch.isfinite(samples.grad).all()


def test_inforeg_rejects_unsupported_objective():
    try:
        InfoReg(loss_func_type='epps_pulley')
    except ValueError as error:
        assert 'sliced_wasserstein' in str(error)
    else:
        raise AssertionError('Expected unsupported InfoReg objective to fail.')
