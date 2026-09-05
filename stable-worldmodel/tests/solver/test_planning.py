"""CEM/GD contracts for full action-block plans and warm-start placement."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from stable_worldmodel.solver import CEMSolver, GradientSolver


class QuadraticCost(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    def get_cost(self, info, actions):
        return (actions - 0.25).square().sum(dim=(-1, -2))


def configure(solver):
    solver.configure(
        action_space=gym.spaces.Box(-1, 1, shape=(1, 2), dtype=np.float32),
        n_envs=1,
        config=SimpleNamespace(horizon=5, action_block=5),
    )
    return solver


@pytest.mark.parametrize('kind', ['cem', 'gd'])
def test_solver_finds_lower_cost_action_blocks(kind):
    options = dict(
        model=QuadraticCost(), n_steps=30, num_samples=100, seed=3072
    )
    if kind == 'cem':
        solver = CEMSolver(**options, topk=10)
    else:
        solver = GradientSolver(
            **options,
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={'lr': 0.1},
        )
    configure(solver)
    result = solver({'pixels': torch.zeros(1, 1, 1)})['actions']
    assert result.shape == (1, 5, 10)
    assert torch.isfinite(result).all()
    assert (result - 0.25).square().mean() < 0.02


def test_gd_complete_warm_start_converts_dtype():
    solver = configure(
        GradientSolver(QuadraticCost(), n_steps=1, num_samples=2)
    )
    solver.init_action(1, torch.zeros(1, 5, 10, dtype=torch.float64))
    assert solver.init.dtype == torch.float32
    assert solver.init.shape == (1, 2, 5, 10)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA is not available'
)
def test_gd_complete_cpu_warm_start_moves_to_cuda():
    solver = configure(
        GradientSolver(
            QuadraticCost().cuda(), n_steps=1, num_samples=2, device='cuda'
        )
    )
    solver.init_action(1, torch.zeros(1, 5, 10))
    assert solver.init.device.type == 'cuda'
