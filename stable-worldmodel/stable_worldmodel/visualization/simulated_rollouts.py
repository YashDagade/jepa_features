"""Portal-conditioned real versus imagined rollout videos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from stable_worldmodel.envs.tunnel_rooms.env import (
    HUB,
    PAIRED_COLOR,
    PAIRED_SHAPE,
)
from stable_worldmodel.plot import save_panel_videos
from stable_worldmodel.wm.decoder import GlobalDecoder
from stable_worldmodel.wm.utils import preserve_module_modes


COLOR_LABELS = ('green', 'orange', 'blue')
SHAPE_LABELS = ('triangle', 'diamond', 'star')
_IDENTITY_LABELS = {
    PAIRED_COLOR: COLOR_LABELS,
    PAIRED_SHAPE: SHAPE_LABELS,
}
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_SEPARATOR = 4
try:
    _FONT = ImageFont.truetype('DejaVuSans.ttf', 13)
except OSError:
    _FONT = ImageFont.load_default()


@dataclass(frozen=True)
class SimulatedRolloutResult:
    video_path: Path
    panel_labels: tuple[str, ...]
    teleport_counts: tuple[int, ...]
    success: tuple[bool, ...]


def ordered_hub_endpoints(env) -> tuple[int, ...]:
    """Order visible cues by identity and hidden portals by vertical position."""
    if env.identity_mode == PAIRED_COLOR:
        return tuple(
            env.cue_assignment.index(identity)
            for identity in range(env.num_leaves)
        )
    if env.identity_mode == PAIRED_SHAPE:
        return tuple(env.hub_endpoint_indices())
    y_positions = env.endpoint_positions[: env.num_leaves, 1]
    return tuple(int(index) for index in np.argsort(y_positions))


def portal_labels(env, endpoints: tuple[int, ...]) -> tuple[str, ...]:
    """Return stable panel labels for the ordered portals."""
    labels = _IDENTITY_LABELS.get(env.identity_mode)
    if labels is not None:
        return labels[: len(endpoints)]
    if env.num_leaves == 1:
        return ('only',)
    labels = (
        ('top', 'bottom')
        if env.num_leaves == 2
        else ('top', 'middle', 'bottom')
    )
    return labels[: len(endpoints)]


def near_portal_positions(
    env, endpoints: tuple[int, ...], *, distance: float
) -> np.ndarray:
    """Place starts in the hub immediately left of the selected portals."""
    if distance <= env.trigger_radius:
        raise ValueError(
            'simulated rollout start_distance must exceed trigger_radius so '
            'the reset itself cannot teleport.'
        )
    starts = env.endpoint_positions[list(endpoints)].copy()
    starts[:, 0] -= float(distance)
    for start in starts:
        if env.room_index(start) != HUB:
            raise ValueError(f'Near-portal start is outside the hub: {start}.')
    return starts.astype(np.float32)


def initialize_portal_starts(
    world,
    *,
    seed: int,
    start_distance: float,
    goal_identity: int = 0,
    scenario: dict[str, Any] | None = None,
) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Reset one scenario and move each vectorized env beside one portal."""
    # 1. Replay one shared scenario so panels differ only by their start portal.
    options = None
    if scenario is not None:
        options = [{'scenario': dict(scenario)} for _ in range(world.num_envs)]
    world.reset(seed=[int(seed)] * world.num_envs, options=options)
    # 2. Order portals consistently and construct safe, near-portal starts.
    reference = world.envs.envs[0].unwrapped
    endpoints = ordered_hub_endpoints(reference)
    if len(endpoints) != world.num_envs:
        raise ValueError(
            f'Expected one env per portal ({len(endpoints)}), got '
            f'{world.num_envs}.'
        )
    starts = near_portal_positions(
        reference, endpoints, distance=float(start_distance)
    )
    # 3. Resolve the requested cue identity to its paired leaf-room goal.
    if reference.identity_mode in _IDENTITY_LABELS:
        hub_endpoint = int(goal_identity)
        if reference.identity_mode == PAIRED_COLOR:
            hub_endpoint = reference.cue_assignment.index(hub_endpoint)
        endpoint = reference._paired_endpoint(hub_endpoint)
        goal_room = endpoint - reference.num_leaves + 1
    else:
        goal_room = (reference.num_leaves + 1) // 2
    bounds = reference._room_bounds(goal_room)
    goal = [bounds[1] - 10.0, bounds[2:4].mean()]
    goal = np.asarray(goal, dtype=np.float32)
    # 4. Give every environment the same goal and its own portal-adjacent start.
    for wrapped, start in zip(world.envs.envs, starts, strict=True):
        env = wrapped.unwrapped
        env._set_goal_state(goal)
        env._set_state(start)

    # 5. Refresh wrapper observations after the direct environment state changes.
    zeros = np.zeros(world.envs.action_space.shape, dtype=np.float32)
    _, world.rewards, world.terminateds, world.truncateds, world.infos = (
        world.envs.step(zeros)
    )
    world.policy.reset()
    return endpoints, portal_labels(reference, endpoints)


def decode_next_prediction(
    model, decoder, info: dict[str, Any], actions: torch.Tensor
) -> np.ndarray:
    """Decode the first future latent along the selected CEM plan."""
    # 1. Copy every numeric rollout input to the model device with a time axis.
    device = next(model.parameters()).device
    rollout_info = {}
    for key, value in info.items():
        if torch.is_tensor(value):
            rollout_info[key] = value.to(device).unsqueeze(1)
        elif isinstance(value, np.ndarray) and value.dtype.kind not in 'USO':
            rollout_info[key] = torch.from_numpy(value).to(device).unsqueeze(1)

    # 2. Roll out the chosen actions and select the first predicted future frame.
    with torch.inference_mode():
        rollout = model.rollout(rollout_info, actions.to(device).unsqueeze(1))
        features = rollout.get('predicted_emb')
        if features is None:
            raise KeyError(
                'World-model rollout has no predicted visual embedding.'
            )
        history = int(rollout_info['pixels'].shape[2])
        prediction = features[:, 0, history]

        # 3. These baselines use one actual CLS-derived vector per frame.
        if not isinstance(decoder, GlobalDecoder):
            raise TypeError(
                'Simulated rollouts require GlobalDecoder, got '
                f'{type(decoder).__name__}.'
            )
        # 4. Reject any token axis instead of silently pooling it.
        if prediction.ndim != 2:
            raise ValueError(
                'GlobalDecoder requires (B, D) rollout features, got '
                f'{tuple(prediction.shape)}.'
            )
        images = decoder(prediction).detach().float().cpu()

    # 5. Undo ImageNet normalization and return display-ready uint8 frames.
    images = (images * _STD + _MEAN).clamp(0.0, 1.0)
    return images.mul(255).to(torch.uint8).permute(0, 2, 3, 1).numpy()


def _stack_frames(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    separator = np.full((_SEPARATOR, top.shape[1], 3), 28, dtype=np.uint8)
    return np.concatenate([top, separator, bottom], axis=0)


def real_imagined_frame(real: np.ndarray, imagined: np.ndarray) -> np.ndarray:
    """Stack real and one-step imagined frames with compact labels."""
    real = np.asarray(real, dtype=np.uint8)
    imagined = np.asarray(imagined, dtype=np.uint8)
    if real.shape != imagined.shape:
        raise ValueError(
            f'Real and imagined frames differ: {real.shape} != {imagined.shape}.'
        )
    image = Image.fromarray(_stack_frames(real, imagined))
    draw = ImageDraw.Draw(image)
    draw.text((6, 5), 'REAL', fill=(20, 20, 20), font=_FONT)
    draw.text(
        (6, real.shape[0] + _SEPARATOR + 5),
        'WM +5',
        fill=(20, 20, 20),
        font=_FONT,
    )
    return np.asarray(image)


def save_simulated_rollout_panel(
    output_dir: Path,
    labels: tuple[str, ...],
    real_frames: list[list[np.ndarray]],
    imagined_frames: list[list[np.ndarray]],
    *,
    fps: int,
) -> Path:
    """Write one real-over-imagined column per portal and return its path."""
    # 1. Check that every portal has one real and one imagined frame stream.
    if not (len(labels) == len(real_frames) == len(imagined_frames)):
        raise ValueError(
            'Panel labels and rollout streams must have equal length.'
        )
    if not labels:
        raise ValueError(
            'Simulated rollout panels require at least one portal.'
        )

    # 2. Pair actual and predicted frames for each portal. The shared goal is
    # intentionally omitted: duplicating it above and below added two useless
    # views. TR2 therefore has four views and TR3 has six views.
    panels = {}
    for label, real, imagined in zip(
        labels, real_frames, imagined_frames, strict=True
    ):
        if len(real) != len(imagined):
            raise ValueError('Real and imagined rollout lengths differ.')
        frames = [
            real_imagined_frame(actual, predicted)
            for actual, predicted in zip(real, imagined, strict=True)
        ]
        panels[f'{label} portal'] = [np.stack(frames)]

    # 3. Delegate video encoding and return its deterministic output path.
    save_panel_videos(output_dir, panels, fps=int(fps))
    return Path(output_dir) / 'env_0.mp4'


__all__ = [
    'SimulatedRolloutResult',
    'decode_next_prediction',
    'initialize_portal_starts',
    'ordered_hub_endpoints',
    'portal_labels',
    'preserve_module_modes',
    'real_imagined_frame',
    'save_simulated_rollout_panel',
]
