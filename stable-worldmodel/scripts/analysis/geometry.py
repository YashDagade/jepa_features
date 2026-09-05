"""Valid-center sampling and roomwise rendering for static goal energies."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from stable_worldmodel.envs.tunnel_rooms.env import (
    HUB,
    IMG_SIZE,
    TunnelRoomsEnv,
)
from stable_worldmodel.envs.tunnel_rooms.rendering import visible_room_bounds

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@dataclass(frozen=True)
class Reference:
    endpoint: int
    room: int
    position: np.ndarray

    @property
    def label(self) -> str:
        return f'tunnel_{self.endpoint + 1}'


def room_for_endpoint(endpoint: int, num_leaves: int) -> int:
    if endpoint < num_leaves:
        return HUB
    return int(endpoint) - int(num_leaves) + 1


def endpoint_room(env: TunnelRoomsEnv, endpoint: int) -> int:
    return room_for_endpoint(endpoint, int(env.num_leaves))


def make_env(row: dict) -> TunnelRoomsEnv:
    env = TunnelRoomsEnv(
        identity_mode=row['identity_mode'],
        marker_style=row.get('marker_style', 'outline'),
        num_leaves=int(row.get('num_leaves', 2)),
    )
    env.reset(seed=row['scenario_seed'], options={'scenario': row})
    return env


def marker_size(env: TunnelRoomsEnv) -> float:
    return float(env.variation_space['tunnel']['marker_size'].value[0])


def marker_reference_extent(env: TunnelRoomsEnv) -> float:
    """Return the marker half-extent used to clear reference agents."""
    if env.marker_style == 'outline':
        return marker_size(env)
    return max(float(env.marker_width), float(env.marker_height)) / 2.0


def build_references(env: TunnelRoomsEnv) -> list[Reference]:
    refs = []
    marker_extent = marker_reference_extent(env)
    clearance = max(
        env.trigger_radius + 1.0,
        marker_extent + env.agent_render_radius + 2.0,
        env.landing_offset,
    )
    for endpoint in range(len(env.endpoint_positions)):
        room = endpoint_room(env, endpoint)
        refs.append(
            Reference(
                endpoint=endpoint,
                room=room,
                position=find_reference_position(
                    env, endpoint, room, clearance
                ),
            )
        )
    return refs


def find_reference_position(
    env: TunnelRoomsEnv,
    endpoint: int,
    room: int,
    clearance: float,
) -> np.ndarray:
    point = env.endpoint_positions[endpoint]
    direction = reference_direction(env, endpoint)
    for offset in np.linspace(clearance, env.rearm_distance, 15):
        candidate = point + np.array([direction * offset, 0.0], np.float32)
        if is_good_reference(env, candidate, point, room):
            return candidate.astype(np.float32)

    bounds = env._room_bounds(room)
    xs = np.linspace(bounds[0], bounds[1], 80)
    ys = np.linspace(bounds[2], bounds[3], 80)
    candidates = np.array(np.meshgrid(xs, ys, indexing='xy')).reshape(2, -1).T
    candidates = sorted(candidates, key=lambda p: np.linalg.norm(p - point))
    for candidate in candidates:
        candidate = candidate.astype(np.float32)
        if is_good_reference(env, candidate, point, room):
            return candidate
    raise RuntimeError(
        f'Could not find a valid reference for endpoint {endpoint}.'
    )


def reference_direction(env: TunnelRoomsEnv, endpoint: int) -> float:
    """Point away from the portal wall into the endpoint's own room."""
    return -1.0 if endpoint < int(env.num_leaves) else 1.0


def is_good_reference(
    env: TunnelRoomsEnv,
    candidate: np.ndarray,
    endpoint_position: np.ndarray,
    room: int,
) -> bool:
    if env.room_index(candidate) != room:
        return False
    distance = np.linalg.norm(candidate - endpoint_position)
    return bool(
        distance > env.trigger_radius
        and distance
        > marker_reference_extent(env) + env.agent_render_radius + 1.0
    )


def build_grid(
    env: TunnelRoomsEnv, grid_size: int
) -> list[tuple[int, int, np.ndarray]]:
    xs = np.linspace(0, IMG_SIZE - 1, grid_size)
    ys = np.linspace(0, IMG_SIZE - 1, grid_size)
    grid = []
    for y_idx, y in enumerate(ys):
        for x_idx, x in enumerate(xs):
            position = np.array([x, y], dtype=np.float32)
            if env.room_index(position) is not None:
                grid.append((y_idx, x_idx, position))
    return grid


def preprocess_images(images: list[np.ndarray]) -> torch.Tensor:
    array = np.stack(images)
    pixels = torch.as_tensor(array).permute(0, 3, 1, 2).float() / 255.0
    pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD
    return pixels.unsqueeze(1)


def interpolate_to_canvas(
    env: TunnelRoomsEnv,
    values: np.ndarray,
    rooms: set[int],
) -> np.ndarray:
    """Extend valid agent-center samples over each room's visible floor."""
    grid_size = values.shape[0]
    xs = np.linspace(0, IMG_SIZE - 1, grid_size)
    ys = np.linspace(0, IMG_SIZE - 1, grid_size)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing='xy')
    finite = np.isfinite(values)
    canvas = np.full((IMG_SIZE, IMG_SIZE), np.nan, dtype=np.float32)

    for room in rooms:
        sample_mask = finite & np.array(
            [
                env.room_index(np.array([x, y], dtype=np.float32)) == room
                for x, y in zip(grid_x.ravel(), grid_y.ravel())
            ],
            dtype=bool,
        ).reshape(values.shape)
        sample_points = np.stack(
            [grid_x[sample_mask], grid_y[sample_mask]], axis=1
        ).astype(np.float32)
        sample_values = values[sample_mask].astype(np.float32)
        if len(sample_values) == 0:
            continue

        pixel_mask = floor_mask(env, {room})
        py, px = np.nonzero(pixel_mask)
        pixels = np.stack([px, py], axis=1).astype(np.float32)
        canvas[pixel_mask] = inverse_distance_weight(
            pixels, sample_points, sample_values
        )
    return canvas


def inverse_distance_weight(
    pixels: np.ndarray,
    samples: np.ndarray,
    values: np.ndarray,
    k: int = 8,
) -> np.ndarray:
    k = min(k, len(values))
    chunks = []
    for start in range(0, len(pixels), 4096):
        chunk = pixels[start : start + 4096]
        dist2 = ((chunk[:, None, :] - samples[None, :, :]) ** 2).sum(axis=2)
        nearest = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
        d = np.take_along_axis(dist2, nearest, axis=1)
        v = values[nearest]
        weights = 1.0 / (d + 1e-3)
        chunks.append((weights * v).sum(axis=1) / weights.sum(axis=1))
    return np.concatenate(chunks).astype(np.float32)


def floor_mask(env: TunnelRoomsEnv, rooms: set[int]) -> np.ndarray:
    """Return every visible floor pixel in the requested rendered rooms."""
    ys, xs = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)
    for room in rooms:
        bounds = visible_room_bounds(room, int(env.num_leaves))
        mask |= (
            (bounds[0] <= xs)
            & (xs <= bounds[1])
            & (bounds[2] <= ys)
            & (ys <= bounds[3])
        )
    return mask


def common_references(scenario: dict) -> list[Reference]:
    """Use shared goal centers that clear the largest cue markers."""
    row = dict(
        scenario,
        identity_mode='paired_shape',
        marker_style='shape_cues',
        marker_width=20,
        marker_height=20,
    )
    env = make_env(row)
    try:
        return build_references(env)
    finally:
        env.close()
