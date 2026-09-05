"""Pixel-exact Tunnel Rooms marker and agent rendering.

The live environment and immutable dataset rerendering both call this module.
Dataset derivation starts from recorded source frames, so these functions copy
the input and change only marker and agent pixels. This keeps historical walls
and backgrounds untouched while matching freshly rendered environment frames.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence

import numpy as np

from .marker_raster import (
    NEUTRAL_RGB,
    SHAPE_ORDER,
    SHAPE_SIZES,
    marker_mask,
    marker_size,
    marker_slices,
    shape_raster_contract,
)


# Fixed canvas geometry shared by physics bounds and marker placement.
IMG_SIZE = 224
BORDER_SIZE = 14
WALL_CENTER = 112
WALL_WIDTH_DEFAULT = 10

# Stable visual defaults used by all published Tunnel Rooms datasets.
AGENT_COLOR = np.asarray([213, 38, 45], dtype=np.uint8)
BACKGROUND_COLOR = np.asarray([244, 245, 241], dtype=np.uint8)
SEMANTIC_COLORS = (
    np.asarray([40, 153, 78], dtype=np.uint8),
    np.asarray([232, 142, 34], dtype=np.uint8),
    np.asarray([83, 98, 216], dtype=np.uint8),
    np.asarray([235, 20, 120], dtype=np.uint8),
)
SEMANTIC_COLOR_NAMES = ('green', 'orange', 'blue', 'magenta')

FILLED_MARKER_STYLE = 'filled18x18'
FILLED_MARKER_WIDTH = 18
FILLED_MARKER_HEIGHT = 18
FILLED_AGENT_RENDER_RADIUS = 9
PAIRED_SHAPE_ORDER = SHAPE_ORDER
PAIRED_SHAPE_STYLE = 'shape_cues'


def visible_room_bounds(room: int, num_leaves: int) -> np.ndarray:
    """Return inclusive pixel bounds for one visible room floor."""
    half_wall = WALL_WIDTH_DEFAULT // 2
    high = IMG_SIZE - BORDER_SIZE - 1
    if room == 0:
        return np.asarray(
            [BORDER_SIZE, WALL_CENTER - half_wall - 1, BORDER_SIZE, high]
        )
    if not 1 <= room <= int(num_leaves):
        raise ValueError(f'Unknown room index: {room}')
    return np.asarray(
        [
            WALL_CENTER + half_wall,
            high,
            BORDER_SIZE
            if room == 1
            else leaf_split_y(room - 1, num_leaves) + half_wall,
            high
            if room == int(num_leaves)
            else leaf_split_y(room, num_leaves) - half_wall - 1,
        ]
    )


def leaf_split_y(split_index: int, num_leaves: int) -> int:
    """Return the rendered center row of a wall between leaf rooms."""
    return round(
        BORDER_SIZE + (IMG_SIZE - 2 * BORDER_SIZE) * split_index / num_leaves
    )


def marker_appearance() -> dict[str, int | str]:
    """Return the scenario-level filled-marker contract."""
    return {
        'marker_style': FILLED_MARKER_STYLE,
        'marker_width': FILLED_MARKER_WIDTH,
        'marker_height': FILLED_MARKER_HEIGHT,
    }


def filled_appearance() -> dict[str, int | float | str]:
    """Return the complete appearance contract for filled datasets."""
    return {
        **marker_appearance(),
        'agent_render_radius': FILLED_AGENT_RENDER_RADIUS,
    }


def shape_marker_appearance() -> dict[str, int | str]:
    """Return the scenario-level bounding contract for shape cues."""
    return {
        'marker_style': PAIRED_SHAPE_STYLE,
        'marker_width': max(width for width, _ in SHAPE_SIZES),
        'marker_height': max(height for _, height in SHAPE_SIZES),
    }


def shape_appearance(
    source_revision: str = 'working-tree',
) -> dict[str, object]:
    """Return the complete provenance contract for shape-cue datasets."""
    return {
        **shape_marker_appearance(),
        'agent_render_radius': FILLED_AGENT_RENDER_RADIUS,
        'identity_mode': 'paired_shape',
        **shape_raster_contract(),
        'renderer_source_revision': str(source_revision),
    }


def marker_bounds(
    endpoint: Sequence[float], image_shape: Sequence[int]
) -> tuple[slice, slice]:
    """Return the strict 18x18 slice centered on one endpoint."""
    center_x = int(np.floor(float(endpoint[0]) + 0.5))
    center_y = int(np.floor(float(endpoint[1]) + 0.5))
    rows, columns = marker_slices(
        endpoint,
        (FILLED_MARKER_WIDTH, FILLED_MARKER_HEIGHT),
        image_shape,
    )
    if columns.start != center_x - FILLED_MARKER_WIDTH // 2:
        raise ValueError(f'Filled marker at x={endpoint[0]} leaves the image.')
    if rows.start != center_y - FILLED_MARKER_HEIGHT // 2:
        raise ValueError(f'Filled marker at y={endpoint[1]} leaves the image.')
    return rows, columns


def shape_marker_bounds(
    endpoint: Sequence[float],
    image_shape: Sequence[int],
    identity: int,
    endpoint_index: int,
    num_leaves: int,
) -> tuple[slice, slice]:
    """Place one complete shape inside the endpoint's visible room."""
    room = (
        0 if endpoint_index < num_leaves else endpoint_index - num_leaves + 1
    )
    allowed_bounds = (
        visible_room_bounds(room, num_leaves)
        if tuple(image_shape[:2]) == (IMG_SIZE, IMG_SIZE)
        else None
    )
    return marker_slices(
        endpoint,
        marker_size(identity),
        image_shape,
        allowed_bounds,
    )


def endpoint_colors(
    condition: str,
    num_leaves: int,
    pairing: int,
    cue_assignment: Sequence[int] | None = None,
    semantic_colors: Sequence[Sequence[int]] | None = None,
) -> tuple[np.ndarray, ...]:
    """Return default endpoint colors using environment pairing semantics."""
    if condition == 'hidden':
        return tuple(NEUTRAL_RGB for _ in range(2 * num_leaves))
    if condition != 'paired_color':
        raise ValueError(f'Unknown Tunnel Rooms condition: {condition}')
    if not 2 <= num_leaves <= len(SEMANTIC_COLORS):
        raise ValueError(
            f'Unsupported leaf count for semantic colors: {num_leaves}'
        )
    palette = SEMANTIC_COLORS
    if semantic_colors is not None:
        palette = np.asarray(semantic_colors)
        if (
            palette.shape != (num_leaves, 3)
            or not np.issubdtype(palette.dtype, np.integer)
            or np.any((palette < 0) | (palette > 255))
        ):
            raise ValueError(
                'Semantic colors must be one RGB integer triple per leaf.'
            )
    permutation = _pairing_permutation(num_leaves, pairing)
    cues = tuple(
        range(num_leaves) if cue_assignment is None else cue_assignment
    )
    if sorted(cues) != list(range(num_leaves)):
        raise ValueError('cue_assignment must be a complete cue permutation.')
    pair_identities = (
        *range(num_leaves),
        *(permutation.index(leaf) for leaf in range(num_leaves)),
    )
    return tuple(palette[cues[pair]] for pair in pair_identities)


def shape_identity_indices(num_leaves: int, pairing: int) -> tuple[int, ...]:
    """Return the hub and leaf shape identities in endpoint order."""
    if not 2 <= num_leaves <= len(PAIRED_SHAPE_ORDER):
        raise ValueError(
            f'Unsupported leaf count for paired shapes: {num_leaves}'
        )
    permutation = _pairing_permutation(num_leaves, pairing)
    return (
        *range(num_leaves),
        *(permutation.index(identity) for identity in range(num_leaves)),
    )


def render_filled_markers(
    source_frames: np.ndarray,
    scenario: Mapping[str, object],
    condition: str,
    agent_positions: np.ndarray,
    *,
    colors: Sequence[np.ndarray] | None = None,
    agent_color: np.ndarray = AGENT_COLOR,
) -> np.ndarray:
    """Overlay filled markers and the agent on recorded or fresh frames."""
    return render_markers(
        source_frames,
        scenario,
        condition,
        agent_positions,
        marker_style=FILLED_MARKER_STYLE,
        colors=colors,
        agent_color=agent_color,
    )


def render_shape_markers(
    source_frames: np.ndarray,
    scenario: Mapping[str, object],
    condition: str,
    agent_positions: np.ndarray,
    *,
    background_color: np.ndarray = BACKGROUND_COLOR,
    marker_color: np.ndarray = NEUTRAL_RGB,
    agent_color: np.ndarray = AGENT_COLOR,
) -> np.ndarray:
    """Replace prior markers with neutral paired shapes and draw the agent."""
    if condition != 'paired_shape':
        raise ValueError(
            f'Shape rerendering requires paired_shape, got {condition!r}.'
        )
    return render_markers(
        source_frames,
        scenario,
        condition,
        agent_positions,
        marker_style=PAIRED_SHAPE_STYLE,
        background_color=background_color,
        marker_color=marker_color,
        agent_color=agent_color,
    )


def render_markers(
    source_frames: np.ndarray,
    scenario: Mapping[str, object],
    condition: str,
    agent_positions: np.ndarray,
    *,
    marker_style: str,
    colors: Sequence[np.ndarray] | None = None,
    background_color: np.ndarray = BACKGROUND_COLOR,
    marker_color: np.ndarray = NEUTRAL_RGB,
    agent_color: np.ndarray = AGENT_COLOR,
) -> np.ndarray:
    """Render one marker style and draw the agent last on copied frames."""
    frames, single, endpoints, num_leaves, positions = _render_inputs(
        source_frames, scenario, agent_positions
    )
    endpoint_rgb: tuple[np.ndarray, ...] = ()
    if marker_style != PAIRED_SHAPE_STYLE:
        endpoint_rgb = (
            tuple(colors)
            if colors is not None
            else endpoint_colors(
                condition,
                num_leaves,
                int(scenario['pairing']),
                scenario.get('cue_assignment'),
                semantic_colors=scenario.get('tunnel_colors'),
            )
        )
        if len(endpoint_rgb) != len(endpoints):
            raise ValueError(
                'One marker color is required for every endpoint.'
            )
    rendered = frames.copy()
    if marker_style == FILLED_MARKER_STYLE:
        for endpoint, color in zip(endpoints, endpoint_rgb, strict=True):
            rows, columns = marker_bounds(endpoint, rendered.shape[1:3])
            rendered[:, rows, columns] = color
    elif marker_style == PAIRED_SHAPE_STYLE:
        identities = shape_identity_indices(
            num_leaves, int(scenario['pairing'])
        )
        for endpoint_index, (endpoint, identity) in enumerate(
            zip(endpoints, identities, strict=True)
        ):
            rows, columns = shape_marker_bounds(
                endpoint,
                rendered.shape[1:3],
                identity,
                endpoint_index,
                num_leaves,
            )
            marker = rendered[:, rows, columns]
            marker[:] = background_color
            marker[:, marker_mask(identity)] = marker_color
    elif marker_style == 'outline':
        _draw_outline_markers(
            rendered,
            endpoints,
            endpoint_rgb,
            float(scenario.get('marker_size', 7.0)),
            float(scenario.get('marker_stroke_width', 1.0)),
            background_color,
        )
    else:
        raise ValueError(f'Unknown tunnel marker style: {marker_style}')
    _draw_agent(
        rendered,
        positions,
        float(scenario.get('agent_render_radius', FILLED_AGENT_RENDER_RADIUS)),
        agent_color,
        use_float32_grid=marker_style == 'outline',
    )
    return rendered[0] if single else rendered


# -----------------------------------------------------------------------------
# Private helpers
# -----------------------------------------------------------------------------


def _pairing_permutation(num_leaves: int, pairing: int) -> tuple[int, ...]:
    """Validate and return the selected hub-to-leaf permutation."""
    permutations = tuple(itertools.permutations(range(num_leaves)))
    if not 0 <= int(pairing) < len(permutations):
        raise ValueError(f'Invalid pairing {pairing} for {num_leaves} leaves.')
    return permutations[int(pairing)]


def _render_inputs(
    source_frames: np.ndarray,
    scenario: Mapping[str, object],
    agent_positions: np.ndarray,
) -> tuple[np.ndarray, bool, np.ndarray, int, np.ndarray]:
    """Normalize the shared frame, endpoint, and agent-position contract."""
    frames = np.asarray(source_frames)
    single = frames.ndim == 3
    if single:
        frames = frames[None]
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            'Tunnel Rooms frames must have shape (N, H, W, 3) or (H, W, 3).'
        )
    if frames.dtype != np.uint8:
        raise ValueError(
            f'Tunnel Rooms frames must be uint8, got {frames.dtype}.'
        )

    endpoints = np.asarray(scenario['endpoint_positions'], dtype=np.float32)
    num_leaves = int(scenario.get('num_leaves', len(endpoints) // 2))
    if endpoints.shape != (2 * num_leaves, 2):
        raise ValueError(
            f'Expected {(2 * num_leaves, 2)} endpoints, got {endpoints.shape}.'
        )
    positions = np.asarray(agent_positions, dtype=np.float32)
    if positions.ndim == 1:
        positions = positions[None]
    if positions.shape != (len(frames), 2):
        raise ValueError(
            f'Expected agent positions with shape {(len(frames), 2)}, '
            f'got {positions.shape}.'
        )
    return frames, single, endpoints, num_leaves, positions


def _draw_agent(
    frames: np.ndarray,
    positions: np.ndarray,
    radius: float,
    color: np.ndarray,
    *,
    use_float32_grid: bool = False,
) -> None:
    """Draw agents last so they remain visible over tunnel markers."""
    grid_y, grid_x = np.ogrid[: frames.shape[1], : frames.shape[2]]
    if use_float32_grid:
        grid_y = grid_y.astype(np.float32)
        grid_x = grid_x.astype(np.float32)
    center_x = positions[:, 0, None, None]
    center_y = positions[:, 1, None, None]
    mask = (grid_x[None] - center_x) ** 2 + (
        grid_y[None] - center_y
    ) ** 2 <= radius**2
    frames[mask] = color


def _draw_outline_markers(
    frames: np.ndarray,
    endpoints: np.ndarray,
    colors: Sequence[np.ndarray],
    radius: float,
    stroke_width: float,
    background_color: np.ndarray,
) -> None:
    """Draw legacy outline markers with the environment's float32 grid."""
    y, x = np.ogrid[: frames.shape[1], : frames.shape[2]]
    y = y.astype(np.float32)
    x = x.astype(np.float32)
    for endpoint, color in zip(endpoints, colors, strict=True):
        outer = (np.abs(x - endpoint[0]) <= radius) & (
            np.abs(y - endpoint[1]) <= radius
        )
        inner = (np.abs(x - endpoint[0]) <= radius - stroke_width) & (
            np.abs(y - endpoint[1]) <= radius - stroke_width
        )
        frames[:, outer] = color
        frames[:, inner] = background_color
