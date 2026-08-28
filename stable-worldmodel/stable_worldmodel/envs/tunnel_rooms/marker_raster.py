"""Marker appearance and raster contracts for Tunnel Rooms.

This module owns the persisted marker description shared by the environment,
dataset manifests, and evaluation. It also provides the exact native-resolution
shape masks used to render semantic cues.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_MARKER_APPEARANCE = {
    'marker_style': 'outline',
    'marker_width': 15,
    'marker_height': 15,
}
DEFAULT_AGENT_RENDER_RADIUS = 7.0

SHAPE_ORDER = ('triangle', 'diamond', 'star')
SHAPE_SIZES = ((18, 18), (20, 20), (20, 20))
SHAPE_RASTER_SPEC = {
    'triangle': {
        'vertices': [[8.5, 0.0], [0.0, 17.0], [17.0, 17.0]],
    },
    'diamond': {
        'center': [9.5, 9.5],
        'manhattan_radius': 9.5,
    },
    'star': {
        'center': [9.5, 9.5],
        'outer_radius': 9.8,
        'inner_radius': 5.5,
        'num_points': 5,
        'start_angle_radians': -math.pi / 2,
    },
}
NEUTRAL_RGB = np.array([62, 94, 112], dtype=np.uint8)


def configured_marker_appearance(
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Parse the optional nested marker appearance from a Hydra config."""
    legacy_keys = sorted(set(DEFAULT_MARKER_APPEARANCE).intersection(config))
    if legacy_keys:
        raise ValueError(
            'Marker appearance config must use the nested '
            f'marker_appearance mapping; remove top-level keys {legacy_keys}.'
        )
    configured = config.get('marker_appearance')
    if configured is None:
        return None
    return validate_marker_appearance(configured)


def marker_appearance_spec(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete marker contract selected by a config."""
    return configured_marker_appearance(config) or dict(
        DEFAULT_MARKER_APPEARANCE
    )


def validate_marker_appearance(
    appearance: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize one complete persisted marker contract."""
    appearance = dict(appearance)
    required = set(DEFAULT_MARKER_APPEARANCE)
    missing = sorted(required - set(appearance))
    extra = sorted(set(appearance) - required)
    if missing or extra:
        raise ValueError(
            'marker_appearance must contain exactly marker_style, '
            f'marker_width, and marker_height; missing={missing}, '
            f'extra={extra}.'
        )

    style = str(appearance['marker_style'])
    width = int(appearance['marker_width'])
    height = int(appearance['marker_height'])
    if (
        width != appearance['marker_width']
        or height != appearance['marker_height']
    ):
        raise ValueError('Marker dimensions must be integers.')

    if style == 'filled18x18':
        if (width, height) != (18, 18):
            raise ValueError(
                'filled18x18 requires marker_width=18 and marker_height=18.'
            )
    elif style == 'shape_cues':
        expected = (
            max(shape_width for shape_width, _ in SHAPE_SIZES),
            max(shape_height for _, shape_height in SHAPE_SIZES),
        )
        if (width, height) != expected:
            raise ValueError(
                f'shape_cues requires marker_width={expected[0]} and '
                f'marker_height={expected[1]}.'
            )
    elif style == 'outline':
        if width != height or width <= 0 or width % 2 == 0:
            raise ValueError(
                'outline markers require equal positive odd dimensions.'
            )
    else:
        raise ValueError(f'Unknown tunnel marker style: {style}')

    return {
        'marker_style': style,
        'marker_width': width,
        'marker_height': height,
    }


def scenario_marker_appearance(
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    """Read a scenario appearance, treating absent legacy fields as outline."""
    keys = set(DEFAULT_MARKER_APPEARANCE)
    present = keys.intersection(scenario)
    if not present:
        return dict(DEFAULT_MARKER_APPEARANCE)
    if present != keys:
        missing = sorted(keys - present)
        raise ValueError(
            f'Scenario marker appearance is incomplete; missing={missing}.'
        )
    return validate_marker_appearance({key: scenario[key] for key in keys})


def strip_default_scenario_appearance(
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    """Omit default appearance fields from a legacy-compatible scenario."""
    compatible = dict(scenario)
    direct_appearance = {
        key: compatible.get(key) for key in DEFAULT_MARKER_APPEARANCE
    }
    if direct_appearance == DEFAULT_MARKER_APPEARANCE:
        for key in DEFAULT_MARKER_APPEARANCE:
            compatible.pop(key)
    if compatible.get('agent_render_radius') == DEFAULT_AGENT_RENDER_RADIUS:
        compatible.pop('agent_render_radius')
    return compatible


def sha256_appearance_spec(spec: Mapping[str, Any]) -> str:
    """Hash a spec without changing hashes of legacy outline artifacts."""
    compatible = strip_default_scenario_appearance(spec)
    if compatible.get('marker_appearance') == DEFAULT_MARKER_APPEARANCE:
        compatible.pop('marker_appearance')
    return hashlib.sha256(_canonical_json(compatible).encode()).hexdigest()


@lru_cache(maxsize=len(SHAPE_ORDER))
def marker_mask(identity_index: int) -> np.ndarray:
    """Return one approved native-resolution shape mask without antialiasing."""
    if not 0 <= identity_index < len(SHAPE_ORDER):
        raise ValueError(f'Unknown marker identity index: {identity_index}')

    width, height = SHAPE_SIZES[identity_index]
    image = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(image)
    shape = SHAPE_ORDER[identity_index]
    if shape == 'triangle':
        draw.polygon([(8.5, 0), (0, 17), (17, 17)], fill=1)
    elif shape == 'diamond':
        yy, xx = np.indices((height, width))
        mask = np.abs(xx - 9.5) + np.abs(yy - 9.5) <= 9.5
        mask.setflags(write=False)
        return mask
    else:
        center = 9.5
        points = []
        for vertex in range(10):
            radius = 9.8 if vertex % 2 == 0 else 5.5
            angle = -math.pi / 2 + vertex * math.pi / 5
            points.append(
                (
                    center + radius * math.cos(angle),
                    center + radius * math.sin(angle),
                )
            )
        draw.polygon(points, fill=1)

    mask = np.asarray(image, dtype=bool)
    mask.setflags(write=False)
    return mask


def marker_size(identity_index: int) -> tuple[int, int]:
    """Return the approved ``(width, height)`` for one shape identity."""
    if not 0 <= identity_index < len(SHAPE_ORDER):
        raise ValueError(f'Unknown marker identity index: {identity_index}')
    return SHAPE_SIZES[identity_index]


def marker_mask_sha256(identity_index: int) -> str:
    """Hash the exact row-major boolean raster for provenance checks."""
    mask = np.ascontiguousarray(marker_mask(identity_index), dtype=np.uint8)
    return hashlib.sha256(mask.tobytes()).hexdigest()


def marker_slices(
    endpoint: Sequence[float],
    size: tuple[int, int],
    image_shape: Sequence[int],
    allowed_bounds: Sequence[int] | None = None,
) -> tuple[slice, slice]:
    """Place a complete mask, clamped inside its visible room when provided."""
    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    width, height = size
    if allowed_bounds is None:
        x_low, x_high, y_low, y_high = 0, image_width - 1, 0, image_height - 1
    else:
        x_low, x_high, y_low, y_high = map(int, allowed_bounds)
    if width > x_high - x_low + 1 or height > y_high - y_low + 1:
        raise ValueError('Marker does not fit inside its allowed bounds.')

    center_x = int(np.floor(float(endpoint[0]) + 0.5))
    center_y = int(np.floor(float(endpoint[1]) + 0.5))
    x_start = min(max(center_x - width // 2, x_low), x_high - width + 1)
    y_start = min(max(center_y - height // 2, y_low), y_high - height + 1)
    x_stop = x_start + width
    y_stop = y_start + height
    if not (0 <= x_start < x_stop <= image_width):
        raise ValueError(f'Marker at x={endpoint[0]} leaves the image.')
    if not (0 <= y_start < y_stop <= image_height):
        raise ValueError(f'Marker at y={endpoint[1]} leaves the image.')
    return slice(y_start, y_stop), slice(x_start, x_stop)


def shape_raster_contract() -> dict[str, object]:
    """Return the complete JSON-safe cue raster contract."""
    return {
        'shape_order': list(SHAPE_ORDER),
        'shape_sizes': {
            shape: list(size)
            for shape, size in zip(SHAPE_ORDER, SHAPE_SIZES, strict=True)
        },
        'shape_raster_spec': SHAPE_RASTER_SPEC,
        'shape_mask_sha256': {
            shape: marker_mask_sha256(identity)
            for identity, shape in enumerate(SHAPE_ORDER)
        },
        'marker_rgb': NEUTRAL_RGB.tolist(),
        'placement': 'clamp_complete_mask_to_visible_room_floor',
    }


# -----------------------------------------------------------------------------
# Private serialization helpers
# -----------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    """Convert scientific Python values used by persisted specs to JSON."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value).__name__} to JSON.')


def _canonical_json(value: Any) -> str:
    """Serialize hash inputs exactly as the legacy dataset code did."""
    return json.dumps(
        value,
        default=_json_default,
        sort_keys=True,
        separators=(',', ':'),
    )
