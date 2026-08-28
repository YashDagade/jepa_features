"""Configurable hub-and-leaves benchmark for cue-conditioned dynamics."""

from __future__ import annotations

import itertools
import math
from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stable_worldmodel import spaces as swm_spaces

from .marker_raster import (
    NEUTRAL_RGB,
    SHAPE_SIZES,
)
from .rendering import (
    BORDER_SIZE,
    FILLED_MARKER_HEIGHT,
    FILLED_MARKER_WIDTH,
    IMG_SIZE,
    SEMANTIC_COLOR_NAMES,
    SEMANTIC_COLORS,
    WALL_CENTER,
    WALL_WIDTH_DEFAULT,
    leaf_split_y,
    render_markers,
)
from .routing import nearest_unexplored_hub, nearest_unexplored_order


# Physics and task constants shared by all sampled scenarios.
DOT_RADIUS_DEFAULT = 7
AGENT_RENDER_RADIUS_DEFAULT = 7
SPEED_DEFAULT = 5.0
SUCCESS_DISTANCE = 16.0

HUB = 0
UPPER_LEAF = 1
LOWER_LEAF = 2

HIDDEN = 0
PAIRED_COLOR = 1
PAIRED_SHAPE = 2

OUTLINE = 0
FILLED_18X18 = 1
SHAPE_CUES = 2
MARKER_STYLES = ('outline', 'filled18x18', 'shape_cues')
DEFAULT_MARKER_WIDTH = 15
DEFAULT_MARKER_HEIGHT = 15
PAIR_COLOR_FIELDS = (
    'pair_a_color',
    'pair_b_color',
    'pair_c_color',
    'pair_d_color',
)


def room_bounds(
    room: int,
    num_leaves: int,
    agent_radius: float = DOT_RADIUS_DEFAULT,
) -> np.ndarray:
    """Return the navigable bounds for one hub or leaf room."""
    inset = float(agent_radius) + 2.0
    wall_half = WALL_WIDTH_DEFAULT / 2.0
    outside_low = BORDER_SIZE + inset
    outside_high = IMG_SIZE - BORDER_SIZE - inset
    if room == HUB:
        return np.asarray(
            [
                outside_low,
                WALL_CENTER - wall_half - inset,
                outside_low,
                outside_high,
            ],
            dtype=np.float32,
        )
    if not 1 <= room <= int(num_leaves):
        raise ValueError(f'Unknown room index: {room}')
    usable_low = BORDER_SIZE
    usable_high = IMG_SIZE - BORDER_SIZE
    split = usable_low + (usable_high - usable_low) * (
        (room - 1) / int(num_leaves)
    )
    next_split = usable_low + (usable_high - usable_low) * (
        room / int(num_leaves)
    )
    y_low = split + inset + (wall_half if room > 1 else 0.0)
    y_high = (
        next_split - inset - (wall_half if room < int(num_leaves) else 0.0)
    )
    return np.asarray(
        [WALL_CENTER + wall_half + inset, outside_high, y_low, y_high],
        dtype=np.float32,
    )


def _mode_value(mode: int | str) -> int:
    if isinstance(mode, str):
        names = {
            'hidden': HIDDEN,
            'paired_color': PAIRED_COLOR,
            'paired_shape': PAIRED_SHAPE,
        }
        if mode not in names:
            raise ValueError(f'Unknown tunnel identity mode: {mode}')
        return names[mode]
    if mode not in (HIDDEN, PAIRED_COLOR, PAIRED_SHAPE):
        raise ValueError(f'Unknown tunnel identity mode: {mode}')
    return int(mode)


def _marker_style_value(style: int | str) -> int:
    if isinstance(style, str):
        names = {name: index for index, name in enumerate(MARKER_STYLES)}
        if style not in names:
            raise ValueError(f'Unknown tunnel marker style: {style}')
        return names[style]
    if style not in (OUTLINE, FILLED_18X18, SHAPE_CUES):
        raise ValueError(f'Unknown tunnel marker style: {style}')
    return int(style)


class TunnelRoomsEnv(gym.Env):
    """Hub-and-leaves tunnel environment connected only by portals.

    ``num_leaves=1`` is the single-portal navigation control.
    ``num_leaves=2`` preserves the original two-leaf Tunnel Rooms layout.
    ``num_leaves=3`` adds a rank-certified three-tunnel sampler with a
    close pair, a far third tunnel, and an anchor-adjacent start.
    ``num_leaves=4`` extends the randomized, rank-certified protocol with four
    visually distinct color identities.
    """

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 10}

    def __init__(
        self,
        render_mode: str = 'rgb_array',
        identity_mode: int | str = HIDDEN,
        marker_style: int | str = OUTLINE,
        init_value: dict[str, Any] | None = None,
        num_leaves: int = 2,
    ) -> None:
        super().__init__()
        if render_mode != 'rgb_array':
            raise ValueError(
                'TunnelRoomsEnv supports only rgb_array rendering.'
            )
        if int(num_leaves) not in (1, 2, 3, 4):
            raise ValueError(
                'TunnelRoomsEnv currently supports 1, 2, 3, or 4 leaves.'
            )
        identity_mode_value = _mode_value(identity_mode)
        marker_style_value = _marker_style_value(marker_style)
        if (identity_mode_value == PAIRED_SHAPE) != (
            marker_style_value == SHAPE_CUES
        ):
            raise ValueError(
                'paired_shape and marker_style="shape_cues" must be used together.'
            )

        self.render_mode = render_mode
        self.num_leaves = int(num_leaves)
        self._tunnel_color_names = SEMANTIC_COLOR_NAMES[: self.num_leaves]
        self.layout = f'hub_left_{self.num_leaves}_leaves_right'
        self._pairing_permutations = tuple(
            itertools.permutations(range(self.num_leaves))
        )

        self.observation_space = spaces.Dict(
            {
                'proprio': spaces.Box(
                    0.0, float(IMG_SIZE), shape=(2,), dtype=np.float32
                ),
                'state': spaces.Box(
                    0.0, float(IMG_SIZE), shape=(2,), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        default_endpoints = self._default_endpoints()
        self.variation_space = swm_spaces.Dict(
            {
                'agent': swm_spaces.Dict(
                    {
                        'position': swm_spaces.Box(
                            0,
                            IMG_SIZE - 1,
                            shape=(2,),
                            dtype=np.float32,
                            init_value=np.array([44, 112], np.float32),
                        ),
                        'radius': swm_spaces.Box(
                            1,
                            16,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array(
                                [DOT_RADIUS_DEFAULT], np.float32
                            ),
                        ),
                        'render_radius': swm_spaces.Box(
                            1,
                            16,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array(
                                [AGENT_RENDER_RADIUS_DEFAULT], np.float32
                            ),
                        ),
                        'speed': swm_spaces.Box(
                            0.1,
                            20,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([SPEED_DEFAULT], np.float32),
                        ),
                        'color': swm_spaces.RGBBox(
                            init_value=np.array([213, 38, 45], np.uint8)
                        ),
                    }
                ),
                'target': swm_spaces.Dict(
                    {
                        'position': swm_spaces.Box(
                            0,
                            IMG_SIZE - 1,
                            shape=(2,),
                            dtype=np.float32,
                            init_value=self._default_target(),
                        ),
                        'leaf': swm_spaces.Discrete(
                            self.num_leaves, init_value=0
                        ),
                    }
                ),
                'tunnel': swm_spaces.Dict(
                    {
                        'identity_mode': swm_spaces.Discrete(
                            3, init_value=identity_mode_value
                        ),
                        'marker_style': swm_spaces.Discrete(
                            len(MARKER_STYLES), init_value=marker_style_value
                        ),
                        'pairing': swm_spaces.Discrete(
                            math.factorial(self.num_leaves), init_value=0
                        ),
                        'cue_assignment': swm_spaces.Discrete(
                            math.factorial(self.num_leaves), init_value=0
                        ),
                        'endpoints': swm_spaces.Box(
                            0,
                            IMG_SIZE - 1,
                            shape=(2 * self.num_leaves, 2),
                            dtype=np.float32,
                            init_value=default_endpoints,
                        ),
                        'trigger_radius': swm_spaces.Box(
                            1,
                            30,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([10], np.float32),
                        ),
                        'landing_offset': swm_spaces.Box(
                            1,
                            40,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([14], np.float32),
                        ),
                        'rearm_distance': swm_spaces.Box(
                            1,
                            80,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([28], np.float32),
                        ),
                        'min_hub_separation': swm_spaces.Box(
                            20,
                            120,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([42], np.float32),
                        ),
                        'marker_size': swm_spaces.Box(
                            2,
                            14,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([7], np.float32),
                        ),
                        'marker_stroke_width': swm_spaces.Box(
                            1,
                            6,
                            shape=(1,),
                            dtype=np.float32,
                            init_value=np.array([1], np.float32),
                        ),
                        'hidden_color': swm_spaces.RGBBox(
                            init_value=np.array([62, 94, 112], np.uint8)
                        ),
                        'pair_a_color': swm_spaces.RGBBox(
                            init_value=np.array([40, 153, 78], np.uint8)
                        ),
                        'pair_b_color': swm_spaces.RGBBox(
                            init_value=np.array([232, 142, 34], np.uint8)
                        ),
                        'pair_c_color': swm_spaces.RGBBox(
                            init_value=np.array([83, 98, 216], np.uint8)
                        ),
                        'pair_d_color': swm_spaces.RGBBox(
                            init_value=np.array([235, 20, 120], np.uint8)
                        ),
                    }
                ),
                'appearance': swm_spaces.Dict(
                    {
                        'background': swm_spaces.RGBBox(
                            init_value=np.array([244, 245, 241], np.uint8)
                        ),
                        'walls': swm_spaces.RGBBox(
                            init_value=np.array([24, 26, 28], np.uint8)
                        ),
                    }
                ),
            }
        )
        if init_value:
            self.variation_space.set_init_value(init_value)

        self.agent_position = np.zeros(2, dtype=np.float32)
        self.target_position = np.zeros(2, dtype=np.float32)
        self.endpoint_positions = default_endpoints.copy()
        self._initial_agent_position = self.agent_position.copy()
        self._episode_seed: int | None = None
        self._locked_endpoint: int | None = None
        self._goal: np.ndarray | None = None
        self._search_order: list[int] | None = None
        self._correct_hub_endpoint: int | None = None
        self._target_rank: int | None = None
        self._validation_profile = 'ranked_route'
        self._hub_entry_trace: list[int] = []
        self.teleport_count = 0
        self.route_length = 0.0
        self.step_count = 0
        self._terminated = False

    @property
    def identity_mode(self) -> int:
        return int(self.variation_space['tunnel']['identity_mode'].value)

    @property
    def pairing(self) -> int:
        return int(self.variation_space['tunnel']['pairing'].value)

    @property
    def cue_assignment(self) -> tuple[int, ...]:
        """Map each routed tunnel pair to its visible cue identity."""
        index = int(self.variation_space['tunnel']['cue_assignment'].value)
        return self._pairing_permutations[index]

    @property
    def marker_style(self) -> str:
        style = int(self.variation_space['tunnel']['marker_style'].value)
        return MARKER_STYLES[style]

    @property
    def marker_width(self) -> int:
        if self.marker_style == 'filled18x18':
            return FILLED_MARKER_WIDTH
        if self.marker_style == 'shape_cues':
            return max(width for width, _ in SHAPE_SIZES)
        marker_size = self.variation_space['tunnel']['marker_size'].value[0]
        return int(round(2 * float(marker_size) + 1))

    @property
    def marker_height(self) -> int:
        if self.marker_style == 'filled18x18':
            return FILLED_MARKER_HEIGHT
        if self.marker_style == 'shape_cues':
            return max(height for _, height in SHAPE_SIZES)
        return self.marker_width

    @property
    def agent_radius(self) -> float:
        return float(self.variation_space['agent']['radius'].value[0])

    @property
    def agent_render_radius(self) -> float:
        """Visual radius; independent of collision and room geometry."""
        return float(self.variation_space['agent']['render_radius'].value[0])

    @property
    def speed(self) -> float:
        return float(self.variation_space['agent']['speed'].value[0])

    @property
    def trigger_radius(self) -> float:
        return float(self.variation_space['tunnel']['trigger_radius'].value[0])

    @property
    def landing_offset(self) -> float:
        return float(self.variation_space['tunnel']['landing_offset'].value[0])

    @property
    def rearm_distance(self) -> float:
        return float(self.variation_space['tunnel']['rearm_distance'].value[0])

    @property
    def min_hub_separation(self) -> float:
        return float(
            self.variation_space['tunnel']['min_hub_separation'].value[0]
        )

    @property
    def target_leaf(self) -> int:
        return int(self.variation_space['target']['leaf'].value)

    @property
    def leaf_rooms(self) -> tuple[int, ...]:
        return tuple(range(1, self.num_leaves + 1))

    def hub_endpoint_indices(self) -> tuple[int, ...]:
        return tuple(range(self.num_leaves))

    def leaf_endpoint_indices(self) -> tuple[int, ...]:
        return tuple(range(self.num_leaves, 2 * self.num_leaves))

    def leaf_room_for_target_leaf(self, target_leaf: int) -> int:
        return int(target_leaf) + 1

    def leaf_endpoint_for_room(self, room: int) -> int:
        if room not in self.leaf_rooms:
            raise ValueError(f'Room {room} is not a leaf room.')
        return self.num_leaves + int(room) - 1

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self._episode_seed = seed
        options = deepcopy(options or {})
        scenario = options.get('scenario') or {}
        self._validation_profile = str(
            scenario.get(
                'validation_profile',
                options.pop('validation_profile', 'ranked_route'),
            )
        )
        if self._validation_profile not in {'ranked_route', 'fixed_layout'}:
            raise ValueError(
                'validation_profile must be "ranked_route" or "fixed_layout".'
            )
        self._expand_marker_appearance(options)
        swm_spaces.reset_variation_space(
            self.variation_space, seed, options, default_variations=()
        )
        self._search_order = None
        self._correct_hub_endpoint = None
        self._target_rank = None
        self._tunnel_color_names = SEMANTIC_COLOR_NAMES[: self.num_leaves]

        if 'scenario' in options:
            self._apply_scenario(options['scenario'])
        else:
            self._apply_random_scenario(options)

        self._certify_scenario()
        self._validate_scenario()
        self.endpoint_positions = np.asarray(
            self.variation_space['tunnel']['endpoints'].value,
            dtype=np.float32,
        ).copy()
        self.agent_position = np.asarray(
            self.variation_space['agent']['position'].value,
            dtype=np.float32,
        ).copy()
        self.target_position = np.asarray(
            self.variation_space['target']['position'].value,
            dtype=np.float32,
        ).copy()
        self._initial_agent_position = self.agent_position.copy()
        self._locked_endpoint = None
        self._hub_entry_trace = []
        self.teleport_count = 0
        self.route_length = 0.0
        self.step_count = 0
        self._terminated = False
        self._goal = self._render_frame(self.target_position)
        return self._get_obs(), self._get_info()

    def _expand_marker_appearance(self, options: dict[str, Any]) -> None:
        """Translate the public appearance mapping into variation values."""
        appearance = options.pop('marker_appearance', None)
        if appearance is None:
            return
        if 'scenario' in options:
            raise ValueError(
                'Specify marker appearance in the scenario or reset options, '
                'not both.'
            )
        style, marker_size = self._validate_marker_contract(appearance)
        values = options.setdefault('variation_values', {})
        requested = {
            'tunnel.marker_style': _marker_style_value(style),
            'tunnel.marker_size': np.array([marker_size], np.float32),
        }
        conflicts = sorted(set(values).intersection(requested))
        if conflicts:
            raise ValueError(
                'marker_appearance conflicts with variation_values: '
                f'{conflicts}.'
            )
        values.update(requested)

    @staticmethod
    def _validate_marker_contract(
        appearance: dict[str, Any],
    ) -> tuple[str, float]:
        required = {'marker_style', 'marker_width', 'marker_height'}
        missing = sorted(required - set(appearance))
        extra = sorted(set(appearance) - required)
        if missing or extra:
            raise ValueError(
                'marker_appearance must contain exactly marker_style, '
                f'marker_width, and marker_height; missing={missing}, '
                f'extra={extra}.'
            )
        style_value = _marker_style_value(appearance['marker_style'])
        style = MARKER_STYLES[style_value]
        width = int(appearance['marker_width'])
        height = int(appearance['marker_height'])
        if (
            width != appearance['marker_width']
            or height != appearance['marker_height']
        ):
            raise ValueError('Marker dimensions must be integers.')
        if style == 'filled18x18':
            if (width, height) != (
                FILLED_MARKER_WIDTH,
                FILLED_MARKER_HEIGHT,
            ):
                raise ValueError(
                    'filled18x18 requires marker_width=18 and '
                    'marker_height=18.'
                )
            return style, float((DEFAULT_MARKER_WIDTH - 1) / 2)
        if style == 'shape_cues':
            expected = (
                max(width for width, _ in SHAPE_SIZES),
                max(height for _, height in SHAPE_SIZES),
            )
            if (width, height) != expected:
                raise ValueError(
                    f'shape_cues requires marker_width={expected[0]} and '
                    f'marker_height={expected[1]}.'
                )
            return style, float((DEFAULT_MARKER_WIDTH - 1) / 2)
        if width != height or width <= 0 or width % 2 == 0:
            raise ValueError(
                'outline markers require equal positive odd dimensions.'
            )
        return style, float((width - 1) / 2)

    def _default_endpoints(self) -> np.ndarray:
        if self.num_leaves == 1:
            return np.array([[64, 112], [162, 112]], dtype=np.float32)
        if self.num_leaves == 2:
            return np.array(
                [[64, 70], [64, 154], [162, 70], [162, 154]],
                dtype=np.float32,
            )
        if self.num_leaves == 3:
            return np.array(
                [
                    [64, 46],
                    [64, 112],
                    [64, 178],
                    [162, 46],
                    [162, 112],
                    [162, 178],
                ],
                dtype=np.float32,
            )
        return np.array(
            [
                [64, 38],
                [64, 87],
                [64, 137],
                [64, 186],
                [162, 38],
                [162, 87],
                [162, 137],
                [162, 186],
            ],
            dtype=np.float32,
        )

    def _default_target(self) -> np.ndarray:
        if self.num_leaves == 1:
            return np.array([184, 112], np.float32)
        if self.num_leaves == 2:
            return np.array([184, 70], np.float32)
        return np.array([184, 46 if self.num_leaves == 3 else 38], np.float32)

    def _apply_random_scenario(self, options: dict[str, Any]) -> None:
        self._validate_identity_marker_contract()
        if self.num_leaves in (3, 4):
            retired = {
                'ordered_pair',
                'ordered_pair_index',
                'pairing_index',
                'scenario_index',
                'scenario_seed_base',
                'target_rank',
                'target_rank_probs',
            }
            supplied = sorted(retired.intersection(options))
            if supplied:
                raise ValueError(
                    'Multi-leaf sampling no longer accepts rank- or '
                    f'order-forcing options: {supplied}.'
                )
        explicit = set(options.get('variation_values', {}).keys())
        constraints = options.get('scenario_constraints') or {}
        for _ in range(5000):
            if self.num_leaves == 3:
                sampled = self._sample_three_leaf_scenario()
            elif self.num_leaves == 4:
                sampled = self._sample_four_leaf_scenario()
            elif self.num_leaves == 2:
                sampled = self._sample_two_leaf_scenario()
            else:
                sampled = self._sample_one_leaf_scenario()
            update = {
                k: v
                for k, v in sampled.items()
                if '.' in k and k not in explicit
            }
            self.variation_space.set_value(update)
            self._certify_scenario()
            try:
                self._validate_scenario()
                self._validate_sampling_constraints(constraints)
            except ValueError:
                continue
            break
        else:
            raise RuntimeError(
                'Unable to sample a valid tunnel-room scenario.'
            )

    def _sample_two_leaf_scenario(self) -> dict[str, Any]:
        leaf = int(self.np_random.integers(0, 2))
        return {
            'tunnel.pairing': int(self.np_random.integers(0, 2)),
            'tunnel.endpoints': np.stack(
                [
                    self._sample_position(HUB, portal=True),
                    self._sample_position(HUB, portal=True),
                    self._sample_position(UPPER_LEAF, portal=True),
                    self._sample_position(LOWER_LEAF, portal=True),
                ]
            ).astype(np.float32),
            'agent.position': self._sample_position(HUB),
            'target.leaf': leaf,
            'target.position': self._sample_position(leaf + 1),
        }

    def _sample_one_leaf_scenario(self) -> dict[str, Any]:
        """Sample the direct one-portal navigation control."""
        return {
            'tunnel.pairing': 0,
            'tunnel.endpoints': np.stack(
                [
                    self._sample_position(HUB, portal=True),
                    self._sample_position(UPPER_LEAF, portal=True),
                ]
            ).astype(np.float32),
            'agent.position': self._sample_position(HUB),
            'target.leaf': 0,
            'target.position': self._sample_position(UPPER_LEAF),
        }

    def _sample_three_leaf_scenario(self) -> dict[str, Any]:
        endpoints, start = self._sample_three_leaf_geometry()
        target_leaf = int(self.np_random.integers(0, self.num_leaves))
        pairing = int(
            self.np_random.integers(0, len(self._pairing_permutations))
        )
        target_room = self.leaf_room_for_target_leaf(target_leaf)
        target = self._sample_position(target_room)

        return {
            'tunnel.pairing': pairing,
            'tunnel.endpoints': endpoints.astype(np.float32),
            'agent.position': start.astype(np.float32),
            'target.leaf': target_leaf,
            'target.position': target.astype(np.float32),
        }

    def _sample_four_leaf_scenario(self) -> dict[str, Any]:
        """Sample four randomized portals with a certified search order."""
        endpoints = np.zeros((2 * self.num_leaves, 2), dtype=np.float32)

        # Four narrow vertical bands keep portals well separated while random
        # x positions and a role permutation prevent a fixed identity layout.
        y_values = np.linspace(38.0, 186.0, self.num_leaves)
        y_values += self.np_random.uniform(-2.0, 2.0, self.num_leaves)
        x_values = self.np_random.uniform(56.0, 82.0, self.num_leaves)
        role_labels = self.np_random.permutation(self.num_leaves)
        for role, endpoint_idx in enumerate(role_labels):
            endpoints[endpoint_idx] = [x_values[role], y_values[role]]
        for leaf_idx, room in enumerate(self.leaf_rooms):
            endpoints[self.num_leaves + leaf_idx] = self._sample_position(
                room, portal=True
            )

        target_leaf = int(self.np_random.integers(0, self.num_leaves))
        target_room = self.leaf_room_for_target_leaf(target_leaf)
        return {
            'tunnel.pairing': int(
                self.np_random.integers(0, len(self._pairing_permutations))
            ),
            'tunnel.endpoints': endpoints,
            'agent.position': self._sample_anchor_adjacent_start(
                int(role_labels[0]), endpoints
            ),
            'target.leaf': target_leaf,
            'target.position': self._sample_position(target_room),
        }

    def _sample_three_leaf_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        endpoints = np.zeros((2 * self.num_leaves, 2), dtype=np.float32)
        close_delta = float(self.np_random.uniform(43.0, 56.0))
        y_low, y_high = 38.0, 186.0
        far_high = min(104.0, y_high - y_low - close_delta)
        far_delta = float(self.np_random.uniform(84.0, far_high))
        sign = -1.0 if self.np_random.random() < 0.5 else 1.0
        if sign > 0:
            anchor_bounds = (y_low + far_delta, y_high - close_delta)
        else:
            anchor_bounds = (y_low + close_delta, y_high - far_delta)
        anchor_y = float(self.np_random.uniform(*anchor_bounds))
        y_values = np.array(
            [
                anchor_y,
                anchor_y + sign * close_delta,
                anchor_y - sign * far_delta,
            ],
            dtype=np.float32,
        )
        x_values = self.np_random.uniform(56.0, 82.0, size=3)
        role_labels = self.np_random.permutation(self.num_leaves)
        for role, endpoint_idx in enumerate(role_labels):
            endpoints[endpoint_idx] = [x_values[role], y_values[role]]

        for leaf_idx, room in enumerate(self.leaf_rooms):
            endpoints[self.num_leaves + leaf_idx] = self._sample_position(
                room, portal=True
            )
        start = self._sample_anchor_adjacent_start(
            int(role_labels[0]), endpoints
        )
        return endpoints, start

    def _sample_anchor_adjacent_start(
        self, anchor_endpoint: int, endpoints: np.ndarray
    ) -> np.ndarray:
        anchor = endpoints[anchor_endpoint]
        for _ in range(256):
            x = float(self.np_random.uniform(28.0, 42.0))
            y = float(anchor[1] + self.np_random.uniform(-10.0, 10.0))
            start = np.array([x, y], dtype=np.float32)
            distances = np.linalg.norm(
                endpoints[: self.num_leaves] - start, axis=1
            )
            if int(np.argmin(distances)) == anchor_endpoint:
                if float(distances.min()) >= self.rearm_distance:
                    return start
        return np.array([34.0, anchor[1]], dtype=np.float32)

    def _certify_scenario(self) -> None:
        if self.num_leaves not in (1, 3, 4):
            self._search_order = None
            self._correct_hub_endpoint = None
            self._target_rank = None
            return

        endpoints = np.asarray(
            self.variation_space['tunnel']['endpoints'].value,
            dtype=np.float32,
        )
        start = np.asarray(
            self.variation_space['agent']['position'].value,
            dtype=np.float32,
        )
        return_positions = np.stack(
            [
                self._landing_position(endpoint, endpoints)
                for endpoint in self.hub_endpoint_indices()
            ]
        )
        self._search_order = nearest_unexplored_order(
            start,
            endpoints[: self.num_leaves],
            return_positions,
        )
        pairing = int(self.variation_space['tunnel']['pairing'].value)
        target_leaf = int(self.variation_space['target']['leaf'].value)
        perm = self._pairing_permutations[pairing]
        self._correct_hub_endpoint = int(perm.index(target_leaf))
        self._target_rank = (
            self._search_order.index(self._correct_hub_endpoint) + 1
        )

    def _apply_scenario(self, scenario: dict[str, Any]) -> None:
        if int(scenario.get('num_leaves', self.num_leaves)) != self.num_leaves:
            raise ValueError('Replay scenario num_leaves does not match env.')
        fields = {
            'agent.position': np.asarray(
                scenario['agent_position'], dtype=np.float32
            ),
            'target.position': np.asarray(
                scenario['target_position'], dtype=np.float32
            ),
            'target.leaf': int(scenario['target_leaf']),
            'tunnel.endpoints': np.asarray(
                scenario['endpoint_positions'], dtype=np.float32
            ),
            'tunnel.pairing': int(scenario['pairing']),
            # Legacy scenarios used pair index as cue identity.
            'tunnel.cue_assignment': self._permutation_index(
                scenario.get('cue_assignment', range(self.num_leaves))
            ),
            # Persisted scenarios written before marker styles existed are
            # explicitly legacy outline scenarios, regardless of constructor
            # defaults.
            'tunnel.marker_style': _marker_style_value(
                scenario.get('marker_style', 'outline')
            ),
        }
        optional = {
            'identity_mode': 'tunnel.identity_mode',
            'trigger_radius': 'tunnel.trigger_radius',
            'landing_offset': 'tunnel.landing_offset',
            'rearm_distance': 'tunnel.rearm_distance',
            'min_hub_separation': 'tunnel.min_hub_separation',
            'marker_size': 'tunnel.marker_size',
            'marker_stroke_width': 'tunnel.marker_stroke_width',
            'speed': 'agent.speed',
            'agent_render_radius': 'agent.render_radius',
        }
        for key, path in optional.items():
            if key not in scenario:
                continue
            value: Any = scenario[key]
            if key == 'identity_mode':
                value = _mode_value(value)
            else:
                value = np.array([value], dtype=np.float32)
            fields[path] = value

        # Replay explicit palettes without inventing a second identity mode.
        if 'tunnel_colors' in scenario:
            colors = np.asarray(scenario['tunnel_colors'], dtype=np.uint8)
            if colors.shape != (self.num_leaves, 3):
                raise ValueError(
                    'tunnel_colors must have shape '
                    f'{(self.num_leaves, 3)}, got {colors.shape}.'
                )
            fields.update(
                {
                    f'tunnel.{name}': color
                    for name, color in zip(
                        PAIR_COLOR_FIELDS[: self.num_leaves],
                        colors,
                        strict=True,
                    )
                }
            )
        if 'tunnel_color_names' in scenario:
            names = tuple(str(name) for name in scenario['tunnel_color_names'])
            if len(names) != self.num_leaves or len(set(names)) != len(names):
                raise ValueError(
                    'tunnel_color_names must provide one unique name per leaf.'
                )
            self._tunnel_color_names = names

        appearance_keys = {
            'marker_style',
            'marker_width',
            'marker_height',
        }
        if appearance_keys.intersection(scenario):
            appearance = {
                key: scenario[key]
                for key in appearance_keys
                if key in scenario
            }
            _, marker_size = self._validate_marker_contract(appearance)
            if 'marker_size' in scenario and not np.isclose(
                float(scenario['marker_size']), marker_size
            ):
                raise ValueError(
                    'marker_size disagrees with marker_width/marker_height.'
                )
            fields['tunnel.marker_size'] = np.array(
                [marker_size], dtype=np.float32
            )
        self.variation_space.set_value(fields)

    def _sample_position(self, room: int, portal: bool = False) -> np.ndarray:
        bounds = self._room_bounds(room)
        inset = (
            max(self.landing_offset, self.rearm_distance) if portal else 0.0
        )
        if portal and room == HUB:
            bounds[0] += inset
        elif portal:
            bounds[1] -= inset
        return self.np_random.uniform(bounds[[0, 2]], bounds[[1, 3]]).astype(
            np.float32
        )

    def _room_bounds(self, room: int) -> np.ndarray:
        return room_bounds(room, self.num_leaves, self.agent_radius)

    def room_index(self, position: np.ndarray) -> int | None:
        position = np.asarray(position, dtype=np.float32)
        for room in (HUB, *self.leaf_rooms):
            bounds = self._room_bounds(room)
            if (
                bounds[0] <= position[0] <= bounds[1]
                and bounds[2] <= position[1] <= bounds[3]
            ):
                return room
        return None

    def _validate_identity_marker_contract(self) -> None:
        if (self.identity_mode == PAIRED_SHAPE) != (
            self.marker_style == 'shape_cues'
        ):
            raise ValueError(
                'paired_shape and marker_style="shape_cues" must be used together.'
            )
        hidden_color = np.asarray(
            self.variation_space['tunnel']['hidden_color'].value,
            dtype=np.uint8,
        )
        if self.identity_mode == PAIRED_SHAPE and not np.array_equal(
            hidden_color, NEUTRAL_RGB
        ):
            raise ValueError(
                f'paired_shape requires marker RGB {NEUTRAL_RGB.tolist()}.'
            )

    def _validate_scenario(self) -> None:
        self._validate_identity_marker_contract()
        endpoints = np.asarray(
            self.variation_space['tunnel']['endpoints'].value,
            dtype=np.float32,
        )
        start = np.asarray(
            self.variation_space['agent']['position'].value,
            dtype=np.float32,
        )
        target = np.asarray(
            self.variation_space['target']['position'].value,
            dtype=np.float32,
        )
        if endpoints.shape != (2 * self.num_leaves, 2):
            raise ValueError('Tunnel endpoint array has the wrong shape.')
        if self.landing_offset >= self.rearm_distance:
            raise ValueError(
                'Tunnel landing_offset must be smaller than rearm_distance.'
            )

        required_rooms = [HUB] * self.num_leaves + list(self.leaf_rooms)
        for idx, room in enumerate(required_rooms):
            if self.room_index(endpoints[idx]) != room:
                raise ValueError(f'Tunnel endpoint {idx} is outside its room.')
            if self.room_index(self._landing_position(idx, endpoints)) != room:
                raise ValueError(
                    f'Tunnel endpoint {idx} has an invalid landing.'
                )
            if self.room_index(self._rearm_position(idx, endpoints)) != room:
                raise ValueError(
                    f'Tunnel endpoint {idx} has insufficient re-arm clearance.'
                )
        if self.room_index(start) != HUB:
            raise ValueError('Agent start must be in the hub room.')
        target_room = self.leaf_room_for_target_leaf(self.target_leaf)
        if self.room_index(target) != target_room:
            raise ValueError('Target position does not match target leaf.')

        portal_clearance = max(
            self.min_hub_separation,
            2 * self.trigger_radius + self.agent_radius,
        )
        hub_endpoints = endpoints[: self.num_leaves]
        for left, right in itertools.combinations(range(self.num_leaves), 2):
            if (
                np.linalg.norm(hub_endpoints[left] - hub_endpoints[right])
                < portal_clearance
            ):
                raise ValueError(
                    'Hub tunnels violate configured minimum separation.'
                )
        if min(np.linalg.norm(start - point) for point in hub_endpoints) < (
            self.rearm_distance
        ):
            raise ValueError('Agent starts too close to a tunnel.')
        if self.num_leaves == 2:
            route_checks = list(range(self.num_leaves))
        else:
            route_checks = [
                self._search_order[0]
                if self._search_order
                else int(
                    np.argmin(np.linalg.norm(hub_endpoints - start, axis=1))
                )
            ]
        for intended in route_checks:
            for other in range(self.num_leaves):
                if intended == other:
                    continue
                if (
                    self._segment_distance(
                        start, hub_endpoints[intended], hub_endpoints[other]
                    )
                    <= self.trigger_radius
                ):
                    raise ValueError('A hub route crosses another tunnel.')
        if self.num_leaves >= 3 and self._triangle_area(hub_endpoints) < 120.0:
            raise ValueError('Hub tunnels are too close to collinear.')
        if self.num_leaves == 3 and self._validation_profile == 'ranked_route':
            self._validate_three_leaf_ranked_dynamics(hub_endpoints, start)
        elif self.num_leaves >= 3:
            self._validate_search_choice_margins(hub_endpoints, start)
        target_idx = self.num_leaves + self.target_leaf
        if np.linalg.norm(target - endpoints[target_idx]) < (
            self.trigger_radius + SUCCESS_DISTANCE
        ):
            raise ValueError('Target is too close to its tunnel.')

    def _validate_three_leaf_ranked_dynamics(
        self,
        hub_endpoints: np.ndarray,
        start: np.ndarray,
    ) -> None:
        """Require stable nearest-unexplored choices for all three ranks."""
        return_positions = np.stack(
            [
                self._landing_position(endpoint, hub_endpoints)
                for endpoint in self.hub_endpoint_indices()
            ]
        )
        order = nearest_unexplored_order(
            start,
            hub_endpoints,
            return_positions,
        )
        first, second, _third = order
        if (
            nearest_unexplored_hub(return_positions[first], hub_endpoints)
            != first
            or nearest_unexplored_hub(return_positions[second], hub_endpoints)
            != second
        ):
            raise ValueError(
                'Three-leaf nearest route does not repeat its last tunnel.'
            )
        if (
            nearest_unexplored_hub(
                return_positions[first], hub_endpoints, {first}
            )
            != second
        ):
            raise ValueError('Three-leaf second search choice is unstable.')
        if (
            nearest_unexplored_hub(
                return_positions[second], hub_endpoints, {second}
            )
            != first
        ):
            raise ValueError(
                'Three-leaf one-exclusion route does not return to the '
                'first tunnel.'
            )

        decision_sets = (
            (start, ()),
            (return_positions[first], (first,)),
            (return_positions[second], (second,)),
        )
        for position, excluded in decision_sets:
            candidates = [
                endpoint
                for endpoint in self.hub_endpoint_indices()
                if endpoint not in excluded
            ]
            distances = sorted(
                float(np.linalg.norm(position - hub_endpoints[endpoint]))
                for endpoint in candidates
            )
            if len(distances) > 1 and distances[1] - distances[0] < 5.0:
                raise ValueError(
                    'Three-leaf search choice margin is too small.'
                )

    def _validate_search_choice_margins(
        self, hub_endpoints: np.ndarray, start: np.ndarray
    ) -> None:
        """Require stable nearest-unexplored choices without a specific cycle."""
        return_positions = np.stack(
            [
                self._landing_position(endpoint, hub_endpoints)
                for endpoint in self.hub_endpoint_indices()
            ]
        )
        order = nearest_unexplored_order(
            start, hub_endpoints, return_positions
        )
        current = start
        explored: set[int] = set()
        for choice in order:
            candidates = [
                endpoint
                for endpoint in self.hub_endpoint_indices()
                if endpoint not in explored
            ]
            distances = sorted(
                float(np.linalg.norm(current - hub_endpoints[endpoint]))
                for endpoint in candidates
            )
            if len(distances) > 1 and distances[1] - distances[0] < 5.0:
                raise ValueError('Tunnel search choice margin is too small.')
            explored.add(choice)
            current = return_positions[choice]

    def _validate_sampling_constraints(
        self, constraints: dict[str, Any]
    ) -> None:
        if not constraints:
            return
        endpoints = np.asarray(
            self.variation_space['tunnel']['endpoints'].value,
            dtype=np.float32,
        )
        start = np.asarray(
            self.variation_space['agent']['position'].value,
            dtype=np.float32,
        )
        target = np.asarray(
            self.variation_space['target']['position'].value,
            dtype=np.float32,
        )
        target_idx = self.num_leaves + self.target_leaf
        route = self._route_length_estimates(start, target, endpoints)
        checks = {
            'start_to_nearest_hub_tunnel': min(
                float(np.linalg.norm(start - endpoint))
                for endpoint in endpoints[: self.num_leaves]
            ),
            'target_to_leaf_tunnel': float(
                np.linalg.norm(target - endpoints[target_idx])
            ),
            'visible_route_length': route['visible'],
            'hidden_route_length': route['hidden'],
        }
        for name, value in checks.items():
            lower = constraints.get(f'min_{name}')
            upper = constraints.get(f'max_{name}')
            if lower is not None and value < float(lower):
                raise ValueError(f'{name} is below configured minimum.')
            if upper is not None and value > float(upper):
                raise ValueError(f'{name} is above configured maximum.')

    def _route_length_estimates(
        self,
        start: np.ndarray,
        target: np.ndarray,
        endpoints: np.ndarray,
    ) -> dict[str, float]:
        target_idx = self.num_leaves + self.target_leaf
        hub_for_target = self._paired_endpoint(target_idx)
        target_landing = self._landing_position(target_idx, endpoints)
        visible = float(
            np.linalg.norm(start - endpoints[hub_for_target])
            + np.linalg.norm(target_landing - target)
        )

        current = start
        hidden = 0.0
        tested: set[int] = set()
        for hub_idx in self._hub_search_order(current, endpoints):
            leaf_idx = self._paired_endpoint(hub_idx)
            leaf_landing = self._landing_position(leaf_idx, endpoints)
            hidden += float(np.linalg.norm(current - endpoints[hub_idx]))
            if hub_idx == hub_for_target:
                hidden += float(np.linalg.norm(leaf_landing - target))
                break
            tested.add(hub_idx)
            leaf_rearm = self._rearm_position(leaf_idx, endpoints)
            hub_landing = self._landing_position(hub_idx, endpoints)
            hidden += float(np.linalg.norm(leaf_landing - leaf_rearm))
            hidden += float(np.linalg.norm(leaf_rearm - endpoints[leaf_idx]))
            current = hub_landing
            remaining = [
                idx for idx in self.hub_endpoint_indices() if idx not in tested
            ]
            if not remaining:
                break
        return {'visible': visible, 'hidden': hidden}

    def _hub_search_order(
        self, start: np.ndarray, endpoints: np.ndarray
    ) -> list[int]:
        return_positions = np.stack(
            [
                self._landing_position(endpoint, endpoints)
                for endpoint in self.hub_endpoint_indices()
            ]
        )
        return nearest_unexplored_order(
            start,
            endpoints[: self.num_leaves],
            return_positions,
        )

    def _paired_endpoint(self, endpoint_idx: int) -> int:
        perm = self._pairing_permutations[self.pairing]
        if endpoint_idx < self.num_leaves:
            return self.num_leaves + int(perm[endpoint_idx])
        leaf_offset = endpoint_idx - self.num_leaves
        return int(perm.index(leaf_offset))

    def _landing_position(
        self,
        endpoint_idx: int,
        endpoints: np.ndarray | None = None,
    ) -> np.ndarray:
        points = self.endpoint_positions if endpoints is None else endpoints
        direction = -1.0 if endpoint_idx < self.num_leaves else 1.0
        return points[endpoint_idx] + np.array(
            [direction * self.landing_offset, 0.0], dtype=np.float32
        )

    def _rearm_position(
        self,
        endpoint_idx: int,
        endpoints: np.ndarray | None = None,
    ) -> np.ndarray:
        points = self.endpoint_positions if endpoints is None else endpoints
        direction = -1.0 if endpoint_idx < self.num_leaves else 1.0
        return points[endpoint_idx] + np.array(
            [direction * self.rearm_distance, 0.0], dtype=np.float32
        )

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        old_position = self.agent_position.copy()
        proposed = old_position + action * self.speed
        moved = self._collide(proposed, self.room_index(old_position))

        endpoint = self._triggered_endpoint(old_position, moved)
        if endpoint is None:
            new_position = moved
        else:
            if endpoint < self.num_leaves:
                self._hub_entry_trace.append(int(endpoint))
            paired = self._paired_endpoint(endpoint)
            new_position = self._landing_position(paired)
            self._locked_endpoint = paired
            self.teleport_count += 1

        travelled = np.linalg.norm(moved - old_position)
        self.route_length += float(travelled)
        self.agent_position = new_position.astype(np.float32)
        self.step_count += 1
        self._maybe_rearm()
        self._terminated = self._is_success()
        return (
            self._get_obs(),
            0.0,
            self._terminated,
            False,
            self._get_info(),
        )

    def _collide(self, proposed: np.ndarray, room: int | None) -> np.ndarray:
        if room is None:
            raise RuntimeError('Agent is outside valid room geometry.')
        bounds = self._room_bounds(room)
        return np.clip(proposed, bounds[[0, 2]], bounds[[1, 3]]).astype(
            np.float32
        )

    def _triggered_endpoint(
        self, start: np.ndarray, end: np.ndarray
    ) -> int | None:
        room = self.room_index(start)
        if room == HUB:
            candidates = self.hub_endpoint_indices()
        elif room in self.leaf_rooms:
            candidates = (self.leaf_endpoint_for_room(room),)
        else:
            candidates = ()
        best: tuple[float, int] | None = None
        for idx in candidates:
            if idx == self._locked_endpoint:
                continue
            distance = self._segment_distance(
                start, end, self.endpoint_positions[idx]
            )
            if distance <= self.trigger_radius:
                candidate = (distance, idx)
                if best is None or candidate < best:
                    best = candidate
        return None if best is None else best[1]

    @staticmethod
    def _segment_distance(
        start: np.ndarray, end: np.ndarray, point: np.ndarray
    ) -> float:
        vector = end - start
        denom = float(np.dot(vector, vector))
        if denom <= 1e-12:
            return float(np.linalg.norm(point - start))
        fraction = float(np.dot(point - start, vector) / denom)
        closest = start + np.clip(fraction, 0.0, 1.0) * vector
        return float(np.linalg.norm(point - closest))

    @staticmethod
    def _triangle_area(points: np.ndarray) -> float:
        a, b, c = points[:3]
        ab = b - a
        ac = c - a
        return float(abs(ab[0] * ac[1] - ab[1] * ac[0]) / 2.0)

    def _maybe_rearm(self) -> None:
        if self._locked_endpoint is None:
            return
        endpoint = self.endpoint_positions[self._locked_endpoint]
        if (
            np.linalg.norm(self.agent_position - endpoint)
            >= self.rearm_distance
        ):
            self._locked_endpoint = None

    def _is_success(self) -> bool:
        return bool(
            np.linalg.norm(self.agent_position - self.target_position)
            < SUCCESS_DISTANCE
        )

    def _get_obs(self) -> dict[str, np.ndarray]:
        position = self.agent_position.astype(np.float32).copy()
        return {'proprio': position.copy(), 'state': position.copy()}

    def _get_info(self) -> dict[str, Any]:
        goal_position = self.target_position.astype(np.float32).copy()
        return {
            'env_name': 'TunnelRooms',
            'goal': self._goal.copy(),
            'goal_state': goal_position.copy(),
            'goal_proprio': goal_position.copy(),
            'pos_agent': self.agent_position.astype(np.float32).copy(),
            'pos_target': goal_position.copy(),
            'distance_to_target': np.float32(
                np.linalg.norm(self.agent_position - self.target_position)
            ),
        }

    def _set_state(self, state: np.ndarray) -> None:
        position = np.asarray(state, dtype=np.float32).squeeze()
        if self.room_index(position) is None:
            raise ValueError('State position is outside room geometry.')
        self.agent_position = position.copy()
        self._locked_endpoint = None
        self._terminated = self._is_success()

    def _set_goal_state(self, goal_state: np.ndarray) -> None:
        position = np.asarray(goal_state, dtype=np.float32).squeeze()
        if self.room_index(position) not in self.leaf_rooms:
            raise ValueError('Goal state must be in a leaf room.')
        self.target_position = position.copy()
        self._goal = self._render_frame(self.target_position)

    def render(self) -> np.ndarray:
        return self._render_frame(self.agent_position)

    def _render_frame(self, position: np.ndarray) -> np.ndarray:
        background = self.variation_space['appearance']['background'].value
        walls = self.variation_space['appearance']['walls'].value
        image = np.empty((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        image[:] = background
        border = BORDER_SIZE
        half_wall = WALL_WIDTH_DEFAULT // 2
        image[:border] = walls
        image[-border:] = walls
        image[:, :border] = walls
        image[:, -border:] = walls
        image[:, WALL_CENTER - half_wall : WALL_CENTER + half_wall] = walls
        for split_idx in range(1, self.num_leaves):
            y = leaf_split_y(split_idx, self.num_leaves)
            image[
                y - half_wall : y + half_wall,
                WALL_CENTER + half_wall : IMG_SIZE - border,
            ] = walls

        conditions = {
            HIDDEN: 'hidden',
            PAIRED_COLOR: 'paired_color',
            PAIRED_SHAPE: 'paired_shape',
        }
        scenario = {
            'endpoint_positions': self.endpoint_positions,
            'num_leaves': self.num_leaves,
            'pairing': self.pairing,
            'marker_size': float(
                self.variation_space['tunnel']['marker_size'].value[0]
            ),
            'marker_stroke_width': float(
                self.variation_space['tunnel']['marker_stroke_width'].value[0]
            ),
            'agent_render_radius': self.agent_render_radius,
        }
        return render_markers(
            image,
            scenario,
            conditions[self.identity_mode],
            position,
            marker_style=self.marker_style,
            colors=[
                self._marker_color(index)
                for index in range(len(self.endpoint_positions))
            ],
            background_color=background,
            marker_color=self.variation_space['tunnel']['hidden_color'].value,
            agent_color=self.variation_space['agent']['color'].value,
        )

    def _marker_color(self, endpoint_idx: int) -> np.ndarray:
        tunnel = self.variation_space['tunnel']
        if self.identity_mode in (HIDDEN, PAIRED_SHAPE):
            return tunnel['hidden_color'].value
        pair = self._marker_identity_index(endpoint_idx)
        return tunnel[PAIR_COLOR_FIELDS[self.cue_assignment[pair]]].value

    def _permutation_index(self, values: Any) -> int:
        """Return the canonical index for one complete tunnel permutation."""
        permutation = tuple(int(value) for value in values)
        try:
            return self._pairing_permutations.index(permutation)
        except ValueError as error:
            raise ValueError(
                f'Expected a permutation of 0..{self.num_leaves - 1}, '
                f'got {permutation}.'
            ) from error

    def _marker_identity_index(self, endpoint_idx: int) -> int:
        if endpoint_idx < self.num_leaves:
            return endpoint_idx
        return self._paired_endpoint(endpoint_idx)

    def observable_tunnel_positions(self) -> np.ndarray:
        return self.endpoint_positions.copy()

    def visible_hub_endpoint_for_leaf(self, leaf_room: int) -> int:
        if self.identity_mode not in (PAIRED_COLOR, PAIRED_SHAPE):
            raise RuntimeError('Hidden tunnel pairing is not observable.')
        endpoint = self.leaf_endpoint_for_room(leaf_room)
        return self._paired_endpoint(endpoint)

    def scenario_spec(self) -> dict[str, Any]:
        target_rank = self._target_rank
        rank_labels: dict[str, Any] = {}
        if target_rank is not None:
            rank_labels = {
                'target_rank': int(target_rank),
                'correct_hub_endpoint': int(self._correct_hub_endpoint),
                'search_order': list(self._search_order or []),
            }
        scenario = {
            'layout': self.layout,
            'num_leaves': self.num_leaves,
            'scenario_seed': self._episode_seed,
            'identity_mode': self.identity_mode,
            'marker_style': self.marker_style,
            'marker_width': self.marker_width,
            'marker_height': self.marker_height,
            'pairing': self.pairing,
            **(
                {'cue_assignment': list(self.cue_assignment)}
                if self._validation_profile == 'fixed_layout'
                or self.cue_assignment != tuple(range(self.num_leaves))
                else {}
            ),
            'endpoint_positions': self.endpoint_positions.tolist(),
            'agent_position': self._initial_agent_position.tolist(),
            'agent_render_radius': self.agent_render_radius,
            'target_position': self.target_position.tolist(),
            'target_leaf': self.target_leaf,
            'trigger_radius': self.trigger_radius,
            'landing_offset': self.landing_offset,
            'rearm_distance': self.rearm_distance,
            'min_hub_separation': self.min_hub_separation,
            'speed': self.speed,
            **(
                {'validation_profile': self._validation_profile}
                if self._validation_profile != 'ranked_route'
                else {}
            ),
            **rank_labels,
        }
        # Record only non-default palettes so historical scenario JSON is stable.
        tunnel = self.variation_space['tunnel']
        colors = [
            tunnel[name].value.tolist()
            for name in PAIR_COLOR_FIELDS[: self.num_leaves]
        ]
        default_colors = [
            color.tolist() for color in SEMANTIC_COLORS[: self.num_leaves]
        ]
        # TR4 has no historical implicit palette contract, so persist its new
        # four-color palette even when it matches today's source defaults.
        if self.num_leaves == 4 or colors != default_colors:
            scenario['tunnel_colors'] = colors
        default_names = SEMANTIC_COLOR_NAMES[: self.num_leaves]
        if self.num_leaves == 4 or self._tunnel_color_names != default_names:
            scenario['tunnel_color_names'] = list(self._tunnel_color_names)
        return scenario

    def audit_metadata(self) -> dict[str, Any]:
        metadata = deepcopy(self.scenario_spec())
        metadata.update(
            {
                'teleport_count': self.teleport_count,
                'success': self._terminated,
                'route_length': self.route_length,
                'steps': self.step_count,
                'hub_entry_trace': list(self._hub_entry_trace),
            }
        )
        return metadata
