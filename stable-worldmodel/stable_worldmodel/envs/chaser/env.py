"""Two-agent benchmark for rare cue-conditioned visual dynamics.

The dominant motion is smooth: a red controlled chaser follows a colored target
that travels around a perturbed orbit. Contact creates the rare discontinuity.
In ``reflection`` mode, geometry determines the red agent's destination. In
``cue`` mode, the target's persistent color determines the quadrant rotation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


IMAGE_SIZE = 224
ARENA_CENTER = np.array([112.0, 112.0], dtype=np.float32)
ARENA_INSET = 14.0
RED_SPEED = 5.0
BLUE_SPEED = 3.0
RED_RADIUS = 8.0
BLUE_RADIUS = 11.0
# Require visible overlap instead of teleporting as soon as the disks touch.
CONTACT_DISTANCE = (RED_RADIUS + BLUE_RADIUS) / 2
REARM_DISTANCE = CONTACT_DISTANCE + 8.0
RUNNER_NOISE_DECAY = 0.995
DEFAULT_RUNNER_NOISE_STD = 1.0
MAX_RUNNER_HEADING_NOISE = 1.0
MAPPING_MODES = ('reflection', 'cue')

# Target color is the persistent cue. Positive angles rotate clockwise in
# screen coordinates because image y increases downward. Blue sends the red
# agent to the opposite quadrant; green and purple make the two quarter-turns.
CUE_NAMES = ('blue', 'green', 'purple')
CUE_COLORS = (
    (47, 101, 214),
    (35, 155, 86),
    (139, 75, 191),
)
CUE_ROTATIONS = np.array((np.pi, np.pi / 2, -np.pi / 2), dtype=np.float32)


class ChaserEnv(gym.Env):
    """Continuing 224x224 Chaser environment.

    Episode length is owned by Gymnasium's ``TimeLimit`` wrapper.  This class
    never terminates on contact because repeated chase/contact cycles are the
    data-generating process of interest.
    """

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 10}

    def __init__(
        self,
        render_mode: str = 'rgb_array',
        mapping_mode: str = 'cue',
        runner_noise_std: float = DEFAULT_RUNNER_NOISE_STD,
        cue_ids: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if render_mode != 'rgb_array':
            raise ValueError('ChaserEnv supports only rgb_array rendering.')
        if mapping_mode not in MAPPING_MODES:
            raise ValueError(
                f'mapping_mode must be one of {MAPPING_MODES}; got {mapping_mode!r}.'
            )
        if not np.isfinite(runner_noise_std) or runner_noise_std < 0:
            raise ValueError(
                'runner_noise_std must be finite and non-negative.'
            )
        self.render_mode = render_mode
        self.mapping_mode = mapping_mode
        self.runner_noise_std = float(runner_noise_std)
        self.cue_ids = (
            tuple(range(len(CUE_COLORS)))
            if cue_ids is None
            else tuple(int(cue_id) for cue_id in cue_ids)
        )
        if not self.cue_ids or len(set(self.cue_ids)) != len(self.cue_ids):
            raise ValueError('cue_ids must be non-empty and duplicate-free.')
        if not set(self.cue_ids) <= set(range(len(CUE_COLORS))):
            raise ValueError(
                f'cue_ids must be selected from {list(range(len(CUE_COLORS)))}.'
            )
        self.red_speed = RED_SPEED
        self.blue_speed = BLUE_SPEED
        self.observation_space = spaces.Dict(
            {
                'proprio': spaces.Box(
                    0.0, float(IMAGE_SIZE), shape=(2,), dtype=np.float32
                ),
                'state': spaces.Box(
                    0.0, float(IMAGE_SIZE), shape=(4,), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Sample or replay one deterministic Chaser scenario."""
        super().reset(seed=seed)
        options = deepcopy(options or {})
        unknown = set(options) - {'scenario'}
        if unknown:
            raise ValueError(
                f'Unknown Chaser reset options: {sorted(unknown)}'
            )

        # 1. Sample the smooth orbit and an independent chaser start. Varying
        # initial separation spreads rare contacts across the episode.
        scenario = options.get('scenario') or {}
        scenario_mode = scenario.get('mapping_mode', self.mapping_mode)
        if scenario_mode != self.mapping_mode:
            raise ValueError(
                'The replay scenario mapping_mode must match the environment.'
            )
        scenario_noise = float(
            scenario.get('runner_noise_std', self.runner_noise_std)
        )
        if not np.isclose(scenario_noise, self.runner_noise_std):
            raise ValueError(
                'The replay scenario runner_noise_std must match the environment.'
            )
        scenario_cues = tuple(scenario.get('cue_ids', self.cue_ids))
        if scenario_cues != self.cue_ids:
            raise ValueError(
                'The replay scenario cue_ids must match the environment.'
            )
        self.scenario_seed = int(
            scenario.get(
                'scenario_seed',
                seed
                if seed is not None
                else self.np_random.integers(0, np.iinfo(np.int32).max),
            )
        )
        self.orbit_direction = int(
            scenario.get('orbit_direction', self.np_random.choice((-1, 1)))
        )
        if self.orbit_direction not in (-1, 1):
            raise ValueError('orbit_direction must be -1 or 1.')
        self.orbit_radius = float(
            scenario.get('orbit_radius', self.np_random.uniform(68.0, 72.0))
        )
        self.orbit_phase = float(
            scenario.get('orbit_phase', self.np_random.uniform(0.0, 2 * np.pi))
        )
        self.runner_noise_seed = int(
            scenario.get(
                'runner_noise_seed',
                np.random.SeedSequence([self.scenario_seed, 0x424C5545])
                .generate_state(1)
                .item(),
            )
        )
        self._runner_noise_rng = np.random.default_rng(self.runner_noise_seed)
        self._runner_noise_ema = float(
            self._runner_noise_rng.normal(0.0, self.runner_noise_std)
        )
        default_blue = ARENA_CENTER + self.orbit_radius * np.array(
            [np.cos(self.orbit_phase), np.sin(self.orbit_phase)], np.float32
        )
        position_low = ARENA_INSET + BLUE_RADIUS
        default_red = self.np_random.uniform(
            position_low, IMAGE_SIZE - position_low, size=2
        ).astype(np.float32)
        self.blue_position = self._position(
            scenario.get('blue_position', default_blue), 'blue_position'
        )
        self.red_position = self._position(
            scenario.get('red_position', default_red), 'red_position'
        )

        # 2. Draw the cue from an independent deterministic random stream.
        # This prevents seed-correlated geometry from biasing cue/quadrant
        # cells while preserving exact replay through ``scenario_seed``.
        cue_rng = np.random.default_rng(
            np.random.SeedSequence([self.scenario_seed, 0x435545])
        )
        default_cue = int(cue_rng.choice(self.cue_ids))
        self.cue_id = int(
            scenario.get(
                'cue_id', default_cue if self.mapping_mode == 'cue' else -1
            )
        )
        expected_cues = self.cue_ids if self.mapping_mode == 'cue' else (-1,)
        if self.cue_id not in expected_cues:
            raise ValueError(
                f'cue_id must be {list(expected_cues)} in {self.mapping_mode} mode.'
            )

        # 3. Clear event state. Empty metadata uses numeric sentinels so HDF5
        # keeps one stable schema across ordinary and event transitions.
        self.step_count = 0
        self.event_count = 0
        self._contact_locked = False
        self._last_event = False
        self._last_contact_quadrant = -1
        self._last_destination_quadrant = -1
        self._last_pre_contact_red = np.full(2, np.nan, dtype=np.float32)
        self._last_pre_contact_blue = np.full(2, np.nan, dtype=np.float32)
        self._last_teleport_destination = np.full(2, np.nan, dtype=np.float32)
        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Advance smooth motion, then apply at most one swept contact event."""
        if not hasattr(self, 'red_position'):
            raise RuntimeError('Call reset() before step().')

        # 1. Interpret every non-zero action as a direction. This keeps the red
        # chaser at its fixed five-pixel speed even when policy noise changes
        # the requested heading.
        action = np.asarray(action, dtype=np.float32).reshape(2)
        action = np.clip(action, -1.0, 1.0)
        action_norm = float(np.linalg.norm(action))
        if action_norm > 1e-8:
            action /= action_norm
        red_start = self.red_position.copy()
        blue_start = self.blue_position.copy()
        red_end = self._clip_position(red_start + action * self.red_speed)
        blue_end = self._next_blue_position()

        # 2. Detect contact along relative motion, so two agents cannot cross
        # between rendered frames without producing the rare transition.
        self._clear_last_event()
        contact = self._swept_contact_fraction(
            red_start - blue_start,
            red_end - blue_end,
            CONTACT_DISTANCE,
        )
        if not self._contact_locked and contact is not None:
            self._record_contact(red_start, blue_start, blue_end, contact)
            red_end = self._last_teleport_destination.copy()
            self._contact_locked = True

        # 3. Commit state, then re-arm only after the teleported chaser and
        # target have become safely separated.
        self.red_position = red_end.astype(np.float32)
        self.blue_position = blue_end.astype(np.float32)
        self.step_count += 1
        if (
            self._contact_locked
            and np.linalg.norm(self.red_position - self.blue_position)
            >= REARM_DISTANCE
        ):
            self._contact_locked = False
        return self._get_obs(), 0.0, False, False, self._get_info()

    def render(self) -> np.ndarray:
        """Rasterize the arena and the two agents."""
        image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 255, dtype=np.uint8)
        border = int(ARENA_INSET // 2)
        image[:border] = 30
        image[-border:] = 30
        image[:, :border] = 30
        image[:, -border:] = 30
        self._draw_disk(image, self.red_position, RED_RADIUS, (213, 38, 45))
        target_color = (
            CUE_COLORS[self.cue_id]
            if self.mapping_mode == 'cue'
            else CUE_COLORS[0]
        )
        self._draw_disk(image, self.blue_position, BLUE_RADIUS, target_color)
        return image

    def scenario_spec(self) -> dict[str, Any]:
        """Return the exact reset state needed for deterministic replay."""
        return {
            'scenario_seed': self.scenario_seed,
            'mapping_mode': self.mapping_mode,
            'red_position': self.red_position.tolist(),
            'blue_position': self.blue_position.tolist(),
            'cue_id': self.cue_id,
            'cue_ids': self.cue_ids,
            'orbit_direction': self.orbit_direction,
            'orbit_radius': self.orbit_radius,
            'orbit_phase': self.orbit_phase,
            'runner_noise_seed': self.runner_noise_seed,
            'runner_noise_std': self.runner_noise_std,
        }

    def _next_blue_position(self) -> np.ndarray:
        """Follow a smooth perturbed orbit without wrapping at boundaries."""
        radial = self.blue_position - ARENA_CENTER
        radius = max(float(np.linalg.norm(radial)), 1e-6)
        radial_unit = radial / radius
        tangent = self.orbit_direction * np.array(
            [-radial_unit[1], radial_unit[0]], dtype=np.float32
        )
        # Add strong correlated heading noise to the large runner. Sampling the
        # initial EMA from stationarity makes wandering visible immediately;
        # temporal filtering keeps it smooth and speed remains constant.
        innovation = float(self._runner_noise_rng.normal())
        self._runner_noise_ema = (
            RUNNER_NOISE_DECAY * self._runner_noise_ema
            + self.runner_noise_std
            * np.sqrt(1.0 - RUNNER_NOISE_DECAY**2)
            * innovation
        )
        heading_noise = float(
            np.clip(
                self._runner_noise_ema,
                -MAX_RUNNER_HEADING_NOISE,
                MAX_RUNNER_HEADING_NOISE,
            )
        )
        # Decompose motion into orthogonal tangent/radial components. The
        # random radial component makes the orbit visibly irregular, while a
        # bounded restoring term prevents persistent noise reaching the center
        # or arena edge. Keeping a positive tangent component prevents reversals.
        tangential_strength = float(np.cos(heading_noise))
        random_radial = -self.orbit_direction * float(np.sin(heading_noise))
        radial_strength = float(
            np.clip(
                random_radial - 0.12 * (radius - self.orbit_radius),
                -1.0,
                1.0,
            )
        )
        direction = (
            tangential_strength * tangent + radial_strength * radial_unit
        )
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        return self._clip_position(
            self.blue_position + self.blue_speed * direction
        )

    def _record_contact(
        self,
        red_start: np.ndarray,
        blue_start: np.ndarray,
        blue_end: np.ndarray,
        fraction: float,
    ) -> None:
        """Record privileged event metadata and resolve the teleport mapping."""
        contact_blue = blue_start + fraction * (blue_end - blue_start)
        quadrant = self._quadrant(contact_blue)
        destination = self._teleport_destination(contact_blue)
        self._last_event = True
        self._last_contact_quadrant = quadrant
        self._last_destination_quadrant = self._quadrant(destination)
        self._last_pre_contact_red = red_start.astype(np.float32).copy()
        self._last_pre_contact_blue = blue_start.astype(np.float32).copy()
        self._last_teleport_destination = destination
        self.event_count += 1

    def _teleport_destination(self, contact_blue: np.ndarray) -> np.ndarray:
        """Rotate the contact point while preserving its center-relative radius."""
        # 1. Select the quadrant rotation. Reflection and the blue cue both use
        # a half-turn; green and purple use the two distinct quarter-turns.
        rotation = (
            float(CUE_ROTATIONS[self.cue_id])
            if self.mapping_mode == 'cue'
            else np.pi
        )

        # 2. Rotate the full radial vector, not a normalized direction. This
        # preserves the exact radius at which contact occurred while changing
        # only the target quadrant prescribed by the cue.
        radial = np.asarray(contact_blue, dtype=np.float32) - ARENA_CENTER
        destination = ARENA_CENTER + self._rotate(radial, rotation)

        # 3. Fail loudly if future dynamics changes make the mapping unsafe;
        # clipping would silently break the radius-preservation contract.
        return self._position(destination, 'teleport_destination')

    def _get_obs(self) -> dict[str, np.ndarray]:
        red = self.red_position.astype(np.float32).copy()
        state = np.concatenate((red, self.blue_position)).astype(np.float32)
        return {'proprio': red, 'state': state}

    def _get_info(self) -> dict[str, Any]:
        return {
            'env_name': 'Chaser',
            'pos_agent': self.red_position.astype(np.float32).copy(),
            'pos_red': self.red_position.astype(np.float32).copy(),
            'pos_blue': self.blue_position.astype(np.float32).copy(),
            'distance_to_target': np.asarray(
                np.linalg.norm(self.red_position - self.blue_position),
                dtype=np.float32,
            ),
            # Scalar audit values are zero-dimensional arrays because
            # World.collect intentionally persists array/tensor columns only.
            'event': np.asarray(self._last_event, dtype=np.bool_),
            'event_count': np.asarray(self.event_count, dtype=np.int32),
            'contact_quadrant': np.asarray(
                self._last_contact_quadrant, dtype=np.int8
            ),
            'destination_quadrant': np.asarray(
                self._last_destination_quadrant, dtype=np.int8
            ),
            'cue_id': np.asarray(self.cue_id, dtype=np.int8),
            'pre_contact_red': self._last_pre_contact_red.copy(),
            'pre_contact_blue': self._last_pre_contact_blue.copy(),
            'teleport_destination': self._last_teleport_destination.copy(),
            'orbit_direction': np.asarray(self.orbit_direction, dtype=np.int8),
            'orbit_radius': np.asarray(self.orbit_radius, dtype=np.float32),
            'orbit_phase': np.asarray(self.orbit_phase, dtype=np.float32),
            'runner_noise_seed': np.asarray(
                self.runner_noise_seed, dtype=np.uint32
            ),
            'runner_noise_std': np.asarray(
                self.runner_noise_std, dtype=np.float32
            ),
            'scenario_seed': np.asarray(self.scenario_seed, dtype=np.int64),
        }

    def _clear_last_event(self) -> None:
        self._last_event = False
        self._last_contact_quadrant = -1
        self._last_destination_quadrant = -1
        self._last_pre_contact_red.fill(np.nan)
        self._last_pre_contact_blue.fill(np.nan)
        self._last_teleport_destination.fill(np.nan)

    @staticmethod
    def _quadrant(position: np.ndarray) -> int:
        right = bool(position[0] >= ARENA_CENTER[0])
        bottom = bool(position[1] >= ARENA_CENTER[1])
        return ((1 if right else 0), (2 if right else 3))[bottom]

    @staticmethod
    def _swept_contact_fraction(
        relative_start: np.ndarray,
        relative_end: np.ndarray,
        contact_distance: float,
    ) -> float | None:
        """Return first segment/sphere intersection fraction, if present."""
        delta = relative_end - relative_start
        a = float(np.dot(delta, delta))
        c = float(np.dot(relative_start, relative_start) - contact_distance**2)
        if c <= 0:
            return 0.0
        if a <= 1e-12:
            return None
        b = 2.0 * float(np.dot(relative_start, delta))
        discriminant = b * b - 4.0 * a * c
        if discriminant < 0:
            return None
        root = (-b - np.sqrt(discriminant)) / (2.0 * a)
        return float(root) if 0.0 <= root <= 1.0 else None

    @staticmethod
    def _rotate(vector: np.ndarray, angle: float) -> np.ndarray:
        """Rotate a 2D vector without changing its magnitude."""
        cosine, sine = np.cos(angle), np.sin(angle)
        return np.array(
            [
                cosine * vector[0] - sine * vector[1],
                sine * vector[0] + cosine * vector[1],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _position(value: Any, name: str) -> np.ndarray:
        position = np.asarray(value, dtype=np.float32).reshape(2)
        low = ARENA_INSET + BLUE_RADIUS
        high = IMAGE_SIZE - low
        if np.any(position < low) or np.any(position > high):
            raise ValueError(f'{name} must lie inside [{low}, {high}].')
        return position.copy()

    @staticmethod
    def _clip_position(position: np.ndarray) -> np.ndarray:
        low = ARENA_INSET + BLUE_RADIUS
        return np.clip(position, low, IMAGE_SIZE - low).astype(np.float32)

    @staticmethod
    def _draw_disk(
        image: np.ndarray,
        center: np.ndarray,
        radius: float,
        color: tuple[int, int, int],
    ) -> None:
        y, x = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
        mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2
        image[mask] = color
