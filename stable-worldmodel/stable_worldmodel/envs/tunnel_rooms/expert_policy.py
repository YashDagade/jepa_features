"""Non-privileged prototype policies for Tunnel Rooms."""

from __future__ import annotations

from typing import Any

import numpy as np

from stable_worldmodel.policy import BasePolicy

from .env import HUB, PAIRED_COLOR, PAIRED_SHAPE
from .routing import nearest_unexplored_hub


class TunnelSearchPolicy(BasePolicy):
    """Search tunnels using only visible locations and observed outcomes."""

    def __init__(
        self,
        action_noise: float = 0.0,
        action_repeat_prob: float = 0.0,
        seed: int | None = None,
    ):
        super().__init__()
        self.type = 'tunnel_search'
        self.action_noise = float(action_noise)
        self.action_repeat_prob = float(action_repeat_prob)
        self.rng = np.random.default_rng(seed)
        # Track tried hub endpoints, the current candidate, and whether the
        # arrival portal has been re-armed for a return trip in each env.
        self._tested: list[set[int]] = []
        self._candidate: list[int | None] = []
        self._return_ready: list[bool] = []
        self._last_action: np.ndarray | None = None
        self._last_action_valid = np.empty(0, dtype=bool)

    def set_env(self, env: Any) -> None:
        self.env = env
        count = len(self._base_envs()[0])
        self._tested = [set() for _ in range(count)]
        self._candidate = [None for _ in range(count)]
        self._return_ready = [False for _ in range(count)]
        self._last_action = None
        self._last_action_valid = np.zeros(count, dtype=bool)

    def reset(self, mask: np.ndarray | None = None) -> None:
        """Clear search and action-repeat state for selected environments."""
        selected = self._reset_mask(mask, len(self._tested))
        for idx in np.flatnonzero(selected):
            env_idx = int(idx)
            self._tested[env_idx].clear()
            self._candidate[env_idx] = None
            self._return_ready[env_idx] = False
        self._last_action_valid[selected] = False
        if selected.all():
            self._last_action = None

    def get_action(
        self, info_dict: dict[str, Any], **kwargs: Any
    ) -> np.ndarray:
        envs, vectorized = self._base_envs()
        shape = self.env.action_space.shape
        actions = np.zeros(shape, dtype=np.float32)
        repeat_allowed = self._last_action_valid.copy()

        for idx, env in enumerate(envs):
            position = self._value(info_dict, 'state', idx, vectorized)
            target = self._value(info_dict, 'goal_state', idx, vectorized)
            room = env.room_index(position)
            target_room = env.room_index(target)
            tunnels = env.observable_tunnel_positions()

            if room == target_room:
                waypoint = target
            elif room == HUB:
                self._return_ready[idx] = False
                choice = self._choose_hub_tunnel(idx, env, position)
                self._candidate[idx] = choice
                waypoint = self._hub_waypoint(env, position, tunnels, choice)
            else:
                if self._candidate[idx] is not None:
                    self._remember_failed(idx, self._candidate[idx])
                leaf_idx = env.leaf_endpoint_for_room(room)
                leaf_endpoint = tunnels[leaf_idx]
                if (
                    np.linalg.norm(position - leaf_endpoint)
                    >= env.rearm_distance
                ):
                    self._return_ready[idx] = True
                if not self._return_ready[idx]:
                    waypoint = leaf_endpoint + np.array(
                        [env.rearm_distance + env.speed, 0], np.float32
                    )
                else:
                    waypoint = leaf_endpoint

            action = self._direction(position, waypoint)
            if vectorized:
                actions[idx] = action
            else:
                actions = action

        if self.action_noise > 0:
            actions += self.rng.normal(
                0.0, self.action_noise, actions.shape
            ).astype(np.float32)
        actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
        actions = self._maybe_repeat_action(
            actions, vectorized, repeat_allowed
        )
        self._last_action = np.array(actions, copy=True)
        self._last_action_valid[:] = True
        return actions

    def _remember_failed(self, idx: int, endpoint: int) -> None:
        self._tested[idx].add(int(endpoint))

    def _maybe_repeat_action(
        self, actions: np.ndarray, vectorized: bool, repeat_allowed: np.ndarray
    ) -> np.ndarray:
        """Optionally repeat previous actions, matching Two Rooms collection."""
        if self._last_action is None or self.action_repeat_prob <= 0.0:
            return actions
        if vectorized:
            repeat = self.rng.uniform(0.0, 1.0, size=(actions.shape[0],))
            repeat_mask = repeat_allowed & (repeat < self.action_repeat_prob)
            actions[repeat_mask] = self._last_action[repeat_mask]
        elif (
            bool(repeat_allowed[0])
            and self.rng.uniform(0.0, 1.0) < self.action_repeat_prob
        ):
            actions = self._last_action
        return actions

    def _choose_hub_tunnel(
        self, idx: int, env: Any, position: np.ndarray
    ) -> int:
        """Pick the nearest hub endpoint this env has not already tested."""
        if env.identity_mode in (PAIRED_COLOR, PAIRED_SHAPE):
            target_room = env.leaf_room_for_target_leaf(env.target_leaf)
            return env.visible_hub_endpoint_for_leaf(target_room)
        candidates = [
            endpoint
            for endpoint in env.hub_endpoint_indices()
            if endpoint not in self._tested[idx]
        ]
        if not candidates:
            candidates = list(env.hub_endpoint_indices())
        tunnels = env.observable_tunnel_positions()
        excluded = set(env.hub_endpoint_indices()) - set(candidates)
        return nearest_unexplored_hub(
            position,
            tunnels[list(env.hub_endpoint_indices())],
            excluded,
        )

    def _hub_waypoint(
        self,
        env: Any,
        position: np.ndarray,
        tunnels: np.ndarray,
        choice: int,
    ) -> np.ndarray:
        """Approach hub tunnels via the left lane to avoid accidental portals."""
        target = tunnels[choice]
        blockers = [
            endpoint
            for endpoint in env.hub_endpoint_indices()
            if endpoint != choice
        ]
        crosses_blocker = any(
            env._segment_distance(position, target, tunnels[endpoint])
            <= env.trigger_radius
            for endpoint in blockers
        )
        if not crosses_blocker:
            return target
        bounds = env._room_bounds(HUB)
        safe_x = np.float32(bounds[0] + env.agent_radius + env.speed)
        if abs(float(position[0] - safe_x)) > env.speed:
            return np.array([safe_x, position[1]], dtype=np.float32)
        return np.array([safe_x, target[1]], dtype=np.float32)

    def _base_envs(self) -> tuple[list[Any], bool]:
        """Return raw TunnelRoomsEnv instances from raw or vectorized wrappers."""
        if hasattr(self.env, 'envs'):
            return [item.unwrapped for item in self.env.envs], True
        base = getattr(self.env, 'unwrapped', self.env)
        if hasattr(base, 'envs'):
            return [item.unwrapped for item in base.envs], True
        return [base], False

    @staticmethod
    def _direction(position: np.ndarray, waypoint: np.ndarray) -> np.ndarray:
        """Unit action direction from the current position to a waypoint."""
        direction = np.asarray(waypoint, np.float32) - np.asarray(
            position, np.float32
        )
        norm = float(np.linalg.norm(direction))
        return direction / norm if norm > 1e-8 else np.zeros(2, np.float32)

    @staticmethod
    def _value(
        info_dict: dict[str, Any], key: str, idx: int, vectorized: bool
    ) -> np.ndarray:
        """Read a scalar-env or vector-env info value as a flat float array."""
        value = info_dict[key][idx] if vectorized else info_dict[key]
        return np.asarray(value, dtype=np.float32).squeeze()


class VisibleRoutingPolicy(TunnelSearchPolicy):
    """Backward-compatible name for cue-aware tunnel search."""

    def __init__(
        self,
        action_noise: float = 0.0,
        action_repeat_prob: float = 0.0,
        seed: int | None = None,
    ):
        super().__init__(
            action_noise=action_noise,
            action_repeat_prob=action_repeat_prob,
            seed=seed,
        )
        self.type = 'visible_routing'

    def _choose_hub_tunnel(
        self, idx: int, env: Any, position: np.ndarray
    ) -> int:
        if env.identity_mode not in (PAIRED_COLOR, PAIRED_SHAPE):
            raise RuntimeError(
                'VisibleRoutingPolicy requires paired_color mode or '
                'paired_shape mode.'
            )
        return super()._choose_hub_tunnel(idx, env, position)


class VonMisesCoveragePolicy(BasePolicy):
    """PLDM-style correlated random coverage policy for offline data."""

    def __init__(
        self,
        concentration: float = 5.0,
        max_step_size: float = 2.45,
        seed: int | None = None,
    ):
        super().__init__()
        self.type = 'von_mises_coverage'
        self.concentration = float(concentration)
        self.max_step_size = float(max_step_size)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._directions = np.empty(0, dtype=np.float32)

    def set_seed(self, seed: int | None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def set_env(self, env: Any) -> None:
        self.env = env
        count = len(self._base_envs()[0])
        self._directions = np.full(count, np.nan, dtype=np.float32)

    def reset(self, mask: np.ndarray | None = None) -> None:
        """Clear correlated headings for selected environments."""
        selected = self._reset_mask(mask, len(self._directions))
        self._directions[selected] = np.nan

    def get_action(
        self, info_dict: dict[str, Any], **kwargs: Any
    ) -> np.ndarray:
        envs, vectorized = self._base_envs()
        actions = np.zeros(self.env.action_space.shape, dtype=np.float32)
        for idx, env in enumerate(envs):
            if np.isnan(self._directions[idx]):
                angle = self.rng.uniform(-np.pi, np.pi)
            else:
                angle = self.rng.vonmises(
                    self._directions[idx], self.concentration
                )
            self._directions[idx] = angle
            step_size = self.rng.uniform(0.0, self.max_step_size)
            physical_action = step_size * np.array(
                [np.cos(angle), np.sin(angle)], dtype=np.float32
            )
            action = physical_action / env.speed
            if vectorized:
                actions[idx] = action
            else:
                actions = action
        return np.clip(actions, -1.0, 1.0).astype(np.float32)

    def _base_envs(self) -> tuple[list[Any], bool]:
        """Return raw TunnelRoomsEnv instances from raw or vectorized wrappers."""
        if hasattr(self.env, 'envs'):
            return [item.unwrapped for item in self.env.envs], True
        base = getattr(self.env, 'unwrapped', self.env)
        if hasattr(base, 'envs'):
            return [item.unwrapped for item in base.envs], True
        return [base], False
