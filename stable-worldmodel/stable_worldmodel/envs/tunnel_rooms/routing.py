"""Pure routing helpers for Tunnel Rooms policies and audits."""

from __future__ import annotations

import itertools
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class RankedRoute:
    """Geometry-derived nearest-unexplored route metadata."""

    search_order: tuple[int, ...]
    correct_hub_endpoint: int
    target_rank: int


def nearest_unexplored_hub(
    position: np.ndarray,
    hub_endpoints: np.ndarray,
    explored: Collection[int] = (),
) -> int:
    """Return the nearest unexplored hub endpoint with stable tie-breaking."""
    point = np.asarray(position, dtype=np.float32)
    endpoints = np.asarray(hub_endpoints, dtype=np.float32)
    if endpoints.ndim != 2 or endpoints.shape[1:] != point.shape:
        raise ValueError(
            'Hub endpoints and position have incompatible shapes.'
        )

    explored_set = {int(idx) for idx in explored}
    candidates = [
        idx for idx in range(len(endpoints)) if idx not in explored_set
    ]
    if not candidates:
        raise ValueError('No unexplored hub endpoints remain.')
    return min(
        candidates,
        key=lambda idx: (float(np.linalg.norm(point - endpoints[idx])), idx),
    )


def nearest_unexplored_order(
    initial_position: np.ndarray,
    hub_endpoints: np.ndarray,
    return_positions: np.ndarray,
) -> list[int]:
    """Compute the online search order after each failed tunnel return."""
    endpoints = np.asarray(hub_endpoints, dtype=np.float32)
    returns = np.asarray(return_positions, dtype=np.float32)
    if returns.shape != endpoints.shape:
        raise ValueError('Each hub endpoint requires one return position.')

    current = np.asarray(initial_position, dtype=np.float32)
    explored: set[int] = set()
    order: list[int] = []
    while len(explored) < len(endpoints):
        choice = nearest_unexplored_hub(current, endpoints, explored)
        order.append(choice)
        explored.add(choice)
        current = returns[choice]
    return order


def certify_ranked_route(scenario: Mapping[str, Any]) -> RankedRoute:
    """Derive ranked-search metadata without trusting stored rank labels."""
    num_leaves = int(scenario['num_leaves'])
    if num_leaves < 1:
        raise ValueError('Ranked Tunnel Rooms requires at least one leaf.')

    endpoints = np.asarray(scenario['endpoint_positions'], dtype=np.float32)
    expected_shape = (2 * num_leaves, 2)
    if endpoints.shape != expected_shape or not np.isfinite(endpoints).all():
        raise ValueError(
            'endpoint_positions must contain exactly two finite points per leaf.'
        )
    start = np.asarray(scenario['agent_position'], dtype=np.float32)
    if start.shape != (2,) or not np.isfinite(start).all():
        raise ValueError('agent_position must be one finite point.')
    landing_offset = float(scenario['landing_offset'])
    if not np.isfinite(landing_offset) or landing_offset <= 0.0:
        raise ValueError('landing_offset must be finite and positive.')

    hub_endpoints = endpoints[:num_leaves]
    return_positions = hub_endpoints - np.asarray(
        [landing_offset, 0.0], dtype=np.float32
    )
    order = nearest_unexplored_order(start, hub_endpoints, return_positions)

    permutations = tuple(itertools.permutations(range(num_leaves)))
    pairing = int(scenario['pairing'])
    if pairing not in range(len(permutations)):
        raise ValueError('pairing is outside the permutation table.')
    target_leaf = int(scenario['target_leaf'])
    if target_leaf not in range(num_leaves):
        raise ValueError('target_leaf is outside the leaf range.')
    correct = int(permutations[pairing].index(target_leaf))
    return RankedRoute(
        search_order=tuple(int(endpoint) for endpoint in order),
        correct_hub_endpoint=correct,
        target_rank=order.index(correct) + 1,
    )
