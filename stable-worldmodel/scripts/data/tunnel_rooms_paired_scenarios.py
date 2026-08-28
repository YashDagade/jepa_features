"""Own Tunnel Rooms scenarios, expert policies, and rollout audits.

Training scenarios follow the natural target-independent stream. Evaluation
scenarios come from that same sampling process and are selected afterward to
provide equal support for each realized target rank.
"""

from __future__ import annotations

import json
import math
from contextlib import closing
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from stable_worldmodel.envs.tunnel_rooms import (
    DEFAULT_AGENT_RENDER_RADIUS,
    DEFAULT_MARKER_APPEARANCE,
    TunnelRoomsEnv,
    TunnelSearchPolicy,
    VisibleRoutingPolicy,
    certify_ranked_route,
    configured_marker_appearance,
    scenario_marker_appearance,
    sha256_appearance_spec,
    strip_default_scenario_appearance,
)
from scripts.data.tunnel_rooms_paired_io import (
    EVAL_MANIFEST_NAME,
    SCENARIO_MANIFEST_NAME,
    SCHEMA_VERSION,
    atomic_write_json,
    policy_settings,
)


# ---------------------------------------------------------------------------
# Expert policies and rollout auditing
# ---------------------------------------------------------------------------


def build_expert_policy(
    condition: str,
    seed: int | None = None,
    cfg: DictConfig | None = None,
) -> Any:
    """Build the cue-aware nearest-unexplored expert for one condition."""
    kwargs = policy_settings(cfg)
    if condition in {'hidden', 'paired_color'}:
        return TunnelSearchPolicy(seed=seed, **kwargs)
    raise ValueError(f'Unknown Tunnel Rooms condition: {condition!r}.')


def audit_active_rollout(
    env: TunnelRoomsEnv,
    obs: dict[str, np.ndarray],
    info: dict[str, Any],
    scenario: dict[str, Any],
    condition: str,
    max_steps: int,
    policy: Any,
) -> dict[str, Any]:
    """Finish an already-reset replay and return privileged audit metadata."""
    terminated = False
    truncated = False
    for step in range(max_steps):
        policy_info = {**info, **obs, 'step_idx': np.int32(step)}
        action = policy.get_action(policy_info)
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    metadata = env.audit_metadata()
    metadata.update(
        {
            'episode': scenario['episode'],
            'condition': condition,
            'scenario_seed': scenario['scenario_seed'],
            'success': bool(terminated),
            'timeout': not bool(terminated or truncated),
        }
    )
    return metadata


def audit_scenario_rollout(
    scenario: dict[str, Any],
    condition: str,
    max_steps: int,
    policy_seed: int | None = None,
    env: TunnelRoomsEnv | None = None,
    policy: Any | None = None,
) -> dict[str, Any]:
    """Replay one explicit scenario without writing privileged HDF5 columns."""
    replay = {**scenario, 'identity_mode': condition}
    owns_env = env is None
    env = env or TunnelRoomsEnv(
        identity_mode=condition,
        num_leaves=int(scenario.get('num_leaves', 2)),
    )
    try:
        obs, info = env.reset(options={'scenario': replay})
        policy = policy or build_expert_policy(condition, seed=policy_seed)
        if getattr(policy, 'env', None) is not env:
            policy.set_env(env)
        return audit_active_rollout(
            env,
            obs,
            info,
            scenario,
            condition,
            max_steps,
            policy,
        )
    finally:
        if owns_env:
            env.close()


# ---------------------------------------------------------------------------
# Natural scenario sampling and rank certification
# ---------------------------------------------------------------------------


def _configured_marker_entry(cfg: DictConfig) -> dict[str, Any]:
    appearance = configured_marker_appearance(cfg)
    return {} if appearance is None else {'marker_appearance': appearance}


def _scenario_stream_spec(cfg: DictConfig) -> dict[str, Any]:
    """Return the settings that uniquely define a training scenario stream."""
    spec = {
        'environment': 'swm/TunnelRooms-v1',
        'num_leaves': int(cfg.num_leaves),
        'scenario_seed': int(cfg.scenario_seed),
        'agent_speed': float(cfg.agent_speed),
        **_configured_marker_entry(cfg),
        **(
            {'agent_render_radius': float(cfg.agent_render_radius)}
            if cfg.get('agent_render_radius') is not None
            else {}
        ),
        'scenario_constraints': dict(cfg.get('scenario_constraints') or {}),
    }
    sampling = cfg.get('scenario_sampling')
    if sampling is None:
        spec['sampling'] = 'natural_target_independent'
    else:
        spec['sampling'] = _fixed_layout_spec(sampling, int(cfg.num_leaves))
    return spec


def natural_reset_options(cfg: DictConfig) -> dict[str, Any]:
    """Build reset options without rank schedules or episode-index coupling."""
    options: dict[str, Any] = {
        'variation_values': {
            'agent.speed': np.asarray([float(cfg.agent_speed)], np.float32)
        },
    }
    options.update(_configured_marker_entry(cfg))
    if cfg.get('agent_render_radius') is not None:
        options['variation_values']['agent.render_radius'] = np.asarray(
            [float(cfg.agent_render_radius)], np.float32
        )
    constraints = dict(cfg.get('scenario_constraints') or {})
    if constraints:
        options['scenario_constraints'] = {
            key: float(value)
            for key, value in constraints.items()
            if value is not None
        }
    return options


def _fixed_layout_spec(
    sampling: DictConfig, num_leaves: int
) -> dict[str, Any]:
    """Validate and normalize the explicit fixed-layout sampling contract."""
    resolved = OmegaConf.to_container(sampling, resolve=True)
    if (
        not isinstance(resolved, dict)
        or resolved.get('type') != 'fixed_layout'
    ):
        raise ValueError(
            'scenario_sampling.type must be exactly "fixed_layout".'
        )
    allowed = {'type', 'endpoint_positions', 'start_strata'}
    extra = sorted(set(resolved) - allowed)
    if extra:
        raise ValueError(f'Unsupported fixed-layout settings: {extra}.')

    endpoints = np.asarray(resolved.get('endpoint_positions'), np.float32)
    if endpoints.shape != (2 * num_leaves, 2):
        raise ValueError(
            'scenario_sampling.endpoint_positions must contain exactly two '
            'points per tunnel.'
        )
    strata = resolved.get('start_strata')
    if not isinstance(strata, list) or not strata:
        raise ValueError(
            'scenario_sampling.start_strata must be a non-empty list.'
        )

    normalized_strata = []
    names = set()
    for stratum in strata:
        if not isinstance(stratum, dict):
            raise ValueError('Each start stratum must be a mapping.')
        if set(stratum) != {'name', 'x_range', 'y_ranges'}:
            raise ValueError(
                'Each start stratum requires exactly name, x_range, and '
                'y_ranges.'
            )
        name = str(stratum['name'])
        if not name or name in names:
            raise ValueError(
                'Start stratum names must be non-empty and unique.'
            )
        names.add(name)
        x_range = _coordinate_range(stratum['x_range'], f'{name}.x_range')
        raw_y_ranges = stratum['y_ranges']
        if not isinstance(raw_y_ranges, list) or not raw_y_ranges:
            raise ValueError(f'{name}.y_ranges must be a non-empty list.')
        normalized_strata.append(
            {
                'name': name,
                'x_range': x_range,
                'y_ranges': [
                    _coordinate_range(values, f'{name}.y_ranges')
                    for values in raw_y_ranges
                ],
            }
        )
    return {
        'type': 'fixed_layout',
        'endpoint_positions': endpoints.tolist(),
        'start_strata': normalized_strata,
        'validation_profile': 'fixed_layout',
        'balanced_start_schedule': True,
        'routing_per_episode': 'uniform_permutation',
        'cue_assignment_per_episode': 'independent_uniform_permutation',
    }


def _coordinate_range(values: Any, label: str) -> list[float]:
    """Normalize one finite half-open coordinate sampling interval."""
    array = np.asarray(values, np.float32)
    if array.shape != (2,) or not np.isfinite(array).all():
        raise ValueError(f'{label} must contain two finite coordinates.')
    low, high = map(float, array)
    if low >= high:
        raise ValueError(f'{label} must have low < high.')
    return [low, high]


def _episode_rng(
    base_seed: int, episode: int, stream: int
) -> np.random.Generator:
    """Create an independent deterministic RNG stream for one episode."""
    sequence = np.random.SeedSequence([base_seed, episode, stream])
    return np.random.default_rng(sequence)


def _fixed_reset_options(
    cfg: DictConfig,
    episode: int,
    *,
    scenario_seed: int | None = None,
) -> tuple[dict[str, Any], str]:
    """Sample one balanced start, routing, and cue assignment on fixed geometry."""
    spec = _fixed_layout_spec(cfg.scenario_sampling, int(cfg.num_leaves))
    base_seed = (
        int(cfg.scenario_seed) if scenario_seed is None else int(scenario_seed)
    )
    strata = spec['start_strata']
    block, offset = divmod(episode, len(strata))
    order_rng = _episode_rng(base_seed, block, 0)
    stratum = strata[int(order_rng.permutation(len(strata))[offset])]

    position_rng = _episode_rng(base_seed, episode, 1)
    y_ranges = stratum['y_ranges']
    y_range = y_ranges[int(position_rng.integers(0, len(y_ranges)))]
    start = np.asarray(
        [
            position_rng.uniform(*stratum['x_range']),
            position_rng.uniform(*y_range),
        ],
        np.float32,
    )

    permutations = math.factorial(int(cfg.num_leaves))
    routing_rng = _episode_rng(base_seed, episode, 2)
    cue_rng = _episode_rng(base_seed, episode, 3)
    options = natural_reset_options(cfg)
    options['validation_profile'] = 'fixed_layout'
    options['variation_values'].update(
        {
            'agent.position': start,
            'tunnel.endpoints': np.asarray(
                spec['endpoint_positions'], np.float32
            ),
            'tunnel.pairing': int(routing_rng.integers(0, permutations)),
            'tunnel.cue_assignment': int(cue_rng.integers(0, permutations)),
        }
    )
    return options, str(stratum['name'])


def realized_rank(scenario: dict[str, Any]) -> int:
    """Return the target's one-based position in the realized search order."""
    order = [int(item) for item in scenario.get('search_order', [])]
    correct = scenario.get('correct_hub_endpoint')
    if order and correct is not None and int(correct) in order:
        return order.index(int(correct)) + 1
    raise ValueError('Scenario does not expose a realized target search rank.')


def certify_scenario_routing(scenario: dict[str, Any]) -> dict[str, Any]:
    """Derive online search metadata from geometry instead of stored labels."""
    result = deepcopy(scenario)
    route = certify_ranked_route(result)
    result.update(
        {
            'search_order': list(route.search_order),
            'correct_hub_endpoint': route.correct_hub_endpoint,
            'target_rank': route.target_rank,
        }
    )
    return result


def _normalize_scenario(
    scenario: dict[str, Any], episode: int, scenario_seed: int
) -> dict[str, Any]:
    """Add canonical identity, routing, and hash fields to one scenario."""
    result = certify_scenario_routing(scenario)
    result.pop('identity_mode', None)
    result = strip_default_scenario_appearance(result)
    rank = realized_rank(result)
    result.update(
        {
            'episode': int(episode),
            'scenario_seed': int(scenario_seed),
            'target_rank': rank,
        }
    )
    hash_input = {
        key: value for key, value in result.items() if key != 'episode'
    }
    result['scenario_hash'] = sha256_appearance_spec(hash_input)
    return result


def _generate_scenarios(
    cfg: DictConfig, start: int, stop: int
) -> list[dict[str, Any]]:
    """Generate a deterministic natural or fixed-layout scenario slice."""
    scenarios: list[dict[str, Any]] = []
    fixed_layout = cfg.get('scenario_sampling') is not None
    natural_options = None if fixed_layout else natural_reset_options(cfg)
    with closing(
        TunnelRoomsEnv(identity_mode='hidden', num_leaves=int(cfg.num_leaves))
    ) as env:
        for episode in range(start, stop):
            seed = int(cfg.scenario_seed) + episode
            if fixed_layout:
                options, start_stratum = _fixed_reset_options(cfg, episode)
            else:
                options, start_stratum = natural_options, None
            env.reset(seed=seed, options=options)
            scenario = env.scenario_spec()
            if start_stratum is not None:
                scenario['start_stratum'] = start_stratum
            scenarios.append(_normalize_scenario(scenario, episode, seed))
    return scenarios


def _validate_frozen_scenario(
    cfg: DictConfig,
    scenario: dict[str, Any],
    episode: int,
    path: Path,
) -> None:
    """Reject modified or non-canonical scenarios before collection resumes."""
    expected_seed = int(cfg.scenario_seed) + episode
    if int(scenario.get('episode', -1)) != episode:
        raise RuntimeError(f'Non-canonical episode order in {path}.')
    if int(scenario.get('scenario_seed', -1)) != expected_seed:
        raise RuntimeError(f'Non-canonical scenario seed in {path}.')

    hash_input = {
        key: value
        for key, value in scenario.items()
        if key not in {'episode', 'scenario_hash'}
    }
    if scenario.get('scenario_hash') != sha256_appearance_spec(hash_input):
        raise RuntimeError(f'Scenario integrity hash mismatch in {path}.')

    certified = certify_scenario_routing(scenario)
    route_fields = ('search_order', 'correct_hub_endpoint', 'target_rank')
    if any(scenario.get(key) != certified.get(key) for key in route_fields):
        raise RuntimeError(f'Scenario route certification mismatch in {path}.')

    sampling = cfg.get('scenario_sampling')
    if sampling is not None:
        spec = _fixed_layout_spec(sampling, int(cfg.num_leaves))
        if scenario.get('endpoint_positions') != spec['endpoint_positions']:
            raise RuntimeError(f'Fixed endpoint geometry mismatch in {path}.')
        strata = {row['name']: row for row in spec['start_strata']}
        stratum = strata.get(scenario.get('start_stratum'))
        if stratum is None:
            raise RuntimeError(f'Fixed start stratum mismatch in {path}.')
        x, y = map(float, scenario['agent_position'])
        x_low, x_high = stratum['x_range']
        inside_y = any(
            y_low <= y < y_high for y_low, y_high in stratum['y_ranges']
        )
        if not (x_low <= x < x_high and inside_y):
            raise RuntimeError(
                f'Agent start lies outside its stratum in {path}.'
            )
        if sorted(scenario.get('cue_assignment', [])) != list(
            range(int(cfg.num_leaves))
        ):
            raise RuntimeError(f'Invalid cue assignment in {path}.')


def freeze_scenario_manifest(
    cfg: DictConfig, output_dir: Path
) -> dict[str, Any]:
    """Create or extend the deterministic scenario stream before rollouts."""
    path = output_dir / SCENARIO_MANIFEST_NAME
    stream = _scenario_stream_spec(cfg)
    stream_hash = sha256_appearance_spec(stream)
    target = int(cfg.num_scenarios)
    if target <= 0:
        raise ValueError('num_scenarios must be positive.')

    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest.get('stream_hash') != stream_hash:
            raise RuntimeError(
                f'{path} belongs to a different deterministic scenario stream.'
            )
        scenarios = list(manifest.get('scenarios', []))
        if len(scenarios) > target:
            raise RuntimeError(
                f'{path} already freezes {len(scenarios)} scenarios, more than '
                f'the requested {target}.'
            )
    else:
        scenarios = []

    if len(scenarios) < target:
        scenarios.extend(_generate_scenarios(cfg, len(scenarios), target))
    for episode, scenario in enumerate(scenarios):
        _validate_frozen_scenario(cfg, scenario, episode, path)

    manifest = {
        'schema_version': SCHEMA_VERSION,
        'stream': stream,
        'stream_hash': stream_hash,
        'num_scenarios': len(scenarios),
        'scenarios': scenarios,
    }
    atomic_write_json(path, manifest)
    return manifest


# ---------------------------------------------------------------------------
# Clean-oracle certification and stratified evaluation manifests
# ---------------------------------------------------------------------------


def _certify_eval_candidate(
    scenario: dict[str, Any],
    max_steps: int,
    seed: int,
) -> dict[str, Any] | None:
    """Return oracle metadata when hidden and visible routes match the rank."""
    hidden = audit_scenario_rollout(
        scenario,
        'hidden',
        max_steps,
        policy=TunnelSearchPolicy(seed=seed),
    )
    color = audit_scenario_rollout(
        scenario,
        'paired_color',
        max_steps,
        policy=VisibleRoutingPolicy(seed=seed),
    )
    rank = int(scenario['target_rank'])
    expected_hidden = list(scenario['search_order'][:rank])
    expected_color = [int(scenario['correct_hub_endpoint'])]
    if not bool(hidden['success']) or not bool(color['success']):
        return None
    if hidden.get('hub_entry_trace') != expected_hidden:
        return None
    if color.get('hub_entry_trace') != expected_color:
        return None
    return {
        'hidden': {
            'steps': int(hidden['steps']),
            'route_length': float(hidden['route_length']),
            'hub_entry_trace': expected_hidden,
        },
        'paired_color': {
            'steps': int(color['steps']),
            'route_length': float(color['route_length']),
            'hub_entry_trace': expected_color,
        },
    }


def _assign_media_slots(
    rows: list[dict[str, Any]],
    num_leaves: int,
    videos_per_rank: int,
) -> None:
    """Mark representative short, median, and long hidden-policy routes."""
    for rank in range(1, num_leaves + 1):
        ranked = [row for row in rows if int(row['target_rank']) == rank]
        ranked.sort(
            key=lambda row: float(row['oracle']['hidden']['route_length'])
        )
        if videos_per_rank == 0:
            continue
        quantiles = np.linspace(0.25, 0.75, videos_per_rank)
        chosen: list[int] = []
        for quantile in quantiles:
            candidate = int(round(float(quantile) * (len(ranked) - 1)))
            while candidate in chosen and candidate + 1 < len(ranked):
                candidate += 1
            while candidate in chosen and candidate > 0:
                candidate -= 1
            if candidate in chosen:
                raise RuntimeError('Unable to choose unique eval media cases.')
            chosen.append(candidate)
        for slot, index in enumerate(chosen):
            ranked[index]['media_slot'] = slot


def build_stratified_eval_manifest(
    cfg: DictConfig, output_dir: Path
) -> dict[str, Any] | None:
    """Freeze a deterministic eval set with equal support for every rank."""
    eval_cfg = cfg.get('eval_scenarios')
    if eval_cfg is None or not bool(eval_cfg.get('enable', False)):
        return None

    path = output_dir / EVAL_MANIFEST_NAME
    num_leaves = int(cfg.num_leaves)
    episodes_per_rank = int(eval_cfg.episodes_per_rank)
    videos_per_rank = int(eval_cfg.wandb_videos_per_rank)
    if episodes_per_rank <= 0:
        raise ValueError('eval_scenarios.episodes_per_rank must be positive.')
    if not 0 <= videos_per_rank <= episodes_per_rank:
        raise ValueError(
            'eval_scenarios.wandb_videos_per_rank must be between zero and '
            'episodes_per_rank.'
        )

    sampling = cfg.get('scenario_sampling')
    eval_spec = {
        'environment': 'swm/TunnelRooms-v1',
        'schema_version': SCHEMA_VERSION,
        'num_leaves': num_leaves,
        'scenario_seed': int(eval_cfg.scenario_seed),
        'episodes_per_rank': episodes_per_rank,
        'wandb_videos_per_rank': videos_per_rank,
        'agent_speed': float(cfg.agent_speed),
        **_configured_marker_entry(cfg),
        **(
            {'agent_render_radius': float(cfg.agent_render_radius)}
            if cfg.get('agent_render_radius') is not None
            else {}
        ),
        'scenario_constraints': dict(cfg.get('scenario_constraints') or {}),
        'eval_budget': int(eval_cfg.eval_budget),
        'sampling': (
            'natural_then_stratified_by_target_rank'
            if sampling is None
            else _fixed_layout_spec(sampling, num_leaves)
        ),
    }
    if cfg.get('source_stream_hash') is not None:
        eval_spec['source_stream_hash'] = str(cfg.source_stream_hash)
    eval_hash = sha256_appearance_spec(eval_spec)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get('eval_hash') != eval_hash:
            raise RuntimeError(f'{path} belongs to a different eval stream.')
        if existing.get('generation') != eval_spec:
            raise RuntimeError(
                f'{path} generation settings do not match its eval stream.'
            )
        expected_appearance = eval_spec.get(
            'marker_appearance', DEFAULT_MARKER_APPEARANCE
        )
        rows = existing.get('episodes')
        expected_rows = num_leaves * episodes_per_rank
        if not isinstance(rows, list) or len(rows) != expected_rows:
            raise RuntimeError(
                f'{path} must contain exactly {expected_rows} eval episodes.'
            )
        for index, scenario in enumerate(rows):
            actual_appearance = scenario_marker_appearance(scenario)
            if actual_appearance != expected_appearance:
                raise RuntimeError(
                    f'{path} episode {index} marker appearance does not match '
                    'the eval stream.'
                )
            expected_radius = float(
                eval_spec.get(
                    'agent_render_radius', DEFAULT_AGENT_RENDER_RADIUS
                )
            )
            actual_radius = float(
                scenario.get(
                    'agent_render_radius', DEFAULT_AGENT_RENDER_RADIUS
                )
            )
            if actual_radius != expected_radius:
                raise RuntimeError(
                    f'{path} episode {index} agent render radius does not '
                    'match the eval stream.'
                )
        return existing

    selected_by_rank = {rank: [] for rank in range(1, num_leaves + 1)}
    candidate = 0
    max_candidates = int(eval_cfg.get('max_candidates', 100000))
    options = None if sampling is not None else natural_reset_options(cfg)
    with closing(
        TunnelRoomsEnv(identity_mode='hidden', num_leaves=num_leaves)
    ) as env:
        while candidate < max_candidates and any(
            len(rows) < episodes_per_rank for rows in selected_by_rank.values()
        ):
            seed = int(eval_cfg.scenario_seed) + candidate
            candidate += 1
            if sampling is None:
                reset_options = options
                start_stratum = None
            else:
                reset_options, start_stratum = _fixed_reset_options(
                    cfg,
                    candidate - 1,
                    scenario_seed=int(eval_cfg.scenario_seed),
                )
            env.reset(seed=seed, options=reset_options)
            raw_scenario = env.scenario_spec()
            if start_stratum is not None:
                raw_scenario['start_stratum'] = start_stratum
            scenario = _normalize_scenario(raw_scenario, candidate - 1, seed)
            rank = int(scenario['target_rank'])
            if len(selected_by_rank[rank]) >= episodes_per_rank:
                continue
            oracle = _certify_eval_candidate(
                scenario,
                int(eval_cfg.eval_budget),
                seed,
            )
            if oracle is None:
                continue
            scenario.update(
                {
                    'case_id': f'tr{num_leaves}-seed-{seed}',
                    'oracle': oracle,
                    'media_slot': None,
                }
            )
            selected_by_rank[rank].append(scenario)

    incomplete = {
        rank: len(rows)
        for rank, rows in selected_by_rank.items()
        if len(rows) != episodes_per_rank
    }
    if incomplete:
        raise RuntimeError(
            'Unable to build the stratified eval manifest; '
            f'incomplete rank counts: {incomplete}.'
        )
    selected = [
        row
        for rank in range(1, num_leaves + 1)
        for row in selected_by_rank[rank]
    ]
    _assign_media_slots(selected, num_leaves, videos_per_rank)
    manifest = {
        'schema_version': SCHEMA_VERSION,
        'environment': 'swm/TunnelRooms-v1',
        'num_leaves': num_leaves,
        'episodes_per_rank': episodes_per_rank,
        'wandb_videos_per_rank': videos_per_rank,
        'num_scenarios': len(selected),
        'generation': eval_spec,
        'eval_hash': eval_hash,
        'episodes': selected,
    }
    atomic_write_json(path, manifest)
    return manifest


__all__ = [
    'build_stratified_eval_manifest',
    'certify_scenario_routing',
    'freeze_scenario_manifest',
    'natural_reset_options',
    'realized_rank',
]
