"""Strict ranked Tunnel Rooms manifest loading and episodic evaluation."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stable_worldmodel.envs.tunnel_rooms import (
    DEFAULT_MARKER_APPEARANCE,
    certify_ranked_route,
    scenario_marker_appearance,
    validate_marker_appearance,
)
from stable_worldmodel.envs.tunnel_rooms.marker_raster import (
    SHAPE_ORDER,
    shape_raster_contract,
)


SCHEMA_VERSION = 2
TUNNEL_ROOMS_ENV = 'swm/TunnelRooms-v1'
DEFAULT_EPISODES_PER_RANK = {1: 50, 2: 25, 3: 20}
DEFAULT_WANDB_VIDEOS_PER_RANK = 3
IDENTITY_MODES = ('hidden', 'paired_color', 'paired_shape')
PAIRED_SHAPE_APPEARANCE = {
    'marker_style': 'shape_cues',
    'marker_width': 20,
    'marker_height': 20,
}
PAIRED_SHAPE_AGENT_RENDER_RADIUS = 9.0
PAIRED_SHAPE_RASTER = shape_raster_contract()

_CASE_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')
_EVALUATOR_FIELDS = {'case_id', 'media_slot'}
_REQUIRED_CASE_FIELDS = {
    'case_id',
    'num_leaves',
    'scenario_seed',
    'pairing',
    'endpoint_positions',
    'agent_position',
    'landing_offset',
    'target_position',
    'target_leaf',
    'search_order',
    'target_rank',
    'correct_hub_endpoint',
}


@dataclass(frozen=True)
class RankedCase:
    """One certified scenario in a ranked evaluation manifest."""

    case_id: str
    target_rank: int
    target_shape: str
    search_order: tuple[int, ...]
    correct_hub_endpoint: int
    scenario_seed: int
    media_slot: int | None
    scenario: dict[str, Any]


@dataclass(frozen=True)
class RankedManifest:
    """Validated schema-v2 ranked Tunnel Rooms evaluation manifest."""

    path: Path | None
    sha256: str
    num_leaves: int
    episodes_per_rank: int
    wandb_videos_per_rank: int
    cases: tuple[RankedCase, ...]

    @property
    def num_eval(self) -> int:
        return self.num_leaves * self.episodes_per_rank

    def cases_by_rank(self) -> dict[int, tuple[RankedCase, ...]]:
        return {
            rank: tuple(
                case for case in self.cases if case.target_rank == rank
            )
            for rank in range(1, self.num_leaves + 1)
        }

    def interleaved_cases(self) -> tuple[RankedCase, ...]:
        """Round-robin ranks while preserving manifest order within each rank."""
        by_rank = self.cases_by_rank()
        return tuple(
            by_rank[rank][offset]
            for offset in range(self.episodes_per_rank)
            for rank in range(1, self.num_leaves + 1)
        )

    def media_cases(self, rank: int) -> tuple[RankedCase, ...]:
        selected = [
            case
            for case in self.cases_by_rank()[rank]
            if case.media_slot is not None
        ]
        return tuple(sorted(selected, key=lambda case: int(case.media_slot)))


def _strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{field} must be an integer, got {value!r}.')
    return int(value)


def _reject_legacy_labels(mapping: Mapping[str, Any], location: str) -> None:
    legacy = [
        str(key)
        for key in mapping
        if str(key) == 'labels'
        or str(key) in {'eval_label', 'memory_label', 'scenario_label'}
        or str(key).endswith('_success_label')
    ]
    if legacy:
        raise ValueError(
            f'{location} contains retired label fields: {sorted(legacy)}. '
            'Schema-v2 manifests use one-based target_rank only.'
        )


def _validate_point(value: Any, field: str) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f'{field} must be a two-element list.')
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in value
    ):
        raise ValueError(f'{field} must contain two finite numbers.')


def _agent_render_radius(mapping: Mapping[str, Any], location: str) -> float:
    value = mapping.get('agent_render_radius', 7.0)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f'{location}.agent_render_radius must be positive.')
    return float(value)


def _validate_case(
    row: Any,
    *,
    row_index: int,
    num_leaves: int,
    wandb_videos_per_rank: int,
    marker_appearance: Mapping[str, Any],
    agent_render_radius: float,
) -> RankedCase:
    location = f'episodes[{row_index}]'
    if not isinstance(row, dict):
        raise ValueError(f'{location} must be an object.')
    _reject_legacy_labels(row, location)
    actual_appearance = scenario_marker_appearance(row)
    if actual_appearance != marker_appearance:
        raise ValueError(
            f'{location} marker appearance {actual_appearance} does not match '
            f'the manifest generation contract {dict(marker_appearance)}.'
        )
    actual_radius = _agent_render_radius(row, location)
    if actual_radius != agent_render_radius:
        raise ValueError(
            f'{location} agent_render_radius={actual_radius} does not match '
            f'the manifest generation contract {agent_render_radius}.'
        )

    missing = sorted(_REQUIRED_CASE_FIELDS - row.keys())
    if missing:
        raise ValueError(f'{location} is missing required fields: {missing}.')

    case_id = row['case_id']
    if not isinstance(case_id, str) or not _CASE_ID_RE.fullmatch(case_id):
        raise ValueError(
            f'{location}.case_id must be a path-safe identifier, got {case_id!r}.'
        )

    row_num_leaves = _strict_int(row['num_leaves'], f'{location}.num_leaves')
    if row_num_leaves != num_leaves:
        raise ValueError(
            f'{location}.num_leaves={row_num_leaves} does not match manifest '
            f'num_leaves={num_leaves}.'
        )

    scenario_seed = _strict_int(
        row['scenario_seed'], f'{location}.scenario_seed'
    )
    target_rank = _strict_int(row['target_rank'], f'{location}.target_rank')
    if target_rank not in range(1, num_leaves + 1):
        raise ValueError(
            f'{location}.target_rank must be one-based in [1, {num_leaves}], '
            f'got {target_rank}.'
        )

    search_order_raw = row['search_order']
    if not isinstance(search_order_raw, list):
        raise ValueError(f'{location}.search_order must be a list.')
    search_order = tuple(
        _strict_int(value, f'{location}.search_order[{index}]')
        for index, value in enumerate(search_order_raw)
    )
    expected_endpoints = tuple(range(num_leaves))
    if tuple(sorted(search_order)) != expected_endpoints:
        raise ValueError(
            f'{location}.search_order must be a permutation of '
            f'{list(expected_endpoints)}, got {list(search_order)}.'
        )

    pairing = _strict_int(row['pairing'], f'{location}.pairing')
    permutations = tuple(itertools.permutations(range(num_leaves)))
    if pairing not in range(len(permutations)):
        raise ValueError(
            f'{location}.pairing must be in [0, {len(permutations) - 1}], '
            f'got {pairing}.'
        )
    target_leaf = _strict_int(row['target_leaf'], f'{location}.target_leaf')
    if target_leaf not in range(num_leaves):
        raise ValueError(
            f'{location}.target_leaf must be in [0, {num_leaves - 1}], '
            f'got {target_leaf}.'
        )
    certified_correct = int(permutations[pairing].index(target_leaf))
    correct_hub_endpoint = _strict_int(
        row['correct_hub_endpoint'], f'{location}.correct_hub_endpoint'
    )
    if correct_hub_endpoint != certified_correct:
        raise ValueError(
            f'{location}.correct_hub_endpoint={correct_hub_endpoint} does not '
            f'match pairing/target_leaf endpoint {certified_correct}.'
        )
    route = certify_ranked_route(row)
    if search_order != route.search_order:
        raise ValueError(
            f'{location}.search_order={list(search_order)} does not match the '
            f'geometry-derived order {list(route.search_order)}.'
        )
    if correct_hub_endpoint != route.correct_hub_endpoint:
        raise ValueError(
            f'{location}.correct_hub_endpoint={correct_hub_endpoint} does not '
            f'match geometry/pairing certificate {route.correct_hub_endpoint}.'
        )
    if target_rank != route.target_rank:
        raise ValueError(
            f'{location}.target_rank={target_rank} does not match the certified '
            f'search_order rank {route.target_rank}.'
        )

    endpoints = row['endpoint_positions']
    if not isinstance(endpoints, list) or len(endpoints) != 2 * num_leaves:
        raise ValueError(
            f'{location}.endpoint_positions must contain {2 * num_leaves} points.'
        )
    for endpoint_index, endpoint in enumerate(endpoints):
        _validate_point(
            endpoint, f'{location}.endpoint_positions[{endpoint_index}]'
        )
    _validate_point(row['agent_position'], f'{location}.agent_position')
    _validate_point(row['target_position'], f'{location}.target_position')

    media_slot_raw = row.get('media_slot')
    media_slot = None
    if media_slot_raw is not None:
        media_slot = _strict_int(media_slot_raw, f'{location}.media_slot')
        if media_slot not in range(wandb_videos_per_rank):
            raise ValueError(
                f'{location}.media_slot must be in '
                f'[0, {wandb_videos_per_rank - 1}] or null.'
            )

    scenario = copy.deepcopy(
        {
            key: value
            for key, value in row.items()
            if key not in _EVALUATOR_FIELDS
        }
    )
    return RankedCase(
        case_id=case_id,
        target_rank=target_rank,
        target_shape=SHAPE_ORDER[correct_hub_endpoint],
        search_order=search_order,
        correct_hub_endpoint=correct_hub_endpoint,
        scenario_seed=scenario_seed,
        media_slot=media_slot,
        scenario=scenario,
    )


def parse_ranked_manifest(
    manifest: Any,
    *,
    path: str | Path | None = None,
    sha256: str | None = None,
) -> RankedManifest:
    """Validate and parse one schema-v2 ranked Tunnel Rooms manifest."""
    if not isinstance(manifest, dict):
        raise ValueError('Ranked eval manifest must be a JSON object.')
    _reject_legacy_labels(manifest, 'manifest')

    schema_version = _strict_int(
        manifest.get('schema_version'), 'schema_version'
    )
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f'Unsupported ranked eval schema_version={schema_version}; '
            f'expected {SCHEMA_VERSION}.'
        )
    if manifest.get('environment') != TUNNEL_ROOMS_ENV:
        raise ValueError(
            'Ranked eval manifests are only valid for '
            f'{TUNNEL_ROOMS_ENV}, got {manifest.get("environment")!r}.'
        )

    num_leaves = _strict_int(manifest.get('num_leaves'), 'num_leaves')
    if num_leaves not in DEFAULT_EPISODES_PER_RANK:
        raise ValueError(
            'Ranked Tunnel Rooms eval supports one, two, or three leaves, '
            f'got {num_leaves}.'
        )
    episodes_per_rank = _strict_int(
        manifest.get('episodes_per_rank'), 'episodes_per_rank'
    )
    expected_episodes_per_rank = DEFAULT_EPISODES_PER_RANK[num_leaves]
    if episodes_per_rank != expected_episodes_per_rank:
        raise ValueError(
            f'episodes_per_rank must be {expected_episodes_per_rank} for '
            f'{num_leaves}-leaf ranked eval, got {episodes_per_rank}.'
        )
    wandb_videos_per_rank = _strict_int(
        manifest.get('wandb_videos_per_rank'), 'wandb_videos_per_rank'
    )
    if wandb_videos_per_rank != DEFAULT_WANDB_VIDEOS_PER_RANK:
        raise ValueError(
            'wandb_videos_per_rank must be '
            f'{DEFAULT_WANDB_VIDEOS_PER_RANK}, got '
            f'{wandb_videos_per_rank}.'
        )
    expected_total = num_leaves * episodes_per_rank
    if 'num_scenarios' in manifest:
        num_scenarios = _strict_int(manifest['num_scenarios'], 'num_scenarios')
        if num_scenarios != expected_total:
            raise ValueError(
                f'num_scenarios must equal num_leaves * episodes_per_rank = '
                f'{expected_total}, got {num_scenarios}.'
            )

    generation = manifest.get('generation') or {}
    if not isinstance(generation, dict):
        raise ValueError('generation must be an object when present.')
    marker_appearance = validate_marker_appearance(
        generation.get('marker_appearance', DEFAULT_MARKER_APPEARANCE)
    )
    if marker_appearance == PAIRED_SHAPE_APPEARANCE:
        raster = generation.get('shape_raster')
        if raster != PAIRED_SHAPE_RASTER:
            raise ValueError(
                'paired_shape generation.shape_raster does not match the '
                'shared cue raster contract.'
            )
    agent_render_radius = _agent_render_radius(generation, 'generation')

    rows = manifest.get('episodes')
    if not isinstance(rows, list):
        raise ValueError('episodes must be a list.')
    if len(rows) != expected_total:
        raise ValueError(
            f'Ranked manifest contains {len(rows)} episodes; expected exactly '
            f'{expected_total} ({num_leaves} ranks x {episodes_per_rank}).'
        )
    cases = tuple(
        _validate_case(
            row,
            row_index=index,
            num_leaves=num_leaves,
            wandb_videos_per_rank=wandb_videos_per_rank,
            marker_appearance=marker_appearance,
            agent_render_radius=agent_render_radius,
        )
        for index, row in enumerate(rows)
    )
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        duplicates = sorted(
            case_id for case_id in set(case_ids) if case_ids.count(case_id) > 1
        )
        raise ValueError(
            f'Ranked manifest has duplicate case_id values: {duplicates}.'
        )

    for rank in range(1, num_leaves + 1):
        ranked_cases = [case for case in cases if case.target_rank == rank]
        if len(ranked_cases) != episodes_per_rank:
            raise ValueError(
                f'Rank {rank} contains {len(ranked_cases)} cases; expected '
                f'{episodes_per_rank}.'
            )
        slots = sorted(
            case.media_slot
            for case in ranked_cases
            if case.media_slot is not None
        )
        expected_slots = list(range(wandb_videos_per_rank))
        if slots != expected_slots:
            raise ValueError(
                f'Rank {rank} media_slot values must be exactly '
                f'{expected_slots}, got {slots}.'
            )

    if marker_appearance == PAIRED_SHAPE_APPEARANCE:
        observed_identities = {case.correct_hub_endpoint for case in cases}
        expected_identities = set(range(num_leaves))
        if observed_identities != expected_identities:
            raise ValueError(
                'paired_shape manifest must cover every target shape; '
                f'observed hub identities {sorted(observed_identities)}.'
            )

    if sha256 is None:
        canonical = json.dumps(
            manifest, sort_keys=True, separators=(',', ':'), allow_nan=False
        ).encode('utf-8')
        sha256 = hashlib.sha256(canonical).hexdigest()
    if not isinstance(sha256, str) or not re.fullmatch(
        r'[0-9a-f]{64}', sha256
    ):
        raise ValueError(f'Invalid manifest sha256: {sha256!r}.')
    return RankedManifest(
        path=None if path is None else Path(path),
        sha256=sha256,
        num_leaves=num_leaves,
        episodes_per_rank=episodes_per_rank,
        wandb_videos_per_rank=wandb_videos_per_rank,
        cases=cases,
    )


def load_ranked_manifest(path: str | Path) -> RankedManifest:
    """Read and validate a ranked manifest while hashing its exact bytes."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f'Eval scenario manifest not found: {manifest_path}'
        )
    payload = manifest_path.read_bytes()
    try:
        manifest = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(
            f'Invalid JSON in eval scenario manifest {manifest_path}: {error}'
        ) from error
    return parse_ranked_manifest(
        manifest,
        path=manifest_path,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def load_ranked_manifest_from_cfg(cfg: Any) -> RankedManifest | None:
    """Load a configured ranked manifest without slicing it by ``num_eval``."""
    manifest_path = cfg.eval.get('scenario_manifest')
    if manifest_path in (None, 'null', ''):
        return None
    if str(cfg.world.env_name) != TUNNEL_ROOMS_ENV:
        raise ValueError(
            'eval.scenario_manifest is a ranked Tunnel Rooms manifest and cannot '
            f'be used with world.env_name={cfg.world.env_name!r}.'
        )
    loaded = load_ranked_manifest(str(manifest_path))
    configured_leaves = int(cfg.world.get('num_leaves', 2))
    if configured_leaves != loaded.num_leaves:
        raise ValueError(
            f'Configured world.num_leaves={configured_leaves} does not match '
            f'manifest num_leaves={loaded.num_leaves}.'
        )
    configured_per_rank = cfg.eval.get('episodes_per_rank')
    if configured_per_rank in (None, 'null'):
        raise ValueError('Ranked eval requires eval.episodes_per_rank.')
    if int(configured_per_rank) != loaded.episodes_per_rank:
        raise ValueError(
            f'Configured eval.episodes_per_rank={configured_per_rank} does not '
            f'match manifest value {loaded.episodes_per_rank}.'
        )
    configured_media = cfg.eval.get('wandb_videos_per_rank')
    if configured_media in (None, 'null'):
        raise ValueError('Ranked eval requires eval.wandb_videos_per_rank.')
    if int(configured_media) != loaded.wandb_videos_per_rank:
        raise ValueError(
            f'Configured eval.wandb_videos_per_rank={configured_media} does '
            f'not match manifest value {loaded.wandb_videos_per_rank}.'
        )
    validate_manifest_identity(
        loaded, str(cfg.world.get('identity_mode', 'hidden'))
    )
    return loaded


def validate_identity_mode(identity_mode: str) -> str:
    """Return one accepted replay identity or fail before constructing worlds."""
    mode = str(identity_mode)
    if mode not in IDENTITY_MODES:
        raise ValueError(
            f'Unknown Tunnel Rooms identity mode: {mode!r}; '
            f'expected one of {list(IDENTITY_MODES)}.'
        )
    return mode


def validate_manifest_identity(
    manifest: RankedManifest,
    identity_mode: str,
) -> str:
    """Validate condition-specific rendering without changing manifest cases."""
    mode = validate_identity_mode(identity_mode)
    if mode != 'paired_shape':
        return mode
    for case in manifest.cases:
        appearance = scenario_marker_appearance(case.scenario)
        radius = _agent_render_radius(case.scenario, case.case_id)
        if appearance != PAIRED_SHAPE_APPEARANCE or not math.isclose(
            radius, PAIRED_SHAPE_AGENT_RENDER_RADIUS
        ):
            raise ValueError(
                'paired_shape ranked eval requires the shape-cue radius-9 '
                'manifest contract; '
                f'{case.case_id} has appearance={appearance}, '
                f'agent_render_radius={radius}.'
            )
    return mode


def scenario_for_identity(
    case_or_scenario: RankedCase | Mapping[str, Any],
    identity_mode: str,
    agent_speed: float | None = None,
) -> dict[str, Any]:
    """Replay fixed geometry/target with configured identity and dynamics."""
    mode = validate_identity_mode(identity_mode)
    source = (
        case_or_scenario.scenario
        if isinstance(case_or_scenario, RankedCase)
        else {
            key: value
            for key, value in case_or_scenario.items()
            if key not in _EVALUATOR_FIELDS
        }
    )
    scenario = copy.deepcopy(dict(source))
    if mode == 'paired_shape':
        appearance = scenario_marker_appearance(scenario)
        radius = _agent_render_radius(scenario, 'scenario')
        if appearance != PAIRED_SHAPE_APPEARANCE or not math.isclose(
            radius, PAIRED_SHAPE_AGENT_RENDER_RADIUS
        ):
            raise ValueError(
                'paired_shape replay requires marker_style="shape_cues", '
                'a 20x20 maximum cue extent, and agent_render_radius=9.'
            )
    scenario['identity_mode'] = mode
    if agent_speed is not None:
        speed = float(agent_speed)
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError(
                f'agent_speed must be finite and positive, got {speed}.'
            )
        scenario['speed'] = speed
    return scenario


def validate_complete_episode_rows(
    manifest: RankedManifest,
    episode_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Fail loudly if any manifest case is missing, duplicated, or misreported."""
    expected_ids = [case.case_id for case in manifest.interleaved_cases()]
    observed_ids = [str(row.get('case_id')) for row in episode_rows]
    if len(observed_ids) != manifest.num_eval:
        raise RuntimeError(
            f'Ranked eval produced {len(observed_ids)} rows for '
            f'{manifest.num_eval} manifest cases.'
        )
    if len(set(observed_ids)) != len(observed_ids):
        raise RuntimeError('Ranked eval produced duplicate case rows.')
    if set(observed_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(observed_ids))
        extra = sorted(set(observed_ids) - set(expected_ids))
        raise RuntimeError(
            f'Ranked eval case coverage mismatch; missing={missing}, extra={extra}.'
        )
    expected_by_id = {case.case_id: case for case in manifest.cases}
    for row in episode_rows:
        case_id = str(row['case_id'])
        expected_rank = expected_by_id[case_id].target_rank
        if int(row.get('target_rank', -1)) != expected_rank:
            raise RuntimeError(
                f'Case {case_id} reported target_rank={row.get("target_rank")}; '
                f'expected {expected_rank}.'
            )
        expected_shape = expected_by_id[case_id].target_shape
        if str(row.get('target_shape')) != expected_shape:
            raise RuntimeError(
                f'Case {case_id} reported target_shape='
                f'{row.get("target_shape")!r}; expected {expected_shape!r}.'
            )
        if bool(row.get('success')) and bool(row.get('timeout')):
            raise RuntimeError(
                f'Case {row.get("case_id")} cannot be both success and timeout.'
            )


def build_ranked_results(
    manifest: RankedManifest,
    episode_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build mean success across all cases plus per-rank result structures."""
    validate_complete_episode_rows(manifest, episode_rows)
    by_id = {str(row['case_id']): dict(row) for row in episode_rows}
    ordered_rows = [
        by_id[case.case_id] for case in manifest.interleaved_cases()
    ]
    successes = [bool(row['success']) for row in ordered_rows]
    num_success = int(sum(successes))

    by_rank: dict[str, dict[str, Any]] = {}
    for rank in range(1, manifest.num_leaves + 1):
        rank_rows = [
            row for row in ordered_rows if int(row['target_rank']) == rank
        ]
        rank_successes = [bool(row['success']) for row in rank_rows]
        media = []
        for case in manifest.media_cases(rank):
            row = by_id[case.case_id]
            media.append(
                {
                    'media_slot': int(case.media_slot),
                    'case_id': case.case_id,
                    'video_path': str(row['video_path']),
                }
            )
        by_rank[f'rank_{rank}'] = {
            'success_rate': 100.0 * sum(rank_successes) / len(rank_rows),
            'num_success': int(sum(rank_successes)),
            'num_eval': len(rank_rows),
            'case_ids': [str(row['case_id']) for row in rank_rows],
            'media': media,
        }

    identity_modes = {str(row['identity_mode']) for row in ordered_rows}
    if len(identity_modes) != 1:
        raise RuntimeError(
            f'Ranked eval mixed identity modes: {sorted(identity_modes)}.'
        )
    identity_mode = identity_modes.pop()
    by_shape: dict[str, dict[str, Any]] = {}
    if identity_mode == 'paired_shape':
        for shape in SHAPE_ORDER[: manifest.num_leaves]:
            shape_rows = [
                row for row in ordered_rows if row['target_shape'] == shape
            ]
            shape_successes = [bool(row['success']) for row in shape_rows]
            by_shape[shape] = {
                'success_rate': (
                    100.0 * sum(shape_successes) / len(shape_rows)
                ),
                'num_success': int(sum(shape_successes)),
                'num_eval': len(shape_rows),
            }

    results = {
        # This is the arithmetic mean over every evaluated scenario.
        'success_rate': 100.0 * num_success / manifest.num_eval,
        'num_success': num_success,
        'num_eval': manifest.num_eval,
        'episode_successes': successes,
        'seeds': [int(row['scenario_seed']) for row in ordered_rows],
        'completed_episodes': manifest.num_eval,
        'timed_out_episodes': int(
            sum(bool(row['timeout']) for row in ordered_rows)
        ),
        'manifest_sha256': manifest.sha256,
        'identity_mode': identity_mode,
        'episodes': ordered_rows,
        'by_rank': by_rank,
    }
    if by_shape:
        results['by_shape'] = by_shape
    return results


def evaluate_ranked_manifest(
    manifest: RankedManifest,
    *,
    batch_size: int,
    eval_budget: int,
    identity_mode: str,
    agent_speed: float,
    video_root: str | Path,
    world_factory: Callable[[int], Any],
    policy_factory: Callable[[], Any],
    frame_from_info: Callable[[Any], np.ndarray],
    expert_rollout: Callable[[int, dict[str, Any]], list[np.ndarray]],
    save_episode_video: Callable[
        [Path, list[np.ndarray], list[np.ndarray], np.ndarray], None
    ],
) -> dict[str, Any]:
    """Evaluate every ranked case exactly once in deterministic wait batches."""
    batch_size = int(batch_size)
    eval_budget = int(eval_budget)
    if batch_size <= 0:
        raise ValueError(f'batch_size must be positive, got {batch_size}.')
    if eval_budget <= 0:
        raise ValueError(f'eval_budget must be positive, got {eval_budget}.')
    identity_mode = validate_manifest_identity(manifest, identity_mode)

    root = Path(video_root)
    root.mkdir(parents=True, exist_ok=True)
    ordered_cases = manifest.interleaved_cases()
    episode_rows: list[dict[str, Any]] = []

    for start in range(0, manifest.num_eval, batch_size):
        batch = ordered_cases[start : start + batch_size]
        world = world_factory(len(batch))
        if int(world.num_envs) != len(batch):
            world.close()
            raise RuntimeError(
                f'Ranked eval batch has {len(batch)} cases but world has '
                f'{world.num_envs} envs.'
            )
        policy = policy_factory()
        world.set_policy(policy)
        scenarios = [
            scenario_for_identity(case, identity_mode, agent_speed)
            for case in batch
        ]
        seeds = [case.scenario_seed for case in batch]

        frames: dict[int, list[np.ndarray]] = {}
        goal_frames: dict[int, np.ndarray] = {}
        steps = [0 for _ in batch]
        outcomes: dict[int, dict[str, Any]] = {}
        hub_entry_traces: dict[int, list[int]] = {}
        try:
            world.reset(
                seed=seeds,
                options=[{'scenario': scenario} for scenario in scenarios],
            )
            for env_idx in range(len(batch)):
                frames[env_idx] = [
                    frame_from_info(world.infos['pixels'][env_idx])
                ]
                goal_frames[env_idx] = frame_from_info(
                    world.infos['goal'][env_idx]
                )

            def on_step(active_world: Any) -> None:
                for env_idx in range(len(batch)):
                    if env_idx in outcomes:
                        continue
                    frames[env_idx].append(
                        frame_from_info(active_world.infos['pixels'][env_idx])
                    )
                    goal_frames[env_idx] = frame_from_info(
                        active_world.infos['goal'][env_idx]
                    )
                    steps[env_idx] += 1

            def on_done(env_idx: int, _ep_idx: int, active_world: Any) -> None:
                env_idx = int(env_idx)
                if env_idx in outcomes:
                    raise RuntimeError(
                        f'Case {batch[env_idx].case_id} completed more than once.'
                    )
                success = bool(
                    np.asarray(active_world.terminateds[env_idx]).squeeze()
                )
                truncated = bool(
                    np.asarray(active_world.truncateds[env_idx]).squeeze()
                )
                outcomes[env_idx] = {
                    'success': success,
                    'timeout': bool(truncated and not success),
                    'status': (
                        'success'
                        if success
                        else 'timeout'
                        if truncated
                        else 'failure'
                    ),
                }

            world._run(
                max_steps=eval_budget,
                mode='wait',
                on_step=on_step,
                on_done=on_done,
            )
            wrapped_envs = getattr(getattr(world, 'envs', None), 'envs', ())
            for env_idx, wrapped_env in enumerate(wrapped_envs):
                raw_env = getattr(wrapped_env, 'unwrapped', wrapped_env)
                if not hasattr(raw_env, 'audit_metadata'):
                    continue
                metadata = raw_env.audit_metadata()
                hub_entry_traces[env_idx] = [
                    int(endpoint)
                    for endpoint in metadata.get('hub_entry_trace', [])
                ]
        finally:
            world.close()

        for env_idx, case in enumerate(batch):
            outcome = outcomes.get(
                env_idx,
                {'success': False, 'timeout': True, 'status': 'timeout'},
            )
            goal_frame = goal_frames[env_idx]
            expert_frames = expert_rollout(seeds[env_idx], scenarios[env_idx])
            video_path = (
                root / f'rank_{case.target_rank}' / f'{case.case_id}.mp4'
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)
            save_episode_video(
                video_path,
                frames[env_idx] or [goal_frame],
                expert_frames or [goal_frame],
                goal_frame,
            )
            if not video_path.is_file():
                raise RuntimeError(
                    f'Panel video was not created for case {case.case_id}: '
                    f'{video_path}'
                )
            episode_rows.append(
                {
                    'case_id': case.case_id,
                    'target_rank': case.target_rank,
                    'target_shape': case.target_shape,
                    'search_order': list(case.search_order),
                    'correct_hub_endpoint': case.correct_hub_endpoint,
                    'scenario_seed': case.scenario_seed,
                    'identity_mode': identity_mode,
                    'success': bool(outcome['success']),
                    'timeout': bool(outcome['timeout']),
                    'status': str(outcome['status']),
                    'steps': int(steps[env_idx]),
                    'hub_entry_trace': hub_entry_traces.get(env_idx, []),
                    'media_slot': case.media_slot,
                    'video_path': str(video_path),
                }
            )

    return build_ranked_results(manifest, episode_rows)
