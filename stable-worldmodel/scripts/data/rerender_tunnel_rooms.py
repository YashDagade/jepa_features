"""Derive immutable filled-marker or shape-cue Tunnel Rooms datasets.

The two strict YAML profiles select only visual and provenance conventions.
Scenario replay, resumable HDF5 publication, sidecars, and verification share
one workflow.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import hydra
from omegaconf import DictConfig, OmegaConf

from scripts.data.filled_marker_artifacts import (
    DEFAULT_RENDERER_METHOD,
    RERENDER_SCHEMA_VERSION,
    RerenderPaths,
    publish_json_idempotently,
    rerender_dataset,
    sha256_file,
    upgrade_scenario_manifest,
    validate_episode_structure,
)
from scripts.data.filled_marker_sidecars import (
    SidecarSettings,
    build_paired_location_sidecars,
    paired_shape_endpoint_order,
)
from stable_worldmodel.envs.tunnel_rooms.rendering import (
    FILLED_AGENT_RENDER_RADIUS,
    PAIRED_SHAPE_ORDER,
    filled_appearance,
    render_filled_markers,
    render_shape_markers,
    shape_appearance,
    shape_marker_appearance,
)
from scripts.plan.ranked_episodic_eval import load_ranked_manifest
from stable_worldmodel.envs.tunnel_rooms import sha256_appearance_spec
from stable_worldmodel.envs.tunnel_rooms.marker_raster import (
    shape_raster_contract,
)

FILLED_PROFILE = 'filled18x18'
SHAPE_PROFILE = 'paired_shape'
SHAPE_RENDERER_METHOD = (
    'clear_source_marker_draw_neutral_paired_shapes_and_exact_agent'
)
PROFILE_RUNTIME_ROOT = {
    FILLED_PROFILE: 'tunnel_rooms_filled18x18',
    SHAPE_PROFILE: 'tunnel_rooms_shape',
}
PROFILE_RENDERER = {
    FILLED_PROFILE: (render_filled_markers, DEFAULT_RENDERER_METHOD),
    SHAPE_PROFILE: (render_shape_markers, SHAPE_RENDERER_METHOD),
}
PROFILE_SIDECAR_KEY = {
    FILLED_PROFILE: 'location_sidecars',
    SHAPE_PROFILE: 'paired_shape_sidecars',
}
SHAPE_SIDECAR_NAMES = {
    'outline_sidecar': 'source_sidecar',
    'filled_sidecar': 'shape_sidecar',
    'outline_sidecar_sha256': 'source_sidecar_sha256',
    'filled_sidecar_sha256': 'shape_sidecar_sha256',
    'outline_source_sha256': 'source_hdf5_sha256',
    'filled_source_sha256': 'shape_hdf5_sha256',
}
PROFILE_SIDECAR_HASHES = {
    FILLED_PROFILE: (
        ('outline_sidecar', 'outline_sidecar_sha256'),
        ('filled_sidecar', 'filled_sidecar_sha256'),
    ),
    SHAPE_PROFILE: (
        ('source_sidecar', 'source_sidecar_sha256'),
        ('shape_sidecar', 'shape_sidecar_sha256'),
    ),
}
COMMON_ROOT_KEYS = {
    'profile',
    'output_root',
    'groups',
    'conditions',
    'chunk_rows',
    'finalize_only',
    'agent_render_radius',
    'location_probe',
    'leaf_groups',
}
GROUP_KEYS = {'num_leaves', 'source_manifest', 'datasets'}
DATASET_KEYS = {
    'source',
    'output_name',
    'source_sidecar_name',
    'destination_sidecar_name',
    'include_positional_targets',
}
LOCATION_PROBE_KEYS = {
    'fit_episodes',
    'validation_episodes',
    'test_episodes',
    'history_size',
    'frameskip',
    'num_preds',
    'seed',
}


def run_from_config(cfg: DictConfig) -> dict[str, Any]:
    """Rerender selected groups and conditions without replacing outputs."""
    profile = _validate_config(cfg)
    render_frames, renderer_method = PROFILE_RENDERER[profile]
    appearance = _profile_appearance(profile, cfg)
    settings = OmegaConf.to_container(cfg.location_probe, resolve=True)
    if not isinstance(settings, dict):
        raise TypeError('location_probe config must be a mapping.')
    sidecar_settings = SidecarSettings(**settings)
    output_root = Path(cfg.output_root).resolve()
    dataset_root = output_root / 'datasets'
    runtime_name = (
        'tunnel_rooms_palette'
        if 'palette' in cfg
        else PROFILE_RUNTIME_ROOT[profile]
    )
    runtime_root = output_root / 'runs' / runtime_name
    conditions = [str(value) for value in cfg.conditions]
    results: dict[str, Any] = {}

    for group_name in cfg.groups:
        group = cfg.leaf_groups[group_name]
        group_root = runtime_root / str(group_name)
        scenario_manifest = group_root / 'scenario_manifest.json'
        first_source = Path(next(iter(group.datasets.values())).source)
        with h5py.File(first_source, 'r', swmr=True) as source_file:
            lengths, _offsets = validate_episode_structure(source_file)
        scenarios, source_manifest_hash = _upgrade_scenarios(
            profile, group, scenario_manifest, len(lengths), cfg.get('palette')
        )
        render_manifest_hash = sha256_file(scenario_manifest)
        records: dict[str, dict[str, Any]] = {}

        for condition in conditions:
            dataset = group.datasets[condition]
            condition_root = group_root / condition
            source = Path(dataset.source)
            paths = RerenderPaths(
                source=source,
                destination=dataset_root / str(dataset.output_name),
                journal=condition_root / 'rerender_journal.jsonl',
                provenance=condition_root / 'rerender_provenance.json',
            )
            provenance = rerender_dataset(
                paths,
                scenarios,
                source_manifest_hash,
                render_manifest_hash,
                condition,
                int(group.num_leaves),
                chunk_rows=int(cfg.chunk_rows),
                render_frames=render_frames,
                renderer_appearance=appearance,
                renderer_method=renderer_method,
            )
            semantic_order = (
                {'semantic_order': paired_shape_endpoint_order}
                if profile == SHAPE_PROFILE
                else {}
            )
            sidecars = build_paired_location_sidecars(
                source,
                paths.destination,
                condition_root / str(dataset.source_sidecar_name),
                condition_root / str(dataset.destination_sidecar_name),
                scenarios,
                condition,
                sidecar_settings,
                outline_sha256=provenance['source_sha256'],
                filled_sha256=provenance['destination_sha256'],
                include_positional_targets=bool(
                    dataset.include_positional_targets
                ),
                **semantic_order,
            )
            sidecar_key = PROFILE_SIDECAR_KEY[profile]
            names = SHAPE_SIDECAR_NAMES if profile == SHAPE_PROFILE else {}
            sidecars = {
                names.get(key, key): value for key, value in sidecars.items()
            }
            record = {'provenance': provenance, sidecar_key: sidecars}
            if profile == SHAPE_PROFILE:
                record['scenario_manifest'] = str(scenario_manifest.resolve())
            publish_json_idempotently(
                condition_root / 'provenance_manifest.json', record
            )
            records[condition] = record

        results[str(group_name)] = (
            records[SHAPE_PROFILE]
            if profile == SHAPE_PROFILE
            else {
                'scenario_manifest': str(scenario_manifest),
                'datasets': records,
            }
        )
    return results


def finalize_provenance(cfg: DictConfig) -> Path:
    """Verify every configured artifact before publishing a root manifest."""
    profile = _validate_config(cfg)
    output_root = Path(cfg.output_root).resolve()
    runtime_name = (
        'tunnel_rooms_palette'
        if 'palette' in cfg
        else PROFILE_RUNTIME_ROOT[profile]
    )
    runtime_root = output_root / 'runs' / runtime_name
    groups: dict[str, Any] = {}

    for group_name, group in cfg.leaf_groups.items():
        group_root = runtime_root / str(group_name)
        scenario_manifest = group_root / 'scenario_manifest.json'
        if not scenario_manifest.is_file():
            raise FileNotFoundError(
                f'{profile} scenario manifest is missing: {scenario_manifest}'
            )
        scenario_document = json.loads(scenario_manifest.read_text())
        source_manifest = Path(scenario_document['source_manifest'])
        _require_recorded_hash(
            source_manifest, scenario_document['source_manifest_sha256']
        )
        scenario_hash = sha256_file(scenario_manifest)
        records: dict[str, dict[str, Any]] = {}

        for condition in group.datasets:
            manifest = group_root / str(condition) / 'provenance_manifest.json'
            if not manifest.is_file():
                raise FileNotFoundError(
                    f'{profile} provenance is missing: {manifest}'
                )
            record = json.loads(manifest.read_text())
            _verify_condition_record(
                profile, str(group_name), record, scenario_hash
            )
            records[str(condition)] = {
                'provenance_manifest': str(manifest.resolve()),
                'provenance_manifest_sha256': sha256_file(manifest),
                **record,
            }

        common = {
            'scenario_manifest': str(scenario_manifest.resolve()),
            'scenario_manifest_sha256': scenario_hash,
            'source_scenario_manifest': str(source_manifest.resolve()),
            'source_scenario_manifest_sha256': scenario_document[
                'source_manifest_sha256'
            ],
        }
        if profile == FILLED_PROFILE:
            groups[str(group_name)] = {**common, 'datasets': records}
        else:
            groups[str(group_name)] = {
                **common,
                'ranked_eval_manifest': _derive_ranked_eval_manifest(
                    Path(group.ranked_eval.source),
                    Path(group.ranked_eval.output),
                ),
                **records[SHAPE_PROFILE],
            }

    aggregate = {
        'schema_version': RERENDER_SCHEMA_VERSION,
        'output_root': str(output_root),
        'renderer': {
            **_profile_appearance(profile, cfg),
            'method': PROFILE_RENDERER[profile][1],
        },
        'groups': groups,
    }
    destination = runtime_root / 'provenance_manifest.json'
    publish_json_idempotently(destination, aggregate)
    return destination


# Private helpers


def _require_recorded_hash(path: Path, expected: str) -> None:
    """Verify a provenance input using the legacy finalizer contract."""
    if not path.is_file():
        raise FileNotFoundError(f'Provenance artifact is missing: {path}')
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f'Provenance hash mismatch for {path}: {actual} != {expected}.'
        )


def _validate_config(cfg: DictConfig) -> str:
    """Reject loose profile names and malformed profile-specific sections."""
    profile = str(cfg.profile)
    if profile not in PROFILE_RUNTIME_ROOT:
        raise ValueError(f'Unknown rerender profile: {profile!r}.')
    root_keys = COMMON_ROOT_KEYS
    if profile == SHAPE_PROFILE:
        root_keys = root_keys | {'shape_order', 'renderer_source_revision'}
    if 'palette' in cfg:
        if profile != FILLED_PROFILE or list(cfg.conditions) != [
            'paired_color'
        ]:
            raise ValueError(
                'Palette overrides require filled paired_color rendering.'
            )
        root_keys = root_keys | {'palette'}
        _require_exact_keys(cfg.palette, {'colors', 'names'}, 'palette')
        for group in cfg.leaf_groups.values():
            count = int(group.num_leaves)
            colors, names = cfg.palette.colors, cfg.palette.names
            if len(colors) != count or len(names) != count:
                raise ValueError(
                    'Palette must contain one color and name per leaf.'
                )
            if len(set(names)) != count or any(
                not str(name) for name in names
            ):
                raise ValueError('Palette names must be nonempty and unique.')
            if any(
                len(rgb) != 3
                or any(type(v) is not int or not 0 <= v <= 255 for v in rgb)
                for rgb in colors
            ):
                raise ValueError(
                    'Palette colors must be RGB integer triples in [0, 255].'
                )
    _require_exact_keys(cfg, root_keys, f'{profile} root')
    _require_exact_keys(
        cfg.location_probe, LOCATION_PROBE_KEYS, 'location_probe'
    )
    if float(cfg.agent_render_radius) != FILLED_AGENT_RENDER_RADIUS:
        raise ValueError(
            f'{profile} requires agent_render_radius={FILLED_AGENT_RENDER_RADIUS}.'
        )
    unknown_groups = sorted(set(cfg.groups) - set(cfg.leaf_groups))
    if unknown_groups:
        raise ValueError(f'Unknown rerender groups: {unknown_groups}.')
    if profile == SHAPE_PROFILE:
        if tuple(cfg.shape_order) != PAIRED_SHAPE_ORDER:
            raise ValueError('shape_order must be triangle, diamond, star.')
        if tuple(cfg.conditions) != (SHAPE_PROFILE,):
            raise ValueError('paired_shape is the only shape condition.')
    for name, group in cfg.leaf_groups.items():
        group_keys = GROUP_KEYS
        if profile == SHAPE_PROFILE:
            group_keys = group_keys | {'ranked_eval'}
        _require_exact_keys(group, group_keys, str(name))
        if profile == SHAPE_PROFILE and set(group.datasets) != {SHAPE_PROFILE}:
            raise ValueError(f'{name} must define one paired_shape dataset.')
        if profile == SHAPE_PROFILE:
            _require_exact_keys(
                group.ranked_eval, {'source', 'output'}, f'{name}.ranked_eval'
            )
        for condition, dataset in group.datasets.items():
            _require_exact_keys(dataset, DATASET_KEYS, f'{name}.{condition}')
    for name in cfg.groups:
        unknown = sorted(
            set(cfg.conditions) - set(cfg.leaf_groups[name].datasets)
        )
        if unknown:
            raise ValueError(f'Unknown conditions for {name}: {unknown}.')
    return profile


def _require_exact_keys(
    cfg: DictConfig, expected: set[str], context: str
) -> None:
    actual = set(cfg)
    if actual != expected:
        raise ValueError(
            f'{context} keys must be {sorted(expected)}, got {sorted(actual)}.'
        )


def _profile_appearance(profile: str, cfg: DictConfig) -> dict[str, object]:
    if profile == FILLED_PROFILE:
        appearance = filled_appearance()
        if 'palette' in cfg:
            appearance['palette'] = OmegaConf.to_container(
                cfg.palette, resolve=True
            )
        return appearance
    return shape_appearance(str(cfg.renderer_source_revision))


def _upgrade_scenarios(
    profile: str,
    group: DictConfig,
    destination: Path,
    episode_count: int,
    palette: DictConfig | None = None,
) -> tuple[list[dict[str, Any]], str]:
    options: dict[str, Any] = {}
    if profile == SHAPE_PROFILE:
        options = {
            'appearance': shape_marker_appearance(),
            'agent_render_radius': FILLED_AGENT_RENDER_RADIUS,
            'sampling': 'exact_scenario_replay_from_filled_paired_color',
            'scenario_updates': {
                'identity_mode': SHAPE_PROFILE,
                'shape_raster': shape_raster_contract(),
            },
            'retain_legacy_marker_fields': False,
        }
    if palette is not None:
        options['scenario_updates'] = {
            'tunnel_colors': OmegaConf.to_container(palette.colors),
            'tunnel_color_names': OmegaConf.to_container(palette.names),
        }
        options['sampling'] = 'exact_scenario_replay_with_palette_override'
    return upgrade_scenario_manifest(
        Path(group.source_manifest),
        destination,
        int(group.num_leaves),
        episode_count,
        **options,
    )


def _verify_condition_record(
    profile: str,
    group_name: str,
    record: dict[str, Any],
    scenario_hash: str,
) -> None:
    rerender = record['provenance']
    sidecar_key = PROFILE_SIDECAR_KEY[profile]
    for path_key, hash_key in (
        ('source_hdf5', 'source_sha256'),
        ('destination_hdf5', 'destination_sha256'),
    ):
        _require_recorded_hash(Path(rerender[path_key]), rerender[hash_key])
    for path_key, hash_key in PROFILE_SIDECAR_HASHES[profile]:
        sidecars = record[sidecar_key]
        _require_recorded_hash(Path(sidecars[path_key]), sidecars[hash_key])
    if rerender['render_scenario_manifest_sha256'] != scenario_hash:
        raise RuntimeError(
            f'{group_name} references a different render manifest.'
        )


def _derive_ranked_eval_manifest(
    source: Path, destination: Path
) -> dict[str, str]:
    """Replay filled-marker eval cases with the shape-cue appearance."""
    if source.resolve() == destination.resolve():
        raise ValueError('Ranked eval source and destination must differ.')
    source_identity = (source.stat().st_ino, source.stat().st_mtime_ns)
    source_hash = sha256_file(source)
    document = deepcopy(json.loads(source.read_text()))
    marker_contract = shape_marker_appearance()
    document['generation']['marker_appearance'] = marker_contract
    document['generation']['shape_raster'] = shape_raster_contract()
    document['eval_hash'] = sha256_appearance_spec(document['generation'])
    for scenario in document['episodes']:
        scenario.update(marker_contract)
        hash_input = dict(scenario)
        for key in (
            'case_id',
            'episode',
            'media_slot',
            'oracle',
            'scenario_hash',
        ):
            hash_input.pop(key, None)
        scenario['scenario_hash'] = sha256_appearance_spec(hash_input)
    if source_identity != (source.stat().st_ino, source.stat().st_mtime_ns):
        raise RuntimeError('Ranked eval source changed during derivation.')
    _require_recorded_hash(source, source_hash)
    publish_json_idempotently(destination, document)
    load_ranked_manifest(destination)
    _require_recorded_hash(source, source_hash)
    return {
        'source': str(source.resolve()),
        'source_sha256': source_hash,
        'destination': str(destination.resolve()),
        'destination_sha256': sha256_file(destination),
    }


@hydra.main(
    version_base=None,
    config_path='config',
    config_name='tunnel_rooms_filled18x18',
)
def main(cfg: DictConfig) -> None:
    result: Any = (
        {'provenance_manifest': str(finalize_provenance(cfg))}
        if bool(cfg.finalize_only)
        else run_from_config(cfg)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
