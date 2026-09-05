"""Evaluate a LeWM checkpoint on certified near/far tunnel scenarios."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

import stable_worldmodel as swm
import helpers
import ranked_episodic_eval as ranked


def write_json(path, value):
    """Write finite results; reject invalid metrics instead of serializing NaNs."""

    def encode(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    path.write_text(
        json.dumps(value, indent=2, default=encode, allow_nan=False) + '\n'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--solver', choices=('cem', 'gd'), default='cem')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument(
        '--identity',
        choices=('paired_color', 'paired_shape'),
        default='paired_color',
    )
    parser.add_argument(
        '--output', type=Path, default=Path('outputs/planning')
    )
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    cfg = OmegaConf.load(Path(__file__).with_name(f'{args.solver}.yaml'))
    cfg.eval.scenario_manifest = str(args.manifest.resolve())
    cfg.world.identity_mode = args.identity
    cfg.world.marker_style = (
        'shape_cues' if args.identity == 'paired_shape' else 'filled18x18'
    )
    cfg.solver.device = args.device
    manifest = ranked.load_ranked_manifest_from_cfg(cfg)
    ranked.validate_manifest_identity(manifest, args.identity)
    if args.validate_only:
        world = helpers.make_episodic_world(cfg, 1)
        try:
            for case in manifest.cases:
                scenario = ranked.scenario_for_identity(
                    case, args.identity, cfg.eval.agent_speed
                )
                world.reset(
                    seed=case.scenario_seed, options={'scenario': scenario}
                )
            print(
                f'Validated {manifest.num_eval} certified cases and their renderings.'
            )
        finally:
            world.close()
        return
    if args.checkpoint is None or args.dataset is None:
        parser.error(
            '--checkpoint and --dataset are required for model evaluation'
        )
    if not args.checkpoint.is_file() or not args.dataset.is_file():
        raise FileNotFoundError(
            'The checkpoint and normalization dataset must exist.'
        )
    cfg.policy = str(args.checkpoint.resolve())
    cfg.eval.dataset_name = str(args.dataset.resolve())
    cfg.dataset.stats = cfg.eval.dataset_name
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = helpers.load_model(cfg)
    process = helpers.build_processors(cfg)
    transform = {key: helpers.img_transform(cfg) for key in ('pixels', 'goal')}

    def make_policy():
        return swm.policy.WorldModelPolicy(
            solver=hydra.utils.instantiate(cfg.solver, model=model),
            config=swm.PlanConfig(**cfg.plan_config),
            process=process,
            transform=transform,
        )

    write_json(
        args.output / 'config.json', OmegaConf.to_container(cfg, resolve=True)
    )
    write_json(
        args.output / 'normalization.json',
        {
            key: {
                'mean': value.mean_,
                'scale': value.scale_,
                'n_samples': value.n_samples_seen_,
            }
            for key, value in process.items()
        },
    )
    with torch.autocast(
        device_type=torch.device(args.device).type,
        dtype=torch.bfloat16,
        enabled=bool(cfg.bf16) and args.device.startswith('cuda'),
    ):
        metrics = ranked.evaluate_ranked_manifest(
            manifest,
            batch_size=cfg.eval.batch_size,
            eval_budget=cfg.eval.eval_budget,
            identity_mode=args.identity,
            agent_speed=cfg.eval.agent_speed,
            video_root=args.output / 'videos',
            world_factory=lambda count: helpers.make_episodic_world(
                cfg, count
            ),
            policy_factory=make_policy,
            frame_from_info=helpers.frame_from_info,
            expert_rollout=lambda seed, scenario: helpers.expert_rollout(
                cfg, seed, cfg.eval.eval_budget, scenario=scenario
            ),
            save_episode_video=helpers.save_episode_video,
        )
    with args.checkpoint.open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
        metrics['checkpoint_sha256'] = digest.hexdigest()
    write_json(args.output / 'metrics.json', metrics)
    print(
        json.dumps(
            {
                key: value['success_rate']
                for key, value in metrics['by_rank'].items()
            }
        )
    )


if __name__ == '__main__':
    main()
