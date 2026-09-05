"""Model loading, normalization and media for episodic Tunnel Rooms planning."""

from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
from stable_worldmodel.envs.tunnel_rooms import VisibleRoutingPolicy
from stable_worldmodel.plot import save_panel_videos
from ranked_episodic_eval import scenario_for_identity, validate_identity_mode


def img_transform(cfg, dtype=torch.float32):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(dtype, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def make_episodic_world(cfg: DictConfig, n_envs: int):
    world_cfg = OmegaConf.to_container(
        cfg.world, resolve=True, throw_on_missing=False
    )
    world_cfg['num_envs'] = n_envs
    world_cfg['max_episode_steps'] = int(cfg.eval.eval_budget)
    if str(world_cfg.get('env_name')) == 'swm/TunnelRooms-v1':
        identity_mode = validate_identity_mode(
            str(world_cfg.get('identity_mode', 'hidden'))
        )
        marker_style = world_cfg.get('marker_style')
        if identity_mode == 'paired_shape':
            if marker_style not in (None, 'shape_cues'):
                raise ValueError(
                    'paired_shape eval requires marker_style="shape_cues".'
                )
            world_cfg['marker_style'] = 'shape_cues'

    # Noisy TunnelRooms was collected with speed 5.0. Fresh episodic evals must
    # use the same dynamics; otherwise we evaluate the checkpoint in a slower
    # environment than the one it saw in dataset-driven eval.
    speed = float(cfg.eval.get('agent_speed', 5.0))
    init_value = dict(world_cfg.get('init_value') or {})
    init_value['agent.speed'] = np.array([speed], dtype=np.float32)
    world_cfg['init_value'] = init_value
    return swm.World(**world_cfg, image_shape=(224, 224))


def frame_from_info(value) -> np.ndarray:
    """Return the latest rendered RGB frame from a batched/history info item."""
    frame = np.asarray(value)
    if frame.ndim > 3:
        frame = frame[-1]
    return frame.copy()


def load_model(cfg: DictConfig):
    policy_name = cfg.get('policy', 'random')
    if policy_name == 'random':
        return None
    model = swm.wm.utils.load_pretrained(policy_name)
    model = model.to(cfg.solver.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    if cfg.get('compile', False):
        encoder_attr = 'backbone' if hasattr(model, 'backbone') else 'encoder'
        setattr(
            model, encoder_attr, torch.compile(getattr(model, encoder_attr))
        )
        model.predictor = torch.compile(model.predictor)
    return model


def build_processors(cfg: DictConfig):
    """Fit the same non-image normalization used by dataset-offset eval."""
    dataset = swm.data.load_dataset(
        cfg.eval.dataset_name,
        cache_dir=cfg.get('cache_dir'),
        keys_to_cache=list(cfg.dataset.keys_to_cache),
    )
    processors = {}
    for col in cfg.dataset.keys_to_cache:
        if col == 'pixels' or col not in dataset.column_names:
            continue
        processor = preprocessing.StandardScaler()
        values = dataset.get_col_data(col)
        values = values[~np.isnan(values).any(axis=1)]
        processor.fit(values)
        processors[col] = processor
        if col != 'action':
            processors[f'goal_{col}'] = processor
    return processors


def scenario_identity(cfg: DictConfig, scenario: dict) -> dict:
    """Replay manifest geometry/target under configured identity and speed."""
    if str(cfg.world.env_name) == 'swm/TunnelRooms-v1':
        return scenario_for_identity(
            scenario,
            str(cfg.world.get('identity_mode', 'hidden')),
            float(cfg.eval.get('agent_speed', 5.0)),
        )
    return dict(scenario)


def expert_rollout(
    cfg: DictConfig,
    seed: int,
    eval_budget: int,
    scenario: dict | None = None,
) -> list[np.ndarray]:
    """Replay the same fresh scenario with the deterministic expert policy."""
    world = make_episodic_world(cfg, 1)
    frames: list[np.ndarray] = []
    try:
        world.set_policy(VisibleRoutingPolicy(seed=seed))
        if scenario is None:
            world.reset(seed=seed)
        else:
            world.reset(
                seed=seed,
                options={'scenario': scenario_identity(cfg, scenario)},
            )

        def on_step(active_world):
            frames.append(frame_from_info(active_world.infos['pixels'][0]))

        world._run(
            episodes=1, max_steps=eval_budget, mode='wait', on_step=on_step
        )
    finally:
        world.close()
    return frames


def save_episode_video(
    path: Path,
    agent_frames: list[np.ndarray],
    expert_frames: list[np.ndarray],
    goal_frame: np.ndarray,
) -> None:
    """Save one old-style panel video: agent rollout, expert rollout, goal."""
    tmp_dir = path.parent / f'.{path.stem}_panel'
    shutil.rmtree(tmp_dir, ignore_errors=True)
    save_panel_videos(
        tmp_dir,
        {
            'agent': [np.asarray(agent_frames)],
            'expert': [np.asarray(expert_frames)],
            'goal': [goal_frame],
        },
    )
    generated = tmp_dir / 'env_0.mp4'
    if generated.exists():
        generated.replace(path)
    shutil.rmtree(tmp_dir, ignore_errors=True)
