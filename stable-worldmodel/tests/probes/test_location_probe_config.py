from pathlib import Path

from omegaconf import OmegaConf


CONFIG_PATH = (
    Path(__file__).parents[2]
    / 'scripts'
    / 'probes'
    / 'config'
    / 'location_probe.yaml'
)
CHASER_CONFIG_PATH = CONFIG_PATH.with_name('chaser_probe.yaml')


def test_standalone_location_probe_uses_the_shared_component_namespace():
    cfg = OmegaConf.load(CONFIG_PATH)

    assert 'probe' not in cfg
    assert 'sidecar' not in cfg.dataset
    assert cfg.location_probe.image_size == 224
    assert OmegaConf.is_missing(cfg.location_probe, 'sidecar')


def test_chaser_probe_declares_the_shared_wandb_run_fields():
    cfg = OmegaConf.load(CHASER_CONFIG_PATH)

    assert 'group' in cfg.wandb.config
    assert cfg.wandb.config.resume == 'allow'
