from pathlib import Path

from omegaconf import OmegaConf


CONFIG_PATH = (
    Path(__file__).parents[2]
    / 'scripts'
    / 'probes'
    / 'config'
    / 'location_probe.yaml'
)


def test_standalone_location_probe_uses_the_shared_component_namespace():
    cfg = OmegaConf.load(CONFIG_PATH)

    assert 'probe' not in cfg
    assert 'sidecar' not in cfg.dataset
    assert cfg.location_probe.image_size == 224
    assert OmegaConf.is_missing(cfg.location_probe, 'sidecar')
