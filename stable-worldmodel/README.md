# Stable WorldModel Research Package

This directory contains the Python package used by `jepa_features` for **Tunnel Room: Probing What JEPA World Models Preserve**.
It is based on [Stable WorldModel](https://github.com/galilai-group/stable-worldmodel) and retains its data, environment, action-conditioned prediction, and planning interfaces, extending them with the Tunnel Room environments, global decoders, coordinate probes, and rollout diagnostics.

## Install

```bash
pip install -e '.[train,format]'
```

Verify the training CLI:

```bash
python scripts/train/lewm_idm.py --help
```

## Model Contract

- Native global baselines (LeWM, PLDM, LpWM) select `last_hidden_state[:, 0]` (the ViT CLS token).
- The primary Tunnel Room recipe (`scripts/train/lewm_idm.py` with `feature_mode=global_spatial_8x8`) applies learned spatial $8 \times 8$ patch aggregation and an inverse-dynamics (IDM) auxiliary objective.

Large datasets, checkpoints, and run outputs are excluded from Git. Override dataset paths via environment variables (e.g., `JEPA_FEATURES_TR2_DATASET`) or Hydra CLI overrides (`data.dataset.name=/path/to/dataset.h5`).

The upstream package is MIT licensed; see [`LICENSE`](LICENSE).
