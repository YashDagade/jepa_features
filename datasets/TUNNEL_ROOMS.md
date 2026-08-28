# Tunnel Rooms Color datasets

Tunnel Rooms is a continuing 224-by-224 visual environment with one hub and
two or three leaf rooms. Each hub endpoint is paired with one leaf endpoint.
In the color condition the shared color identifies that pairing, while room
geometry and endpoint positions vary across episodes.

The model receives only image observations and actions. Stored state and
geometry columns are privileged provenance used by collection audits and
fixed evaluation labels; they are excluded by every training preset.

## Two objectives

The corpus separates two dynamical signals:

1. **Approach:** movement reduces distance to the selected hub endpoint.
2. **Teleport:** crossing the endpoint produces the paired discontinuous
   transition.

`scripts/data/audit_objective_exposure.py` reports both adjacent-transition
rates and the exposure induced by the exact `frameskip=5`, four-frame training
window. Transitions that satisfy neither definition remain `other`; they are
not silently assigned to the dominant objective.

## Reproducibility

Large HDF5 files remain external. The tracked manifests under
`datasets/manifests/` bind each experiment to its absolute source path,
SHA-256, row/episode counts, render provenance, and fixed episode-disjoint
probe sidecar. The schema-2 TR2 sidecar contains both semantic color-bound
targets (green/orange hub and leaf endpoints) and position-only targets
(top/bottom endpoints), allowing the same feature snapshots to distinguish
cue binding from ordinary geometry.

Collection and rerender tools live under `stable-worldmodel/scripts/data/`;
the environment and expert live under
`stable_worldmodel/envs/tunnel_rooms/`. Generated datasets, journals, and
videos must stay outside Git.
