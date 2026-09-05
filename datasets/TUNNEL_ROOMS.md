# Collect and probe Tunnel Rooms

Run these commands from the `stable-worldmodel` directory after installing the
package as described in the repository README. Choose a writable output root:

```bash
export STABLEWM_HOME=/absolute/path/to/tunnel_rooms_outputs
python -m scripts.data.collect_tunnel_rooms
```

The default configuration freezes 10,000 two-leaf scenarios, then collects
one successful Color trajectory for each scenario. It uses scenario
and policy seed 41021, action noise 2.0, action-repeat probability 0.05, speed
5, and a 500-step episode limit. Shape and Palette are derived from these
Color trajectories. Collection uses outline 15-by-15 markers; the next step
derives the filled-marker training images.

The collector writes `datasets/tunnel_room_color.h5` below `STABLEWM_HOME`. Its
`runs/tunnel_rooms_collection/scenario_manifest.json` fixes geometry, pairings,
initial states, and goals. Retry seeds are deterministic and condition-specific;
every scenario must succeed within the configured attempt limit. The journal
commits each episode only after its HDF5 append succeeds. Resume validates
committed data and discards uncommitted tails. Existing outputs are not silently
replaced.

The CLI also generates `runs/tunnel_rooms_collection/eval_scenarios.json`, with
25 oracle-validated episodes per rank and a separate scenario seed 852031.
Rank is the target tunnel's position in the hidden policy's search order.
These generated cases use the collection marker appearance and are separate
from the paper's archived evaluation cases. The planning entry point can use
the bundled paper manifest for the corresponding archived protocol.
For a small collection smoke run, override `num_scenarios=6` and
`eval_scenarios.enable=false`; the default probe sidecar sizes require a larger
corpus.

## Derive visual conditions and probe sidecars

```bash
python -m scripts.data.rerender_tunnel_rooms --config-name tunnel_rooms_filled18x18
python -m scripts.data.rerender_tunnel_rooms --config-name tunnel_rooms_filled18x18 finalize_only=true
python -m scripts.data.rerender_tunnel_rooms --config-name tunnel_rooms_shape
python -m scripts.data.rerender_tunnel_rooms --config-name tunnel_rooms_shape finalize_only=true
python -m scripts.data.rerender_tunnel_rooms --config-name tunnel_rooms_palette
python -m scripts.data.rerender_tunnel_rooms --config-name tunnel_rooms_palette finalize_only=true
```

Color uses filled 18-by-18 green/orange markers. Shape replaces the paired
identities with neutral triangle/diamond markers. Palette uses filled green
RGB (40,153,78) and blue RGB (83,98,216) markers. All render the agent with radius
9. Shape and Palette replay the same Color trajectories and preserve action,
position, episode boundaries, pairing, and goal columns. Source HDF5 files stay
unchanged. SHA-256 hashes, immutable scenario manifests, render journals, and
validation results accompany the outputs; `finalize_only=true` verifies and
publishes aggregate provenance.

| Condition | HDF5 below `datasets/` | Sidecar below `runs/` |
| --- | --- | --- |
| Color | `tunnel_room_color_filled18x18.h5` | `tunnel_rooms_filled18x18/two_leaf/paired_color/location_probe.h5` |
| Shape | `tunnel_room_shape.h5` | `tunnel_rooms_shape/two_leaf/paired_shape/location_probe.h5` |
| Palette | `tunnel_room_palette.h5` | `tunnel_rooms_palette/two_leaf/paired_color/location_probe.h5` |

Each sidecar selects 512 fit, 128 validation, and 256 test episodes, using seed
180018. It records exact frame rows, episode IDs, splits, target names, and the
source hash. The composite targets include agent position, four positional
endpoints, and four cue-identified endpoints. Positional top/bottom labels sort
hub endpoints and leaf endpoints separately by vertical coordinate. Cue labels
identify the hub and leaf endpoints connected by the same cue; they do not
mean top/bottom. Shape uses triangle/diamond and Palette uses green/blue names.

## Coordinate probes

```bash
python -m scripts.probes.location_probe \
  world_model.checkpoint=/absolute/path/to/weights_epoch_10.pt \
  location_probe.sidecar="$STABLEWM_HOME/runs/tunnel_rooms_filled18x18/two_leaf/paired_color/location_probe.h5" \
  output.directory="$STABLEWM_HOME/probes/color"
```

Use the corresponding sidecar for Shape or Palette. The source HDF5 is resolved
from the sidecar; an explicit `dataset.source_hdf5` must resolve to that same
file. For CPU execution add `device=cpu location_probe.autocast=false`.
Logging to W&B is disabled by default.

The affine ridge probe standardizes both features and targets using the union
of fit and validation episodes; ridge alpha is fixed at 10. Test episodes are
held out. Reports separate current projected states and action-conditioned
predicted states and retain MAE and R-squared for each endpoint. MAE averages
absolute x/y coordinate errors in pixels; it is not Euclidean distance.
Endpoint R-squared averages the two coordinate-wise values, excluding
zero-variance coordinates. Aggregate metrics are unweighted averages of the
reported points. A geometry-only or cue-only summary must explicitly select
its four endpoint names rather than use an average that also includes the agent.

## Objective exposure

```bash
python -m scripts.data.audit_objective_exposure \
  "$STABLEWM_HOME/datasets/tunnel_room_color_filled18x18.h5" \
  --environment tunnel --frameskip 5 --num-steps 4 \
  --output "$STABLEWM_HOME/objective_exposure.json"
```

This counts the training windows under the stated sampling contract and
separates windows containing a tunnel transition from approach-only windows.
It uses explicit event columns when present and a declared position-jump
threshold otherwise. It does not train or evaluate a world model.
