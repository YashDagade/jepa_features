# Paper data

This directory contains compact, tracked numeric sources for the manuscript.
Large datasets, checkpoints, decoder videos, and raw W&B exports remain outside
the paper tree.


The current main comparison is spatial 8×8 + IDM 3.125 on Color and Shape.
`spatial_primary_probes.csv`, `spatial_primary_planning.csv`, and
`spatial_primary_provenance.csv` align the primary metrics to exact
checkpoints, with separate file/state hashes. `spatial_primary_decoder_sources.csv`
identifies the complete six-clip montage. `spatial_primary_lpwm_planning.csv`
and `spatial_primary_lpwm_recipe.json` preserve the stronger same-campaign
LpWM positive control and its objective/activation differences.
`spatial_alternate_palette_support.csv` retains green/blue runs at IDM0/.625;
these are appendix support, not matched3.125 primary runs.

The following CLS and legacy files remain supporting evidence. `short_paper_noidm_color_*`
records independently reproduced, same-campaign no-IDM Color probes, planning,
and provenance. `short_paper_idm_color_cls_planning.csv` holds the actual
Color CLS+IDM counts (CEM 68/8%, GD 52/28%), which must not be confused with
spatial planning. These planning runs use historical checkpoints and must not
be attributed to the newly matched representation checkpoints.
`short_paper_cls_decoder_sources.csv` maps the matched native-CLS Color,
Shape, and green/blue decoder figure to epoch-10 validation GIFs, all with
IDM 3.125 and source `c4a2df269e16`.

`short_paper_matched_probe_results.csv` records all six completed conditions
(three cues by IDM 0 or 3.125), using one source, seed 3072, ten epochs,
batch size 128, and the same four-endpoint coordinate contract. All stored
affine-probe coordinate metrics were independently reproduced exactly from
their retained feature caches. `short_paper_matched_probe_provenance.csv`
locks result, feature-cache, sidecar, resolved-config, and checkpoint-file
hashes. The new standalone probes hash the serialized checkpoint file;
their explicit `checkpoint_file_sha256` fields must not be confused with the
historical training callback's canonical tensor-state hashes. These runs
have no planning evaluation, so historical planning remains separate evidence.

Files used by the LeWM-focused revision:

- `short_paper_probe_results.csv` is the compact, common-contract source for
  historical affine-probe comparisons (the active main table uses
  `spatial_primary_probes.csv`). Cue and position values are
  unweighted four-endpoint means; legacy `encoder` keys are correctly labeled
  as current post-projector states. Shape position $R^2$ was reconstructed
  post hoc from the retained immutable feature cache under the original ridge
  contract; the same reconstruction reproduces the archived MAE before adding
  $R^2$.
- `short_paper_cls_probe_provenance.csv` and
  `short_paper_cls_probe_targets.csv` record the completed September 4 Shape
  and green/blue CLS controls. Exact frozen-feature ridge reconstruction
  reproduces archived cue scores before supplying Shape slot coordinates.
  The 640 fitting examples combine the 512 fit and 128 validation examples
  because alpha is fixed; 256 test examples remain separate. The green/blue
  CLS control has negative cue and location R² and is not a selective-cue
  replication. Legacy `encoder` metric keys mean current projected states.
- `short_paper_cls_checkpoint_provenance.csv` distinguishes serialized-file
  SHA-256 from the canonical tensor-state hash used in the probe table.
- `short_paper_cls_planning.csv` freezes completed Shape CLS CEM/GD counts
  and the separate rank manifest; this is not a matched Color–Shape comparison.
  Green/blue CLS planning was disabled by configuration, not a failed job.
- `short_paper_probe_reconstructions.csv` freezes the Shape reconstruction's
  per-target MAE and coordinate-wise $R^2$, together with checkpoint, feature
  cache, composite sidecar, and source-dataset hashes.
- `short_paper_transition_diversity.csv` freezes the exact one-seed
  transition-window pilot. It records both the non-monotonic affine-probe
  results and the approximate epoch-compute multiplier, so the associated
  decoder strip is not misrepresented as a causal recovery result.
- `short_paper_decoder_sources.csv` maps every source image in the active
  decoder figures to a W&B run, source revision, tracked file, hash, and claim
  boundary.
- `short_paper_protocol.csv` freezes the active TR2 trajectory-collection,
  temporal-sampling, and rank-planning settings.  In particular, collector
  action repetition is separated from the five distinct primitive actions in
  each model action block.

- `lewm_alternate_palette_probe.csv` records the matched green/orange and
  green/blue global localization controls at auxiliary weights 0 and .625. The
  standard rows come from the historical LeWM color-identity factorial and the
  alternate rows from the audited blue/green control. The layouts, actions,
  targets, rewards, and episode structure match across 10,000 scenarios after
  excluding `tunnel_colors`, `tunnel_color_names`, and the palette-dependent
  `scenario_hash`.
- `lewm_palette_decoder_summary.csv` records the held-out green/orange decoder
  audit and the smaller logged-media summaries for the standard and blue/green
  palettes. The `analysis_protocol` column distinguishes the prespecified
  episode audit from the weaker media-support summaries.
- `metric_sensitivity.csv` contains the six-setting standard-palette decoder
  support-means used for the 4.0--7.5% descriptive range in an earlier draft. The
  held-out E024 audit in `lewm_palette_decoder_summary.csv` is the stronger
  inferential result.
- `historical_lewm_global_probe_rows.csv` records the nine seed-by-feature
  LeWM rows behind the native-global, learned-global-MLP, and
  position-aware-patch-feature localization comparison. Identity-conditioned
  hub MAE is the unweighted mean of the green- and orange-hub target MAEs;
  position-only hub MAE is the unweighted mean of the top- and bottom-hub
  target MAEs. The companion summary reports arithmetic means and sample
  standard deviations over seeds 3072, 4072, and 5072. The six native/MLP
  sources are local fit-end metric JSONs; the position-aware runs were reused
  from the objective-swap grid and are locked by run, downloaded summary/config
  hashes, and source commit. The `current_projected_*` columns correct the
  source metric's legacy `encoder` label: the location extractor reads the
  post-projector current state. Checkpoints for the position-aware rows are no
  longer local, so their checkpoint-hash cells are intentionally empty.
- `lewm_planning_cue_ablation.csv` records the completed historical LeWM
  visible-versus-hidden Color and Shape planning evaluations. Every row includes
  the exact W&B run identifier, project, seed, solver, rank, and episode count.
  These are one-seed, separately trained models with a historical
  position-aware global state and auxiliary inverse-dynamics weight 3.125.
- `lewm_planning_protocols.csv` records source commits, checkpoint epochs,
  manifest hashes, and the within-pair matching scope. Color uses exact shared
  manifests. Shape uses condition-specific manifests with matched case
  identifiers and geometry. The TR2 commit comes from archived W&B metadata;
  its evaluator is identical to the later TR3 revision.
- `lewm_action_margin_cases.csv` contains the compact, case-level result from
  the frozen correct-versus-wrong portal-route audit. It reports the native
  cost margin and correct-route rank for the event-time-matched primary library
  and route-timing/precision sensitivities.
- `lewm_action_margin_summary.csv` contains the corresponding all-case and
  rank-stratified aggregates, including Wilson intervals for the correct-route
  unique-best rate. Rank 2 under event-time-matched BF16 scoring is the sole
  primary population.
- `lewm_action_margin_provenance.csv` records the source, checkpoint, dataset,
  manifest, candidate origin, and hashes of the ignored raw audit outputs. The
  180 primary candidates were newly prespecified before scoring; they are not
  recovered CEM/GD candidates.
- `lewm_checkpoint_bridge_probe.csv` records target-level and derived-mean
  location-probe metrics for the separate epoch-10, position-aware LeWM
  checkpoint used by the planning audits, together with constant-prediction
  baselines independently recomputed from the retained feature cache. Its
  companion provenance table makes explicit that probe and planning populations
  share a checkpoint but are not the same examples.
- `lewm_semantic_routing_interactions.csv` and
  `lewm_semantic_routing_rank2.csv` record the prespecified coupled cue/routing
  counterfactual. They separate the all-case primary paired interaction from
  the descriptive rank-2 candidate-ranking slice. The provenance table records
  raw hashes, validation gates, and the failed semantic-head competence check;
  fitted-head rescue results are therefore intentionally excluded.
- `lewm_semantic_routing_path_contrast.csv` records the direct within-case
  difference between replayed-outcome and predicted-rollout interactions in
  BF16 and FP32. Intervals use 20,000 paired bootstrap resamples with seed
  20260903; this is the inferential comparison of the two paths shown in the
  revised failure-localization figure.
- `lewm_semantic_routing_path_contrast_cases.csv` contains the 60 paired
  predicted, replayed, and replay-minus-prediction interactions used to
  regenerate those means, medians, and bootstrap intervals.
- `lewm_semantic_routing_rank2_cases.csv` is the case-level source for the
  predicted-versus-environment-replay margin panel. For each frozen rank-2 case,
  margin is the minimum wrong-route native cost minus the correct-route native
  cost; positive values therefore indicate that the correct route is uniquely
  preferred within the three-candidate library.
- `lewm_solver_trace_summary.csv` records the fresh first-replan CEM/GD audit,
  including best-evaluated versus returned routes, physical action-bound
  violations, and common-batch clipping-cost shifts. Its provenance table
  distinguishes this exact-code replay from the historical closed-loop runs
  and lists the first-replan and one-seed claim boundaries.
- `shape_followup_prespecification.csv` preserves the launch-time specification
  for the native-patch DINO and learned-global LpWM Shape follow-ups; its status
  column has been reconciled with the completed local and W&B artifacts.
- `shape_followup_results.csv` records their final-checkpoint per-coordinate
  MAE and $R^2$, averaged over the identity-conditioned or position-only hub
  and destination targets. The DINO ridge probe averages the patch-mode tokens
  to one 384-dimensional vector per frame; its decoder still consumes the full
  patch field. These are one-seed representation-location checks, not
  acquisition curves or model rankings. `declared_source_commit` is the commit
  recorded by each run's guarded `run_info.txt`, not W&B's automatically
  detected Git metadata.

The verified alternate palette is green/blue. Red is the agent color in the
rendered data; no red/green portal experiment was recovered from the tracked
repository, historical worktrees, manifests, or relevant W&B projects.

`shape_followup_prespecification.csv` remains a historical launch-time record;
only its completion field has been reconciled with the finished artifacts. It
should not be used as a current scheduler ledger.

## Static representation landscapes

- `lewm_similarity_maps.npz` contains compact raw maps for all six matched
  epoch-10 checkpoints: four reference images, 20x20 cosine and summed squared
  projected-distance arrays, valid-grid indices/positions, and exact reference
  positions. It also retains pairwise reference similarities/distances and
  the three renderings' reference images. Full encoder features stay ignored.
  There are 210 finite lattice positions per map; other positions are NaN.
  Keys begin with `lewm_<cue>_` or `lewm_idm_<cue>_`, where cue is `color`,
  `shape`, or `green_blue`. These arrays permit plotting without GPU inference.
- `lewm_similarity_summary.csv` provides the reference-pair scores and
  per-room grid means. All values describe one prespecified layout, episode
  1893, and are not aggregate planning results.
- `lewm_similarity_provenance.json` records source revision, checkpoint,
  configuration, manifest and raw-array hashes, reference coordinates,
  inference precision, and the exact interpolation/color normalization.
  `artifact://`, `dataset://`, and `source://` locators preserve provenance
  without exposing workstation paths; resolved local records stay ignored.
- `lewm_similarity_historical_provenance.csv` identifies the 15 historical
  checkpoint conditions and 30 cropped map panels in the three-page atlas.
  Each retained source was configured for a 40x40 grid. These are qualitative
  cosine maps with independent percentile scales, not a matched comparison.

Fresh cosine similarity uses current encoder-plus-projector states. The summed
squared distance uses those same static states; neither statistic calls the
predictor or evaluates an action sequence. A similarity peak is not a peak
in the planner's cost. Historical numeric colorbars are omitted because the
legacy renderer normalized the floor overlay and colorbar from different
arrays. Fresh maps use the same normalization for their colors and labels.

Historical DINO maps use encoded visual features after their configured
aggregator: Color uses patch passthrough followed by token mean, while Shape
uses its learned global MLP. The atlas labels distinguish these readouts.

`lewm_goal_distance_main_provenance.json` records the paired-goal main figure
from the preceding CLS draft using compact arrays: three CLS+IDM checkpoints, two leaf goals,
and summed squared projected distance. It preserves the input array SHA-256,
source provenance hash, reference indices, exact per-column scale bounds and
rendering contract. The input arrays and model outputs were not changed.

`lewm_spatial_similarity_maps.npz` and its companion `_provenance.json`
contain fresh static maps for the historical spatial + IDM3.125 Color and
Shape checkpoints used by the primary probes/planning. The NPZ follows the
existing `lewm_idm_<cue>_<metric>` key schema, but this separate file contains
SPATIAL readouts; the provenance explicitly labels that distinction. It
retains all four reference images and E/S arrays plus reference-pair metrics
and valid-center coordinates; full features remain ignored. The source
checkpoint hashes match the primary result records. The historical strict
loader used original configurations and state keys, with no conversion.
`lewm_spatial_goal_distance_main_provenance.json` records figure scales and
source hashes. `lewm_spatial_similarity_summary.csv` provides the two leaf
reference-pair metrics and posthoc Pearson correlation between goal E maps
over the 210 valid centers. These describe one layout, not aggregate
performance or absence of all cue information. The summary builder is
`runs/pta_revision_audit/spatial_map_summary.py`.

## Geodesic navigation reference

`tunnel_geodesic_reference.npz` contains distances to the same two leaf goals
at the same 210 valid positions as the learned maps. It also retains costs
at four reference positions, exact portal and landing geometry, and results
from half-pixel refinement. Its companion provenance JSON specifies the
finite state and action graph, source physics, and hashes.

State includes the rearm lock; plotted starting states are unlocked. Swept
triggers, paired landings, and rearming follow the simulator. Costs sum
physical movement before teleportation, in pixels. Exact point goals differ
from the planner's 16 px success region. Shortest paths are exact on the
declared graph; continuous-action optimality and numerical equivalence to
latent energies are not claimed.

The main figure samples and interpolates this reference exactly as the
learned maps, with a distinct pixel-distance scale shared by both goals.
Large computational graphs and temporary builders stay under ignored
`runs/pta_revision_audit/geodesic_reference/`.

`tunnel_geodesic_reference_validation.json` records independent simulator
replay: 2,297 sampled edges and 12 complete routes, plus structural checks
of all 420 query–goal paths. Position and lock states agree exactly; maximum
accumulated distance error is 1.41e-6 px. Nine relevant historical/current
physics methods also agree by AST.

## Historical decoder factorial and timing

`lewm_tr2_idm_readout_decoder_frames.npz` retains one input plus twelve
reconstructions from the TR2 CLS/spatial × six-IDM-weight sweep. The companion
provenance JSON records every source video hash, exact frame and crop,
configuration, run, source revision, and timing limitation. The stored final
checkpoint hashes identify the same runs, not the weights that produced
these periodic training snapshots. W&B rejected their requested history
step; retained video filename label 90001 is not treated as a verified optimizer
step or final-checkpoint identifier. Every input video is byte-identical.

`spatial_primary_decoder_sources.csv` now explicitly records that historical
`final_epoch_*` clips come from the first batch of the final training epoch.
Their final checkpoint hashes identify run provenance only. This differs
from the current matched CLS validation GIFs, recorded after final training.
