# Paper data

This directory contains compact, tracked numeric sources for the manuscript.
Large datasets, checkpoints, decoder videos, and raw W&B exports remain outside
the paper tree.

Files added for the LeWM-focused revision:

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
  support-means used for the 4.0--7.5% descriptive range in the main text. The
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
  identifiers and geometry.
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
