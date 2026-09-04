# Figures

The active manuscript uses five main figures and two appendix figures. PDF and
PNG versions are tracked; LaTeX includes the vector PDF, and PNG supports quick
visual inspection. Temporary builders remain under ignored `runs/` paths.

## Main figures

- `lewm_tunnel_pipeline` shows a real three-room factual/counterfactual pair
  with fixed geometry and an exact three-route native-cost profile. The cost
  panel is action-conditioned; it is not a relabeled static encoder-similarity
  map. Builder: `runs/action-margin-v1/build_pipeline_figure.py`.
- `lewm_patch_global_accessibility` compares 192-dimensional ordered-patch and
  native-global readouts at initialization and epochs 1 and 10, with a separate
  epoch-50 checkpoint. Builder: `runs/final-paper-figures-v1/build_patch_global.py`.
- `lewm_stagewise_and_ablations` pairs actual standard-color, Shape, and
  alternate-palette frames with their corresponding hue or localization
  results. The columns are separate experiments and are not pooled.
  Builder: `runs/stagewise-figure-v2/build_stagewise_figure.py`.
- `lewm_action_margin` plots every fixed-candidate cost margin, the correct
  route's cost-rank distribution, and the route-timing sensitivity. Positive
  margins favor the cue-correct route. Builder:
  `runs/action-margin-v1/build_paper_assets.py`.
- `lewm_failure_localization` compares successive planning requirements: static
  cue-hub accessibility, predicted/replayed semantic response, replayed route
  ranking, and first-replan solver outcomes. Its semantic panel includes the direct
  within-case replay-minus-prediction contrast. Builder:
  `runs/failure-localization-figure-v1/build_failure_localization.py`.

## Appendix figures

- `lewm_native_global_decoder_progression` is a qualitative decoder control.
  The decoder is detached from the world-model features and is not used for
  planning, so the image strip does not define an acquisition time.
- `lewm_planning_open_problem` reports one-seed, separately trained
  cue-visible/cue-hidden planning controls for Color and Shape. Scenario
  geometry is matched within comparisons, but checkpoint variation is not.
  Builder: `runs/paper-revision-v3/build_planning_control_figure.py`.

Every quantitative plot reads compact paper tables or frozen experiment
outputs and checks expected populations or values before writing. Continuous
colors use perceptually ordered maps; categorical encodings pair color with
marker, line style, hatching, or direct labels. No panel contains author names,
account URLs, workstation paths, or mutable run-display titles.
