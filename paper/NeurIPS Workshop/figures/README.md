# Figures

The active four-page manuscript uses three LeWM-based main figures and one appendix figure. PDF and PNG
versions are tracked; LaTeX includes the vector PDF, and PNG supports quick
visual inspection. Temporary builders remain under ignored `runs/` paths.

## Active main figures

- `tunnel_variants_and_rank` shows real standard Color, Shape, and green/blue
  TR2 frames plus an oracle-certified TR2 rank-2 scenario.  The numbered
  portal order is computed from the evaluation manifest, not from the model.
- `lewm_cls_cue_decoders` shows all six logged epoch-10 validation clips
  (zero-based frame 2) for Color, Shape, and green/blue under CLS + IDM 3.125.
  All three models use source `c4a2df269e16`, seed 3072, and the matched
  six-condition training protocol. The logged four-column GIF grid contains
  six clips and two empty padding tiles; the figure reflows the six clips
  into three columns without changing their pixels or order. Exact media,
  checkpoint-file, and tracked-image hashes are in
  `data/short_paper_cls_decoder_sources.csv`. The builder is
  `runs/pta_revision_audit/finalize_matched_cue_results.py`.

## Historical spatial decoder figure

- `lewm_three_cue_decoder_composite` shows input and detached-decoder rows for
  standard green/orange, triangle/diamond Shape, and alternate green/blue cue
  variants. It includes LeWM only. Exact source hashes and W&B provenance are
  in `data/short_paper_decoder_sources.csv`.
- `lewm_transition_diversity_decoder` shows representative LeWM decoder
  snapshots at baseline and +10/50/100% transition-centered windows. This is a
  qualitative, compute-confounded pilot with different frames across conditions; the non-monotonic probe results are
  frozen in `data/short_paper_transition_diversity.csv`.

## Archived supporting figures

The remaining tracked figures support older, broader drafts and appendices but
are not included by the active `main.tex`.

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

## Active appendix figure

- `tr2_color_mae_scale` places per-coordinate MAE on an actual 224x224 TR2
  Color frame. It visualizes exact L1 contours at 4, 15, 27, and 30 pixels. The
  caption and paper use task-specific null MAE and R2 rather than defining a
  universal pixel cutoff.

## Archived appendix figures

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

## Static similarity and squared-distance figures

- `lewm_similarity_matched` shows the three matched CLS+IDM models for an
  upper-leaf reference image, marked by the red agent. Its shared cosine scale
  is [0.6, 1.0], with yellow denoting high similarity. The canvas is 7.2 by
  2.35 inches. Main text treats the two bright destination regions as a
  descriptive finding in one layout, not proof of a learned teleport rule.
- `lewm_similarity_matched_appendix.pdf` has two pages: CLS+IDM, then native
  CLS without IDM. Three rows are Color, Shape, and green/blue. Four columns
  show cosine for the upper/lower reference, then summed squared projected
  distance for the same two references. Cosine retains the shared main scale;
  distance shares one printed scale between the two references of each
  model/cue checkpoint. Yellow means high similarity or low distance.
- `lewm_similarity_historical_atlas.pdf` has three pages covering 15 selected
  historical model/readout/cue/exposure conditions across LeWM, PLDM, LpWM,
  DINO, and SMWM. Each case shows the first hub and first leaf reference.
  Original map pixels are cropped without recovering numerical values.
  Their independently normalized colors are retained; inconsistent legacy
  numerical colorbars are removed. These historical cases are not matched
  to the new six-checkpoint comparison or to one another.

Fresh maps evaluate only the 210 valid centers of a 20x20 lattice. For display,
each room's visible floor uses eight-neighbor inverse squared-distance
interpolation, with weights `1/(distance_squared + 0.001)`. The overlay mixes
15% original floor and 85% heatmap; walls, markers and the reference agent
remain byte-identical. These plots use static projected embeddings, with no
actions, action-conditioned predictions, decoder, or simulator rollout.

The compact reproducibility arrays and provenance are in `../data/` under
`lewm_similarity*`. Ignored builders are
`runs/pta_revision_audit/build_fresh_similarity.py` and
`runs/pta_revision_audit/build_historical_similarity.py`; the frozen diagnostic
script is `runs/pta_revision_audit/fresh_similarity.py`. The fresh GPU job
finished without training; all source frames first passed byte-exact checks
against their dataset renderings.
