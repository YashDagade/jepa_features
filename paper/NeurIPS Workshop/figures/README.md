# Figures

The active four-page manuscript uses three main figures plus appendix diagnostics. PDF and PNG
versions are tracked; LaTeX includes the vector PDF, and PNG supports quick
visual inspection. Temporary builders remain under ignored `runs/` paths.

## Active main figures

The active manuscript uses `tunnel_variants_and_rank.pdf`,
`lewm_spatial_primary_decoders.pdf`, and `lewm_spatial_goal_distance_main.pdf`.
The decoder figure includes all six final-epoch clips at frame2 for Color and
Shape, spatial8×8+IDM3.125, aligned to the primary probe/planning checkpoints.
`../data/spatial_primary_decoder_sources.csv` records the source files and
hashes. The paired-goal landscape is documented at the end of this file.

## Retained task and CLS figures

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

- `lewm_goal_distance_main` shows both destination goals for all three matched
  CLS+IDM models as a six-panel static-energy figure. Scales and provenance
  are described in the paired-goal section below.

## Historical spatial decoder figures

- `lewm_three_cue_decoder_composite` shows input and detached-decoder rows for
  standard green/orange, triangle/diamond Shape, and alternate green/blue cue
  variants. It includes LeWM only. Exact source hashes and W&B provenance are
  in `data/short_paper_decoder_sources.csv`.
- `lewm_transition_diversity_decoder` shows representative LeWM decoder
  snapshots at baseline and +10/50/100% transition-centered windows. This is a
  qualitative, compute-confounded pilot with different frames across conditions; the non-monotonic probe results are
  frozen in `data/short_paper_transition_diversity.csv`.

## Archived supporting figures

The following figures support older, broader drafts and are not part of the
main argument. Active appendix landscapes are described separately below.

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
  2.35 inches. This retained single-goal cosine view is superseded in the preceding CLS draft
  paper by the paired-goal squared-distance figure.
- `lewm_similarity_matched_appendix.pdf` has two pages: CLS+IDM, then native
  CLS without IDM. Three rows are Color, Shape, and green/blue. Four columns
  show cosine for the upper/lower reference, then summed squared projected
  distance for the same two references. Cosine retains the shared [0.6, 1.0] scale;
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

## Main paired-goal distance figure

`lewm_goal_distance_main.pdf` and `.png` show static summed squared projected
distance for the three matched CLS+IDM checkpoints: Color, Shape and green/blue
columns; upper-leaf and lower-leaf goal rows. Canvas size is 7.2 by 3.9 inches.
The red agent marks each goal reference. Each column has one linear colorbar
shared by its two goals: yellow is low cost, purple high cost. Exact maxima
are 96.7884570, 122.0351787 and 103.1921188 respectively; colors across columns
do not share an absolute numerical scale. Walls and markers are preserved.

This figure reads the existing compact `lewm_similarity_maps.npz` directly,
with no model inference or new runs. It uses the same valid-center evaluation
and floor interpolation described above. `E` compares encoded static frames;
it is not an action-conditioned predicted rollout cost. The prior cosine
figure and full six-model appendix remain available. Figure-specific source
hashes, scale bounds and reference indices are recorded in
`../data/lewm_goal_distance_main_provenance.json`. The ignored builder is
`runs/pta_revision_audit/build_goal_distance_main.py`.

## Primary spatial checkpoint landscapes

`lewm_spatial_goal_distance_main.pdf` and `.png` compare historical Color
and Shape spatial + IDM 3.125 checkpoints, aligned exactly to the primary
probe and planning records, with the geodesic reference described below.
Rows select upper/lower leaf goals. Each column shares its
linear cost scale across goals: yellow low, purple high. Bounds are
[0,94.7650924] and [0,93.4958958]; the red agent marks the goal.
`lewm_spatial_similarity_appendix.pdf` adds both cosine and squared-distance
maps for both goals. Cosine uses the common [0.6,1] scale; E uses each printed
checkpoint scale. These are fresh static encoded-frame diagnostics, with
no action-conditioned rollout or new training.

The original immutable historical loader strictly loaded the original config
and state keys; no checkpoint conversion was performed. Both renderings and
all four reference frames/coordinates passed byte-exact checks against the
dataset and prior diagnostic inputs. Episode1893, the 20x20 candidate grid,
210 valid centers and floor interpolation are unchanged. Source/data details
are in `../data/lewm_spatial_similarity*` and
`../data/lewm_spatial_goal_distance_main_provenance.json`. Ignored builders:
`runs/pta_revision_audit/historical_spatial_similarity.py` and
`runs/pta_revision_audit/build_spatial_goal_distance_main.py`. Previous CLS
figures remain explicitly separate; they cannot be relabeled as spatial.

## Geodesic comparison in the main figure

`lewm_spatial_goal_distance_main.pdf/png` now has three columns: learned
Color energy, learned Shape energy, and a geodesic navigation reference
through the true pairing. The figure is 7.0 × 3.8 inches. The learned arrays,
their scales, and checkpoint attribution remain unchanged.

The geodesic column uses the same 210 positions, exact goal images, roomwise
interpolation, and preserved markers. Its separate pixel scale is shared
across goals. It includes triggering, landing, and rearm state, with a
declared finite movement graph and exact point goals.

Inputs and protocol: `../data/tunnel_geodesic_reference.npz` and companion
provenance. The builder is ignored
`runs/pta_revision_audit/plot_geodesic_comparison.py`. This is a navigation
reference; latent energies need not equal path length.
