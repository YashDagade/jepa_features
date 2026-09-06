# Figures

The active four-page manuscript uses three main figures plus appendix diagnostics. PDF and PNG
versions are tracked; LaTeX includes the vector PDF, and PNG supports quick
visual inspection. Temporary builders remain under ignored `runs/` paths.

## Active main figures

The active manuscript uses `tunnel_variants_and_rank.pdf`,
`lewm_spatial_three_cue_decoders.pdf`, and
`lewm_spatial_three_cue_goal_distance.pdf`. Color, Shape, and Palette all use
spatial 8×8 + IDM 3.125. The decoder composite retains all six clips per
variant at frame 2. Color/Shape are first-batch final-epoch training snapshots;
Palette is validation after the final epoch. Timing/populations differ.

The energy composite has four columns: three learned maps and one unchanged
geodesic reference. All use the same 210 valid centers, reference positions,
portal geometry, and pairings. Learned energy shares a scale across goals
within each checkpoint; geodesic distance has separate pixel units. The
appendix `lewm_spatial_three_cue_landscapes.pdf` includes cosine similarity
and squared distance for both destination goals of all three checkpoints.

Compact source images, arrays, hashes, and timing are documented in
`../data/spatial_three_cue_figures_provenance.json`,
`../data/spatial_three_cue_landscapes_provenance.json`, and the dedicated
Palette static/decoder manifests. Ignored builders are under
`runs/pta_revision_audit/palette_continuity/`. Older two-cue figures below
remain archived artifacts and are no longer included by the manuscript.

## Retained task and CLS figures

- `tunnel_variants_and_rank` shows real standard Color, Shape, and green/blue
  Tunnel Room frames plus an oracle-certified Tunnel Room rank-2 scenario.  The numbered
  portal order is computed from the evaluation manifest, not from the model.
  Panel (d) uses filled 18-pixel markers from case `tr2-seed-852040`, with
  R1/R2 labels, approach arrows and a star at the sampled goal. The example
  is selected for separated portal geometry; panels (a–c) are unchanged.
  `data/tunnel_rank_illustration_provenance.json` records the exact scenario,
  distances, marker style and rendering checks.
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

- `tr2_color_mae_scale` places per-coordinate MAE on an actual 224x224 Tunnel Room
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

## Historical Tunnel Room readout and IDM sweep

`lewm_tr2_idm_readout_decoders.pdf/png` presents six selected Tunnel Room conditions
from the archived factorial: IDM weights 0, .125, and 3.125, with CLS and
spatial rows and a shared input. All panels use frame 0 of the first montage clip. Four 30×30 cue
crops use coordinates derived only from that input. The underlying videos
have identical input bytes. No hue, saturation, contrast, or content was changed.

Snapshots are retained periodic training media with filename label 90001,
not final-checkpoint reconstructions. The source report selected a common
periodic snapshot and a non-occluded frame using the input only. The new
figure fixes frame 0 across every displayed condition.
`../data/lewm_tr2_idm_readout_decoder_frames.npz` retains all twelve original
conditions; its companion provenance JSON identifies every source, the six
displayed conditions, and the figure builder.

## Submission font audit

All active figure PDFs use embedded TrueType fonts. Seven former Type 3
figures were regenerated with `pdf.fonttype = 42` for NeurIPS compliance.
Page dimensions and embedded scientific image bytes are unchanged; rendering
differences are confined to font rasterization. Before/after hashes and visual
comparison measurements are in `../data/submission_format_audit.json`.


## Controlled portal swaps

`lewm_portal_swap_states.pdf` illustrates the visible/routing intervention and
agent/portal-translation controls on a single training geometry excluded from
aggregate statistics. `lewm_portal_swap_raw_event_curves.pdf` plots observed
and predicted paired squared latent separation over 16 evaluation geometries,
with medians and interquartile bands, plus actual position separation. Time
zero is the first five-step sample at or after crossing; raw latent scales
are only comparable within each model. `data/lewm_portal_swap_arrays.npz`
retains the numerical plotting inputs. Source plots and larger artifacts are
under ignored `runs/pta_feedback/portal_swap/`.


`lewm_goal_landscape_and_portal_swap.pdf` is the current main Figure 3.
Panel (a) embeds `lewm_spatial_three_cue_goal_distance.pdf` unchanged;
panel (b) plots observed/predicted paired squared latent distances from
`data/lewm_portal_swap_arrays.npz`. Curves are medians and interquartile bands
over 16 evaluation geometries. The full appendix plot retains the additional
physical-position separation panel. `data/lewm_main_swap_figure_provenance.json`
records source and output hashes. The temporary builder is under ignored
`runs/pta_feedback/presentation/`.

## Active figure presentation

Before the Color-focused reorganization, all 14 figure placements were audited for embedded
descriptive titles, subtitles, and caption-like prose. These explanations now
belong in the LaTeX captions. Axes, legends, panel letters, and experimental
condition labels remain in the figures. Text-only PDF edits preserve source
images and plotted values; verification records are in
`../data/figure_title_cleanup_{decoders,landscapes,main_swap}.json`.

## Color-focused main figures

`lewm_color_reconstruction_and_energy.pdf` combines all six Color input/decoder
examples with the original Color energy maps and geodesic reference.
`lewm_color_portal_swap.pdf` pairs the saved two-world trajectory illustration
with the Color divergence curve. The energy panels are scaled uniformly to
match the reconstruction column height. The trajectory illustration is drawn
from the unchanged saved image/position arrays, with curved dashed arrows
for instantaneous teleports and a compact start/end legend. Composition bounds,
source hashes, and trajectory checks are recorded in
`../data/color_focus_figure_provenance.json`. Figure 1 is unchanged. The complete
three-variant figures remain available in Appendix L and the detailed protocols.
