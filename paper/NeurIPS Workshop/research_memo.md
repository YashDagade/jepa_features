# Research memo: visible is not bound

## The paper we began with

The starting hypothesis was that a JEPA world model first learns factors
supported by most transitions and only later learns a visible color cue whose
consequence occurs at a sparse event. This predicted a geometry-first,
color-later curriculum and perhaps a common transition count at which color
would appear.

That account does not survive as a general result. Only some SMWM Tunnel runs
show stable binding after geometry. The other Tunnel objectives fail geometry
as well as binding, Chaser cue identity is decodable before training, and the
same objective can stabilize Tunnel binding at epoch 5, at epoch 12, or not by
epoch 50.

## The paper we have now

The stronger result is a decomposition of representation quality:

1. a cue can perturb visual features;
2. a color--place relation can be accessible in ordered spatial tokens;
3. the relation can be linearly accessible in native CLS;
4. a cue-conditioned outcome can be decoded;
5. a prediction can respond specifically at the event;
6. a detached decoder can render the appearance.

These measurements are not interchangeable and do not form a guaranteed
hierarchy. The shortest statement of the paper is:

> Predictive learning does not simply retain or discard a feature. It changes
> where the feature is organized and how prediction responds when it matters.

Tunnel Rooms establishes that spatial-token accessibility need not imply
native-CLS relational access. Chaser establishes that destination decodability
need not imply positive contact-enriched cue alignment. Decoder results show
that rendered hue is a separate property again.

## What the experiments show

### Tunnel Rooms: spatial access versus global organization

- At validated reconstructed initialization, a 192-dimensional PCA of the
  ordered patch map decodes color--place binding at balanced accuracy
  .965--.969 across three unique encoder draws. Native CLS and mean-patch
  readouts remain near chance. A fixed Gaussian random projection gives the
  same qualitative result.
- In the four historical epoch-50 runs, PCA-192 binding is .934--.969 for all
  objectives. Native CLS remains near chance for LeWM, PLDM-no-IDM, and LpWM;
  the observed SMWM run reaches .844.
- Three globally seeded LeWM/SMWM pairs completed 50 epochs. Within a seed,
  the predictive architecture, initial encoder, data order, and sampled-data
  exposure match; SMWM adds an auxiliary IDM head. LeWM develops
  neither persistent CLS geometry nor persistent binding in any seed.
- SMWM develops persistent geometry in all three seeds, beginning at epochs
  5, 5, and 12. Persistent binding plus semantic localization begins at epochs
  5 and 12 in two seeds. The third run has only isolated binding passes and is
  unsupported at epoch 50 despite strong geometry.
- Epochs 5 and 12 correspond to about 6.25M and 15.00M sampled prediction
  intervals (.56M and 1.34M teleport-bearing intervals). The third run remains
  right-censored after 62.50M total and 5.60M teleport-bearing exposures.
  There is therefore no model-independent transition threshold.

The paired comparison is objective-dependent, not an architecture effect.
It is also not an IDM-causal result: LeWM uses prediction plus SIGReg .09,
whereas SMWM removes SIGReg and adds IDM .1.

### Chaser: outcome access versus event enrichment

- Runner color is perfectly linearly separable from every native encoder at
  initialization. Later curves concern retention and reorganization, not first
  acquisition.
- LeWM, LpWM, and PLDM-no-IDM exceed a frozen contact-quadrant-only destination
  baseline by the earliest checkpoint, within epochs (1,2], and within epochs
  (2,3], respectively. SMWM's encoder advantage remains unsupported through
  epoch 50 in the observed run.
- LeWM, PLDM-no-IDM, and LpWM have large cue-swap alignment at both contact and
  the matched no-event window. Their event-minus-control simultaneous lower
  bands never become positive through epoch 50. This is a lack of supported
  contact enrichment, not a lack of cue response or proof that color is
  unused.
- SMWM shows positive contact enrichment beginning in the scheduled interval
  (10,20], while its encoder destination advantage remains unsupported. Its
  projector and predictor nevertheless recover high endpoint destination
  accuracy.

This Chaser result is a one-run-per-objective dissociation. Learned latent
scale and target geometry differ, so it should not be presented as a clean
cross-family ranking of semantic use.

### Reconstruction

LeWM and LpWM can render recognizable portal hues while native-CLS semantic
binding remains near chance. SMWM can expose strong binding while rendering
faint portal hues. Decoder optimization and pixel averaging therefore cannot
timestamp or establish semantic binding.

## Hypotheses that survived

- Visible information need not be organized as a native global relation.
- Outcome accessibility need not coincide with event-specific enrichment.
- Geometry can emerge before, or persist without, stable color binding.
- Objective choice and representation structure affect where information is
  accessible.
- Event frequency is one possible multiplier of predictive value, but it is
  not sufficient to explain the observations.

## Hypotheses that failed or remain unsupported

- There is no shared number of transitions at which these models learn color.
- The current LeWM Tunnel result is not selective color blindness because
  native CLS also fails the dominant geometry targets.
- The data do not establish a universal sigmoid or phase transition.
- Patch-level binding is not acquired late; it is already accessible before
  training in the validated control.
- The study does not establish that inverse dynamics, cue motion, or rarity
  causes the observed organization.
- A non-positive Chaser enrichment statistic does not show that a model has no
  cue response or makes no nonlinear use of color.

## Current best explanation

Random spatial ViT features already provide a rich, position-sensitive basis
for the color--place relation. The learned problem is therefore not simply to
create color information. It is to preserve or reorganize that relation in the
model's chosen global state and, separately, to make the predictive computation
respond at the relevant event. Frequency, effect size, redundancy, global
compression, regularization, and endogenous target geometry can all affect
those steps.

This explanation unifies the environments without pretending they are a
single-factor comparison. Tunnel diagnoses spatial-to-global organization.
Chaser diagnoses decodability-to-event-enrichment. Together they establish the
dissociations; they do not yet identify one mechanism.

## Highest-value next experiments

1. Run a matched SIGReg-by-IDM factorial with the same predictive architecture,
   initialization, data order, sampled-data exposure, and optimizer schedule. Measure patch binding, CLS
   binding, destination access, and cue-swap enrichment.
2. Build a static/moving-by-identity/binding factorial environment to separate
   motion coupling from relational complexity.
3. Construct a physically valid, destination-changing Tunnel counterfactual
   so semantic mapping and future outcome change together.
4. Replicate Chaser interventions and PLDM-no-IDM/LpWM Tunnel curves across
   globally seeded runs before making family-level rankings.
5. Evaluate the dimension-matched patch readout at epoch 50 for the completed
   seeded LeWM/SMWM runs to test whether spatial accessibility remains as
   stable as the historical runs suggest.

## Writing decision

The main paper should remain a four-figure argument:

1. diagnostic map and environments;
2. Tunnel binding at initialization and spatial-versus-CLS access;
3. completed paired Tunnel training trajectories;
4. Chaser destination access versus contact-enriched cue alignment.

Decoder grids, exact thresholds, bootstrap mechanics, hashes, job identifiers,
and incomplete DINO sweeps belong in the appendix. The headline should remain
precise: visible is not bound, and decodable is not necessarily event-specific.
