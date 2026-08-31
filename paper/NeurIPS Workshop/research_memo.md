# Research memo: when JEPA world models organize predictive color

## The starting belief

We began with a simple frequency story: a world model would first learn the
factor supported by most transitions and later learn a visually obvious color
cue whose consequence appears only at teleport events. The strongest version
predicted a common, transition-count-dependent color onset across model
families and environments. An early LeWM Tunnel result was summarized as
selective color blindness: geometry was thought to survive while color was
discarded.

## What the completed audit shows

The condition previously abbreviated as `PLDM` is now labeled `PLDM-no-IDM`.
It is a local component ablation that removes the inverse-dynamics term from
published PLDM, not an objective-faithful PLDM implementation.

The matrix does not support a universal color-acquisition count. It contains
four qualitatively different regimes: initialization-level access, delayed
organization, transient access, and right-censored non-acquisition.

### Chaser

- Runner color is linearly separable from every native encoder at epoch 0.
  These curves measure retention and reorganization, not first acquisition.
- A leakage-free contact-only baseline is now fit on 1,024 fit-plus-validation
  episodes, frozen, and scored on 512 test episodes (BA .51118). Against this
  reference and a one-sided simultaneous band over 36 cells, LeWM encoder
  destination advantage is first confirmed and persistent by the earliest
  saved checkpoint, LpWM in (1,2], and PLDM-no-IDM in (2,3]. SMWM remains
  right-censored beyond epoch 50. This privileged baseline is not a Bayes or
  all-pixel oracle; advantage can reflect any extra visual context.
- SMWM temporarily forgets raw cue identity around epochs 3--6 and recovers by
  epochs 8--13. This is a real within-run reorganization pattern, not color
  first appearing late.
- A prespecified paired cue swap across nine frozen checkpoints gives the
  strongest causal-response diagnostic. In the observed runs, LeWM,
  PLDM-no-IDM, and LpWM have no positive simultaneous lower band by epoch 50.
  SMWM is strongly anti-aligned
  at the scheduled epochs 2, 3, and 5, crosses to .3908 at epoch 20, and has first-confirmed and
  persistent response in the interval (10,20], or (1.224M,2.448M] repeated
  event-bearing intervals. At epoch 50 the four estimates are -.0108, .0013,
  .0009, and .5069. The inverted encoder-versus-response trajectories show
  directly that destination decodability is not event-specific predictor use.

### Tunnel Rooms

- Native-CLS LeWM, PLDM-no-IDM, and LpWM do not expose durable color-to-slot binding
  by epoch 50, after about 5.596M repeated teleport-bearing intervals.
  Crucially, they also fail agent geometry. Their result is broad global-state
  failure, not selective abstraction over color alone.
- The observed SMWM run is the sole clear geometry-first, binding-later
  native-CLS organization trajectory. Agent localization is persistent from epoch 4;
  the full geometry criterion, including nearest-entrance identity, is
  persistent from epoch 6 (about .672M event-bearing intervals),
  the first confirmed binding crossing starts at epoch 8 (about .895M), and
  binding plus color-conditioned localization is persistent from epoch 13
  (about 1.455M).
- In this one SMWM run, agent MAE is well summarized by a logistic midpoint at
  epoch 5.81 and binding BA by a midpoint at epoch 15.26. Semantic-place MAE
  is not a clean sigmoid: a broken-linear fit is slightly better. These are
  descriptive curve summaries, not phase transitions or onset estimators.
- LeWM has a joint crossing at epochs 32--33 and then loses it. This is the
  clearest warning against treating two thresholded checkpoints as durable
  acquisition.

### Spatial aggregation and decoders

- A same-checkpoint control localizes the apparent Tunnel loss to global
  access. Fixed 8x8 pooled patch tokens from the native epoch-50 checkpoints
  achieve binding BA .909, .930, .949, and .949 for LeWM, PLDM-no-IDM, LpWM, and
  SMWM, while the first three CLS readouts are .531, .495, and .492. Their
  semantic-hub R2 values move from negative at CLS to .867, .846, and .955.
  The information survives in patch tokens.
- A completed fixed-width follow-up rules out raw feature count as a sufficient
  explanation. PCA compressions of the same spatial map to 192 dimensions
  reach binding BA .934, .934, .969, and .968. Their paired improvements over
  CLS are .403 [.336,.470], .439 [.371,.506], .477 [.411,.542], and .125
  [.077,.173]. A fixed Gaussian random projection also improves every run,
  while mean pooling fails for the first three. A location-sensitive sketch of
  ordered contextual tokens is sufficient for held-out linear decoding at
  these checkpoints; raw width and global token averages are insufficient
  explanations.
- The epoch-50 fixed-width result is still conditional on one frozen run per model. PCA
  uses a fit-plus-validation, unlabeled basis; the random projection is the
  stricter non-data-fitted control. Neither establishes that spatial
  aggregation caused the representation to be learned, nor separates the
  external token index from position/context encoded inside each token.
- The three-anchor follow-up changes the temporal interpretation. At epoch 1,
  PCA-192 binding BA is .961, .957, .976, and .976, while CLS is
  .471, .462, .500, and .506. All four pass the prespecified 98.75% familywise
  sensitivity lower bound and paired PCA-minus-CLS interval. No PCA
  epoch-1-to-50 binding change is conclusive. Compact patch access is present
  by one completed epoch and is not a demonstrated late acquisition. SMWM is
  the exception at CLS: BA changes .506→.525→.844 across epochs 1/10/50, with
  a positive epoch-10-to-50 paired change. Its CLS access is bracketed between
  those anchors, not timestamped exactly.
- The secondary nominal adjacent-anchor intervals are nonmonotonic: SMWM
  changes -.072 [-.110,-.036] from epoch 1 to 10 and +.065 [.027,.103] from
  10 to 50; PLDM-no-IDM changes -.032 [-.060,-.004] from 10 to 50. PCA-minus-CLS and
  all epoch-change intervals are unadjusted nominal 95%; only epoch-1 PCA
  accessibility receives the four-cell 98.75% sensitivity check.
- PCA is fit separately for each model--checkpoint cell; only the Gaussian
  random-projection matrix is identical across all cells. Cross-checkpoint PCA
  changes therefore compare probe performance, not coordinates in one shared basis.
- The epoch-1 patch result now replicates across three globally seeded LeWM and
  three globally seeded SMWM checkpoints. CLS binding is .460--.496 and every
  nominal interval contains chance; PCA-192 binding is .957--.984 and every
  Bonferroni-six 99.1667% lower bound exceeds .5. All six adjusted paired
  PCA-minus-CLS lower bounds are positive. Fixed Gaussian projection is also
  strong (.870--.913), whereas mean pooling remains near chance. This is the
  first result in the paper that survives multiple globally seeded training
  initializations. The six cells are
  three paired seed values: within each value, LeWM and SMWM share the encoder
  initialization and data order, so they are not six independent seed draws.
- A validated post-hoc reconstruction now resolves the epoch-1 left censoring.
  All six native epoch-0 probe artifacts reproduce exactly before any patch
  extraction (106 floating metrics per artifact, maximum error zero). At reconstructed
  initialization, the three unique encoder draws have CLS BA .466--.502,
  mean-patch BA .452--.511, PCA BA .965--.969, and fixed-projection BA
  .856--.882. Every adjusted PCA lower bound exceeds .5 and every adjusted
  PCA-minus-CLS lower bound is positive. Every nominal and adjusted PCA
  epoch-1-minus-epoch-0 interval contains zero; all 24 six-cell-sensitivity
  readout-change intervals contain zero. Compact patch binding therefore precedes training in
  this control. Training may reorganize global access or predictive use, but
  did not create the simply decodable local relation during epoch 1. This is a
  validated reconstruction, not a saved checkpoint, and represents three
  unique initial encoders rather than six independent draws.
- The prospectively specified epoch-10 anchor replicates compact patch access
  in the same six observed checkpoints. LeWM PCA BA is .911--.962 (mean .941)
  and SMWM is .960--.983 (mean .975); every adjusted PCA lower bound exceeds
  .5 and every adjusted PCA-minus-CLS lower bound is positive. The epoch-1/10
  anchor-persistence rule therefore passes, but it does not establish
  continuous retention between anchors.
- The prespecified five-point noninferiority rule fails only for LeWM seed
  3074: PCA changes -.047 with adjusted interval [-.093,-.002], a modest
  detectable decline that remains far above chance. The strict hidden-from-CLS
  all-six tier also fails for two different reasons: LeWM seed 3074 is below
  chance with a nominal interval that does not contain it, while SMWM seed 3072 has strong
  CLS BA .918 and mean-patch BA .929. The other two SMWM CLS cells remain near
  chance. This seed heterogeneity argues for global-state routing and
  organization, not a uniform architecture-level onset.
- A separately quarantined post-hoc native-CLS audit resolves epochs 0--10 for
  the three SMWM runs. Seed 3072 jumps from binding BA .486 at epoch 4 to .910
  at epoch 5 and remains above criterion through epoch 10 (.918); its agent and
  semantic-location criteria are earlier, while full geometry and binding
  begin at epoch 5. Seed 3073 develops persistent geometry at epoch 5 without
  binding, and seed 3074 has unstable geometry with no confirmed binding.
  Epoch 5 is about .560M repeated teleport-bearing interval exposures under
  this loader; the two non-binding seeds are right-censored beyond 1.119M at
  epoch 10. These are repeated sampled loss exposures, not unique transitions.
  Geometry without confirmed binding therefore appears in two observed runs, whereas
  a uniform geometry-first/binding-later SMWM schedule is not. These landmarks
  are post-hoc, right-censored at epoch 10, and descriptive.

- All four separately trained spatial-8x8 Tunnel encoders expose color-place
  structure at epoch 0. Their trajectories measure retention or erosion, not
  transition-driven acquisition.
- The spatial condition is not a pure CLS-versus-patch readout swap. It adds a
  learned LayerNorm--linear aggregation path, about 2.38M parameters per run,
  and an independent initialization. It demonstrates an architectural boundary
  condition but does not identify pooling as the cause.
- Detached decoder appearance and linear semantics disagree. LeWM and LpWM can
  render green and orange portals while their semantic binding probes remain
  near chance; SMWM can expose binding while rendering faint portal hues.
  Decoder optimization, linear access, relational binding, predicted-state
  access, and supported event-specific predictor response must remain separate claims.

### Exact Tunnel intervention

- A render-only green/orange swap now holds underlying simulator state, geometry, actions, physical
  portal pairing, and destination fixed across 184 first-teleport test
  episodes. The pixel gate found no changes outside the four marker rectangles.
- LeWM, PLDM-no-IDM, and LpWM have near-unit swap-response alignment on both
  ordinary and teleporting steps; none has supported event-minus-control
  enrichment or suppression after a 12-cell simultaneous correction. Their
  weak global binding probes therefore coexist with endpoint visual swap
  sensitivity to the persistent marker colors.
- SMWM's swap alignment falls at teleport at encoder, projector, and predictor
  (event-minus-control -.197, -.304, and -.188); every simultaneous interval
  is negative. This is event-specific directional reorganization, not color
  acquisition. Across all stage and target RMS diagnostics, SMWM spans
  .008--.219 (.008--.022 at projector/predictor and .176--.219 at encoder),
  versus 1.119--1.509 in the other models; learned stage scales differ, so the
  cosine result does not rank cross-family response magnitude.
- The swap changes every colored marker while routing is fixed. It does not
  establish semantic binding or counterfactual destination use. It strengthens
  the paper precisely because it rejects the shortcut inference from visual
  sensitivity to relational use.

## Hypotheses that survived

1. **A visible feature need not be organized as a predictive relation.** This
   is the strongest supported thesis. Chaser cue identity, destination access,
   Tunnel color-to-slot binding, exact swap response, and decoder appearance
   give different answers.
2. **Native CLS can organize factors in an ordered within-run sequence.**
   Tunnel SMWM organizes geometry before stable CLS binding. The claim is local
   to this run, not a universal model-family law.
3. **Event frequency is not sufficient.** Chaser and Tunnel have nearly
   matched event-bearing interval rates but different outcomes. Frequency may
   still matter as a multiplier of predictive value.
4. **Stage and representation structure matter.** The SMWM Chaser encoder
   never jointly exceeds the cue-blind contact baseline while its projector and
   predictor become strong after epoch 20; a CLS-sized random or PCA
   projection of same-checkpoint patches retains Tunnel binding that CLS hides,
   already at reconstructed initialization across three unique encoder draws;
   mean pooling usually loses it;
   and the spatial training architecture
   exposes a different factorization from native CLS.
5. **Decodability and predictive response dissociate over training.** Three
   observed Chaser encoders gain destination access by epochs 1--3 but show no
   positive cue response through epoch 50; SMWM gains positive response in
   (10,20] while its
   encoder remains below the simultaneous contact-baseline criterion.

## Hypotheses that failed or remain unsupported

- There is no common number of transitions at which these models pick up
  color.
- LeWM's current Tunnel behavior is not clean selective color blindness; it
  also fails the dominant geometry targets.
- The Chaser SMWM encoder does not have a supported destination-access onset by
  epoch 50 under the frozen contact-baseline simultaneous criterion. A pointwise epoch-36
  crossing is insufficient after the 36-cell correction.
- The data do not establish a general negative-sigmoid or phase-transition law.
- Compact patch binding is not first acquired during epoch 1 in the globally
  seeded LeWM/SMWM control; it is already accessible at reconstructed
  initialization.
- The current matrix does not establish that inverse dynamics, moving-cue
  attachment, or event frequency causes retention.

## Current best explanation

A candidate factor is valuable only insofar as retaining it reduces target
prediction risk. If its incremental dynamical consequence occurs with
probability $\rho$, frequency can scale that value, but raw persistent color
may also contribute outside event intervals. The effect in learned target
space is endogenous: these runs use the same trainable encoder and projector
on both sides of a non-detached squared prediction loss. Relational complexity,
global compression, other easier predictive factors, and objective-aligned
gradients can therefore move a run among the observed regimes. Tunnel requires a
color identity to bind to a randomized spatial slot while both colors are
simultaneously visible. Chaser presents one episode-level cue attached to the
moving object. This makes relational binding and representation structure the
leading explanation, with cue motion and inverse dynamics still confounded.

The initialization control adds an important refinement: the random spatial
ViT features are already rich enough for a simple linear readout to recover the
color--slot relation. The learned problem is therefore not necessarily to
create local color information. It is to preserve and route that relation into
the compressed global state and, separately, recruit it for predictive
response at the relevant transition. This access--organization--use hierarchy
is the current best unifying explanation of the Tunnel and Chaser results.

## Corrected DINO control

The completed DINO job was initially believed to use a learned CLS projector
and inverse dynamics. Its resolved configuration is instead a frozen raw-DINO
CLS control with IDM disabled. It shows that pretrained DINO already exposes
current agent position (encoder $R^2=.961$), while the learned predictor makes
future position linearly decodable ($R^2=.844$ at fit end). The intended
projector/IDM sweep stopped after roughly 10--12 epochs. Its three largest IDM
weights end at $R^2=.975$--.980 while the lower five range from $-.328$ to
.774, but cells stop at different epochs and lack matched initialization.
This is an incomplete endpoint pattern, not evidence of an IDM threshold.
Hidden TR3 has no color intervention, so no DINO color claim is supported.

## Current reviewer-risk audit

1. **Most trajectory conclusions are still single-run.** The six seeded
   epoch-1/10 anchors and three unique reconstructed initializations now
   establish compact patch accessibility before training and again at epoch
   10, but they do not
   replicate a full learned onset curve. The correct defense is to keep all
   historical acquisition intervals descriptive and finish the active seeded
   trajectories; no wording can substitute for that experiment.
2. **The mechanism remains underidentified.** Chaser and Tunnel differ in cue
   motion, cue count, relational binding, object size, and dynamics. The exact
   swaps sharpen the measurement distinction but hold physical destinations
   fixed, so they do not prove counterfactual routing. A factorial environment
   or valid destination-changing intervention is needed before claiming that
   motion, inverse dynamics, or event frequency causes retention.
3. **The probes audit the training distribution, not encoder generalization.**
   Test episodes are isolated from probe fitting, but the self-supervised
   encoder saw the source HDF5 containing them. This is appropriate for asking
   what the trained representation organizes, but a separate encoder-unseen
   evaluation set is required for a generalization claim.

The current draft states all three limitations directly. Its strongest
defensible contribution is therefore the measurement framework plus replicated
initialization-level and epoch-10 anchor accessibility, not a universal acquisition law or a settled
mechanism.

## Highest-value next experiments

1. **Complete the replicated trajectories.** Evaluate epoch 50 for the
   three seeded LeWM and SMWM runs, and add seeded PLDM-no-IDM/LpWM cells; then
   test a learned aggregation intervention during training.
2. **Replicate the checkpointed interventions.** Repeat the completed Chaser
   trajectory and the exact Tunnel rendering swap across globally seeded
   replicas; design a valid Tunnel counterfactual that changes semantic routing
   without creating an off-dynamics clip.
3. **Globally seeded onset replication.** Finish the active key training cells
   with three stochastic seeds and prespecified onset windows before making
   family-level transition-count claims.
4. **Event-exposure intervention.** With validated global seeding, vary the event
   rate while holding unique examples, updates, initialization, and non-event
   sampling fixed.
5. **Objective and environment factorials.** Sweep a small discriminating set
   of inverse-dynamics weights and independently vary cue motion, cue count,
   and binding structure.
6. **Internal versus external position.** Do not interpret per-episode patch
   shuffling as a clean causal control: it creates a sample-specific
   measurement matrix and turns a negative result into an ambiguous noise
   effect. Instead compare matched permutation-invariant set probes on token
   content alone, content plus correct row/column, and content plus randomly
   reassigned coordinates; separately test whether token values linearly
   reveal their source grid position. Mean pooling already exhausts the linear
   permutation-invariant case, so this diagnostic necessarily introduces
   controlled nonlinearity.

## Experiments started in the extended session

- A minimal global-seeding fix was implemented and tested in isolated commit
  `61b3bf4da93e9a3e5a3a5e88419d0672fb6bf6fd`. It calls Lightning global
  seeding before dataset, model, and Trainer construction, uses the normalized
  integer seed for loader generators, and forwards it to the experiment
  manager. Same-seed initialization hashes match and different seeds differ in
  focused tests; the full native-global test file has 28 passing tests.
- Six guarded 50-epoch native-CLS Tunnel replications are running: LeWM and
  SMWM at seeds 3072, 3073, and 3074 (jobs 16637725--16637730). All passed the
  first-backward health check. The three LeWM replicas are near chance at their
  first post-training binding probe (BA .460--.496, every interval containing
  .5); these live runs are not included in completed-result claims.
- The leakage-free checkpointed Chaser cue intervention completed as job
  16634274 over epochs 1, 2, 3, 5, 10, 20, 36, 45, and 50. Slurm reports
  `COMPLETED 0:0`; result SHA-256 is
  `06799748ee8a4b78dd0cfe119aa8bc43447b222cb99d7d0c10aad60b758f390a`.
  An independent first-pass audit reproduced the contact baseline, all 108 saved stage
  BAs, every destination bootstrap, both onset grids' algebra, and the encoder
  simultaneous critical value exactly.
- An epoch-1/10 outcome-blind fixed-width Tunnel extension, anchored by the
  already-observed epoch-50 result, completed as job 16642019
  (`COMPLETED 0:0`); result SHA-256 is
  `222acd9f282a5e8abfab42a1ce7e77ebed2898d2d871425449b7ef6ac4559f9c`.
  Its 48 binding BAs, 84 paired bootstraps, primary decisions, and epoch-50
  replication were independently recomputed with zero mismatches.
- A globally seeded epoch-1 fixed-width replication completed as job 16644629
  (`COMPLETED 0:0`) for LeWM and SMWM seeds 3072--3074. All six checkpoints
  satisfy the prespecified hidden-from-CLS tier after the Bonferroni-six
  sensitivity analysis; result SHA-256 is
  `908153cbe9db0795ed5c21fe5546a7d7f6f47f5c7d5d79332c55e7e859057843`.
  An independent audit reproduced all primary metrics and bootstraps.
- A reconstruction control completed as job 16648080 (`COMPLETED 0:0`);
  result SHA-256 is
  `42671ac5d3c5afd58213f72f70cb1182397bfaecbe126731d0db7d38a9f5921d`.
  All six native epoch-0 artifacts reproduced exactly before any patch
  extraction. Recursive audits independently verified all 24 confusion
  matrices and intervals, 42 paired bootstraps, cell decisions, and aggregate
  decisions with zero mismatches. The three unique initial encoders already
  expose binding through PCA and fixed random projections, with no adjusted
  readout change established after one epoch.
- The prospectively frozen epoch-10 capacity control completed as job 16652597
  (`COMPLETED 0:0`); result SHA-256 is
  `afa2229cd941ed5cfa8cf445bc9cf1b1701a4d57f14c019bdf71f68effe60e17`
  and manifest-v2 SHA-256 is
  `dd7930c5cd62a79dd90f0444e589bd03273841bd23d04957b529f03347fabafa`.
  The first manifest was preserved but not used after an outcome-blind auditor
  requested an explicit absence-at-freeze record. Two independent numerical
  implementations reproduced 24 metrics, 18 within-checkpoint bootstraps, 24
  epoch-10-minus-1 bootstraps, all decisions, summaries, and provenance with
  zero mismatches.
- The post-hoc SMWM native-CLS epoch-0--10 audit has result SHA-256
  `d31cd6eb42e84afefdcf95422c643c717f66f76c333a350906a9250ff4c8d463`.
  Its independent recomputation found zero mismatches over 33 rows, 66
  classifier records, 132 location records, 18 series, 36 landmarks, 45
  hashes, and 83 provenance checks. The frozen semantic threshold is
  31.187048; direct float64 recomputation is 31.1870489686 and changes no row.
- The exact Tunnel green/orange rendering swap completed as job 16644903
  (`COMPLETED 0:0`) on 184 held-out first-teleport episodes; result SHA-256 is
  `1fb81c7b6dec5151641c647a19d49790640878624d077f34fa76f920d49cb568`.
  Its pixel gate and independent sufficient-statistic audit both passed.

The paper should therefore lead with the measurement distinction and four
regimes. The geometry-first/binding-later Tunnel SMWM native-CLS trajectory is the most
interesting positive case, but it should not be inflated into a universal law.
