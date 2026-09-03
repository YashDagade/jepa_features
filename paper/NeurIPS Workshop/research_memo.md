# Research memo: seen in patches, misranked in latent space

## The hypothesis we began with

The project began with a frequency-first hypothesis: a predictive model should
learn geometry from the majority of transitions and only later acquire a color
cue whose consequence occurs at a sparse transition. This predicted a
geometry-first, color-later curve and perhaps a common transition threshold.

That is not the strongest story supported by the completed evidence. LeWM's
ordered patch features make color--place binding linearly accessible before
predictive training. Its native global feature does not reliably acquire either
binding or geometry, and the historical global-state experiments use different
recipes. There is no supported universal sigmoid or transition threshold.

## The paper we have now

The paper is now a failure-localization study.  It asks where a visible,
task-relevant relation stops helping LeWM choose an action:

> LeWM can expose color--place structure in ordered patch features without
> reliably organizing it in its native global feature.  In a separate
> checkpoint, the static cue map is linearly accessible, but the dynamic agent
> consequence is not.  Re-encoded replayed native costs show a supported mean
> shift under rerouting, whereas predicted costs do not; native latent
> geometry still misranks those outcomes and first-plan optimization adds an action-feasibility
> mismatch.

The compact pipeline is:

```text
pixels -> ViT encoder -> patch features (diagnostic branch)
                      -> global feature -> projected/predicted state
                                        -> latent objective -> optimizer -> action
```

The evidence establishes four distinct empirical separations.  The primary probe audit separates
patch accessibility from native-global accessibility.  The matched checkpoint
separates static cue-map accessibility from action-conditioned future-agent
accessibility.  The coupled cue-to-route intervention partially separates the
predicted-cost path from native scoring of replayed outcomes.  The fresh
solver trace then separates objective misranking from return semantics while
exposing infeasible actions.  The evidence does not support attributing behavior
to one audited stage, and repairing one stage has not yet been shown to solve
closed-loop planning.

## Primary evidence

### Patch features versus the native global feature

- At validated reconstructed initialization, the 192-dimensional ordered-patch
  readout decodes two-room color--place binding at balanced accuracy
  .965--.969 across three independent encoder draws.
- The native 192-dimensional global feature scores .466--.502 at the same
  initialization states.
- At epoch 1, patch/global ranges are .957--.968/.460--.496.
- At epoch 10, patch/global ranges are .911--.962/.423--.483.
- A fixed Gaussian patch projection reaches .856--.882 at initialization,
  whereas a simple patch mean remains near chance.
- A separate historical epoch-50 checkpoint reaches .934 from ordered patch
  features and .531 from the native global feature.

These results do not show patch acquisition during training: patch access is
already high at initialization. They also do not show a stable semantic axis,
because the unsupervised patch projection is refit per checkpoint.

### Native global trajectories

All three native-global LeWM runs completed 50 epochs. None develops persistent
geometry or persistent color--place binding under the frozen joint criteria.
These runs therefore show a broadly weak global state, not selective color
blindness.

### Historical geometry--identity separation

A separate historical LeWM recipe uses a position-aware 192-dimensional global
state. Across three seeds, position-defined hubs are localized to
6.41 +/- 0.34 pixels, while color-conditioned hubs remain at
25.97 +/- 0.20 pixels against a 30.02-pixel semantic null. This is the clean
geometry-versus-identity result, but it must remain labeled as a separate
historical protocol.

Across twelve one-seed LeWM objective settings:

- native patch visible-hue BA is 96--100%;
- the selected global feature varies from .502 to .927;
- the projected state supplied to prediction is only .450--.563.

One selected state has patch hue BA .960, projected hue BA .516, and
target-color BA .506. It supports a downstream loss of semantic access, not an
"encoded but ignored" planning claim.

## Cue-appearance ablations

### Shape

The one-seed Shape experiment replaces color identities with shapes while
retaining the routing problem and randomized geometry. The reported MAEs are
means across hub and destination endpoints, not hub-only errors.

| Dataset | Identity MAE current/predicted | Position MAE current/predicted | Identity null |
|---|---:|---:|---:|
| TR2 Shape | 28.05 / 28.10 | 4.57 / 4.41 | 31.16 |
| TR3 Shape | 29.99 / 29.80 | 2.66 / 2.56 | 30.20 |

This repeats the identity--geometry gap with a non-color cue. It is preliminary
and does not establish LeWM specificity: the tested LpWM global state is weak,
whereas a pretrained DINO mean-token readout makes Shape identity accessible.

### Alternate palette

The recovered alternate palette is **green/blue**, not red/green. The red
object in the images is the agent. We found no red/green portal experiment in
the tracked repository, detached historical worktrees, manifests, filenames,
or relevant W&B projects.

The two blue/green runs use seed 3072 and match 10,000 scenario records to the
standard green/orange runs after excluding the two palette fields and the
palette-dependent scenario hash.

| Objective weight | Green/orange identity / position MAE | Blue/green identity / position MAE |
|---:|---:|---:|
| 0 | 25.79 / 6.06 | 25.87 / 5.75 |
| .625 | 24.84 / 4.50 | 25.48 / 4.37 |

The blue/green palette does not materially improve
identity-conditioned hub localization. Detached decoder separation retention is
2.91% and 2.10% in the blue/green runs.

## Planning evidence

The planner scores the final predicted projected state with an unweighted
latent squared distance to the projected goal. CEM or gradient descent
approximately optimizes that cost over standardized action blocks. Planning
therefore tests the global state, projector, predictor, metric, optimizer,
action conversion, clipping, and replanning jointly.

In one historical position-aware LeWM+IDM recipe, matched scenario manifests
compare visible and hidden cues. For three-room Color:

- CEM hidden ranks are 100/5/0%; visible ranks are 80/40/0%.
- GD hidden ranks are 95/20/0%; visible ranks are 80/45/5%.

For three-room Shape:

- CEM hidden ranks are 90/15/0%; visible ranks are 90/25/0%.
- GD hidden ranks are 80/15/5%; visible ranks are 70/25/15%.

The cue-visible Color run has higher observed rank-2 success than its hidden-cue
counterpart, while rank 3 remains almost unsolved. This does not show that cue
visibility caused the difference: the visible and hidden models were trained
separately with one seed, not compared through a frozen intervention. The direct
conclusion is only that neither condition solves the hardest cases.

A closer historical probe--planning match exists at auxiliary weight .125:
post-projector-current/predictor-output color-conditioned hub MAE is 8.05/8.06
pixels (`R²=.658/.655`) on an episode-disjoint probe set.  Yet current/future-
agent MAE is 39.59/46.35 pixels (`R²=.003/-.045`).  The eight-pixel result is
therefore evidence for an accessible static cue map, not for an accessible
action-conditioned consequence.  We used this exact checkpoint for the
remaining planning audits.

### Matched native-cost action ranking

For each of 60 frozen held-out three-room states, we prespecified three
deterministic portal-directed action sequences before evaluating model costs.
Each sequence spans the exact 25-primitive-action planning horizon.  In the
primary library, leading no-ops align portal entry to primitive step 25 for all
180 candidates, and every horizon is environment-replayed and certified.
Rank 2 is the prespecified primary population because the correct portal lies
behind one geometrically earlier alternative.

For

```text
margin = min_wrong J(a-) - J(a_correct),
```

positive values favor the correct portal.  Under the exact native predicted-
rollout/final-latent-distance cost and historical CUDA BF16 regime:

- the correct rank-2 route is best in 4/20 cases, second in 7/20, and worst in
  9/20;
- the primary median margin is -1.446 and the mean is -1.710;
- the correct route is best in 13/20 rank-1 and 4/20 rank-3 cases;
- across all 60 cases, the model-best route has geometric search rank 1 in 34
  cases.

FP32 event-time scoring yields 5/20 correct-best rank-2 cases and median margin
-1.413.  Moving the same physical routes before rather than after their no-op
padding also yields 5/20 and median -1.042.  BF16/FP32 and event-time/immediate
margin signs each agree in 19/20 primary cases.

This result establishes a native-cost route-ranking failure that does not require
numerical optimization: exact selection among these three semantic routes
would choose a wrong portal in 16/20 focal cases.  It does not quantify how
much that failure contributes to episodic planner failure, and it remains a
one-checkpoint, one-seed, non-exhaustive candidate audit.  Because the cost
combines the action-conditioned predicted rollout and an unweighted latent
distance, it does not by itself distinguish predictor error from metric
weighting or show that color is absent or ignored.

### Coupled cue-to-route intervention

For all 60 frozen cases, we keep geometry, agent state, hub cue markers, target
position, and candidate actions fixed.  We swap two leaf marker identities and
change the physical hub-to-leaf routing consistently, preserving the same
identity-matching rule.  All 360 candidate-condition executions are physically
replayed to the routed leaf.

We define a paired interaction whose positive direction moves cost from the old
correct route to the newly correct route.  In BF16:

- native cost through the predicted rollout has mean interaction `-.063`, 95%
  interval `[-.439,.312]`, with 2/60 desired old-correct-to-new-correct flips;
- native cost on the re-encoded environment-replayed final frame has mean
  interaction `+.604`, `[.357,.856]`, with 3/60 desired flips.

FP32 gives the same contrast (`-.033` predicted versus `+.583` replayed).
Re-encoded replayed native costs therefore show a supported mean shift, whereas
predicted costs show no supported shift.  But replacing prediction does not
rescue route ranking: in factual rank-2 cases the replayed final frame is correct-best in
only 4/20 BF16 cases and 3/20 FP32 cases.  The native representation/distance is
also poorly aligned with the decision.

A prespecified fitted semantic head cannot support a rescue claim.  Its
held-out endpoint and goal balanced accuracies are essentially chance, so the
competence gate fails.  We report that failure in the appendix rather than
interpreting the head's downstream rankings.

### Fresh objective-versus-search trace

On the 20 frozen rank-2 cases, CEM evaluates 9,000 candidates per case and GD
3,000.  For both solvers, neither the lowest-cost observed first-plan candidate
nor the returned first plan enters the correct portal.  Best and returned route
outcomes agree in 18/20 CEM cases and all 20 GD cases, so returning the best
observed candidate would not repair first-route selection in this realization.

Every best and returned plan violates the environment's physical action bounds.
Clipping the returned plan and re-scoring increases native cost in all cases,
by .485 for CEM and .616 for GD on average.  Since the environment clips during
execution, this is a real scored-versus-executable feasibility mismatch.  It
does not prove that clipping causes the semantic failure.  The trace is a fresh
first-replan diagnostic, not a recovery of historical candidate tensors or a
full receding-horizon evaluation.

## Claims that survive

- Ordered patch features can expose color--place binding even when the native
  global feature does not.
- The relation is not acquired late in patch features; it is accessible at
  initialization.
- Geometry can remain accessible while color or shape identity is weak in a
  historical global-state protocol.
- The projector can sharply reduce hue accessibility before prediction and
  planning.
- Historical cue-visible runs can exceed cue-hidden counterparts in some
  geometry-disfavored cells while both leave harder cases nearly unsolved; the
  cause of the difference is open.
- In the matched checkpoint, accurate static color-conditioned hub localization
  does not imply accurate future-agent localization or a useful route ordering.
- The native cost's route misranking occurs before numerical search for the
  audited three-candidate sets.
- A coupled cue-to-route change produces a supported average shift in native
  cost on re-encoded replayed outcomes but no supported shift through the
  predicted rollout.
- Replacing prediction with the replayed final frame does not rescue route
  ranking, so transition prediction is not the sole bottleneck.
- In the fresh first-plan trace, returning the lowest-cost observed candidate
  would not repair first-route selection; every retained plan also violates
  physical action bounds.
- A feature claim must name the representation stage, readout, target, and
  evidence type.

## Claims that fail or remain open

- There is no common transition count at which LeWM learns color.
- The data do not support a universal sigmoid acquisition transition.
- Native-global LeWM is not selectively color blind; geometry also remains
  weak in all three seeded runs.
- Decoder averaging does not establish missing semantic information.
- Patch accessibility does not prove that the projected planning state contains
  the same relation.
- Static cue-hub accessibility does not establish that action-conditioned agent
  outcomes are accessible.
- The counterfactual partially separates prediction from the cost on replayed
  outcomes, but it does not isolate goal encoding, representation geometry, and
  metric weighting from one another.
- The failed semantic-head competence gate does not support a frozen-semantic
  rescue claim.
- The first-replan trace does not establish whether a nonbest sampled route was
  physically correct or whether a bounded solver would solve closed-loop
  planning.
- No verified red/green portal ablation has been recovered.

## Highest-value next experiments

The highest-value remaining audits all keep this exact model and frozen cases:

1. **Feasible optimization.** Enforce physical bounds inside CEM/GD and score
   the clipped actions during optimization, then compare full receding-horizon
   behavior with the frozen implementation.
2. **Context match.** Give planning the three-state predictive history used in
   training rather than the maintained singleton reset context.
3. **Controlled stage repair.** Improve or replace the action-conditioned
   predictor under the native distance; separately fit a semantic metric that
   passes a held-out competence gate while freezing prediction.
4. **Replication.** Repeat the action-margin, coupled intervention, and bounded
   planning comparison across matched LeWM seeds.

## Current manuscript decision

The main paper is LeWM- and Tunnel-focused. It uses **patch features** and
**global features** as the conceptual distinction. Shape and alternate-palette
results are scoped ablations. Planning failure localization is now the
headline.  The new intervention shows a supported mean shift in re-encoded
replayed native costs but no supported shift in predicted costs, plus persistent
native-metric misranking, without treating either as the sole bottleneck.
Chaser and broad model comparisons are absent from the main narrative.

The decoder progression and descriptive historical planner comparison now live
in the appendix, along with the detailed action-margin sensitivities.  The four
main figures focus on the stage schematic, patch/global accessibility, cue
robustness, and the multi-stage planning audit.
The target remains nine pages of main text before references, with unrestricted
appendix length.
