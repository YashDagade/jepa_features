# Research memo: seen in patches, weak globally

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

The paper asks what happens between visual feature availability and the global
state used by prediction and planning:

> Color--place structure can be strongly accessible in LeWM's ordered patch
> features while remaining weak in its native global feature; separate audits
> find weak hue access after projection as well. Historical
> cue-visible and cue-hidden planning runs differ in some cells without solving
> the benchmark, but the present evidence does not identify the cause of those
> differences or the downstream planning bottleneck.

The compact pipeline is:

```text
pixels -> ViT encoder -> patch features (diagnostic branch)
                      -> global feature -> projected/predicted state
                                        -> latent objective -> optimizer -> action
```

The evidence establishes the first gap. It motivates, but does not establish,
a second gap between global accessibility and planning use. A compact formal
section now explains why sparse-event frequency can scale the predictive
incentive without fixing a universal acquisition threshold.

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

One selected state has patch hue BA .960, projected hue BA .516, target-color
BA .506, and correct-portal BA .521. It supports a downstream loss of semantic
access, not an "encoded but ignored" planning claim.

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
and not LeWM-specific: tested historical comparators also struggle with shape
identity.

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

The more separated blue/green palette does not materially improve
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
current/predicted color-conditioned hub MAE is 8.05/8.06 pixels, but both CEM
and GD achieve 10% rank-2 success. This suggests a possible utilization gap,
but it does not directly measure whether the native latent objective ranks the
correct portal below the wrong one.

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
- Existing episodic success does not identify the latent metric, predictor, or
  optimizer as the planning bottleneck.
- No verified red/green portal ablation has been recovered.

## Highest-value next experiment

Use frozen geometry-ranked scenarios and score a prespecified, horizon-matched
set of certified portal-directed candidate actions with the model's own latent
objective. Report:

```text
margin = min_wrong J(a-) - min_correct J(a+)
```

- Negative margin: the lowest-cost wrong candidate beats every certified
  correct candidate within the non-exhaustive set.
- Positive margin rules out only the tested initial candidate-ranking failure;
  candidate construction, optimization, conversion, and later replanning remain.
- Good candidates but bad CEM elite mean: multimodal averaging failure.
- Good standardized plan but bad re-standardized executed-equivalent plan:
  action-interface mismatch within the audited candidate set.

Then add an oracle cue coordinate or frozen semantic metric and test whether
changing only planning geometry changes behavior. A semantic cue swap should
change both visual identity and the actual routing rule while holding geometry
fixed.

## Current manuscript decision

The main paper is LeWM- and Tunnel-focused. It uses **patch features** and
**global features** as the conceptual distinction. Shape and alternate-palette
results are scoped ablations. Planning is a partial empirical result and an
explicit open mechanism, not the headline. Chaser and broad model comparisons
have moved to appendix context.

The current main text is eight full pages before references, with five main
figures, including a LeWM-only decoder progression across training;
references occupy two pages and the appendix is unrestricted.
