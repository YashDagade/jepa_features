# Experimental protocol and reproducibility

This supplement contains the LeWM implementation and the tools needed to run
the study's diagnostics. It is a source release, without the large historical
training corpora or trained checkpoints. The included paper figures report
archived experiments; they are not results of retraining this cleaned release.
Newly collected data and newly trained models should be reported as new runs.

## What each workflow measures

| Evidence | Entry point | Interpretation |
| --- | --- | --- |
| Location and cue MAE/R² | `scripts/probes/location_probe.py` | Linear accessibility of every named endpoint, separately for current and predicted states |
| Decoder montages | `scripts/train/lewm.py`, `lewm_idm.py` | Qualitative reconstruction from detached projected states |
| Near/far rank success | `scripts/plan/evaluate.py` | Receding-horizon planning through the correct tunnel |
| Static goal energy | `scripts/analysis/energy.py` | Squared distance between encoded observations and a goal observation |
| Geodesic reference | `scripts/analysis/geodesic.py` | Travel distance through the true paired tunnels on a finite action graph |
| Transition exposure | `scripts/data/audit_objective_exposure.py` | Counts of sampled windows containing tunnel transitions |

## Training and controls

The main recipe uses a randomly initialized tiny vision transformer with
14-pixel patches, a 192-dimensional projected state, history length three,
action blocks of five primitive actions, ten epochs, and training seed 3072.
The LeWM objective combines prediction MSE and SIGReg (weight 0.09). The primary
adaptation adds an inverse-dynamics MLP with weight 3.125 and a learned readout
from spatially aggregated patches. Native CLS selection and zero IDM weight
provide controls. All settings live in `scripts/train/config/`; the native
entry point does not allocate the inverse head.

The detached image decoder receives the projected state and is trained by
reconstruction MSE. Its gradients stop at that state. Local epoch montages use
the existing last validation batch and retain every selected frame and clip.
The paper's historical decoder panels instead use archived training snapshots;
these image-selection times must not be treated as interchangeable.

Coordinate probes use 640 fit/validation episodes and 256 held-out test
episodes. Fit statistics and fixed ridge regularization use only the former.
Detailed results retain x/y-averaged pixel MAE and coordinate-averaged R² for
each positional or cue-named endpoint. Cue recovery and location recovery
are different targets; neither should be replaced by an aggregate over agent
and tunnel positions.

## Planning

The paper evaluates 25 episodes per target rank. Rank one makes the correct
hub tunnel the first in the distance-based search order; rank two makes it the
second. Each case has a goal observation in the destination room. The primitive
step budget is 400, agent speed is 5, and success is reaching within 16 pixels
of the target under the environment's strict threshold.

Both planners use horizon five, receding horizon five, and five primitive
actions per predicted block. CEM uses 300 samples, 30 elites, and 30 iterations.
GD uses 100 candidates and 30 AdamW iterations with learning rate 0.1 and no
added optimization noise. Actions are normalized using the training corpus;
action clipping occurs in the environment. The supplied Color and Palette manifests
retain their 50 archived cases and rendered cue colors. The supplied Shape
manifest derives Shape rendering from the archived Color geometry; it is
explicitly a derived evaluation artifact. Private provenance
paths were removed; its file hash therefore differs from the private artifact.
Generated rank manifests must be rerendered to match the condition before use.

## Static energy and geodesics

For position p and goal observation g, the energy is
`E(p; g) = sum((z(image_at_p) - z(g)) ** 2)`, where z is the projected state.
This diagnostic calls the encoder and projector. It does not call the dynamics
predictor, optimize actions, or measure a calibrated travel distance.

The paper's fixed scenario is supplied for all three cue variants. A 20×20
lattice spanning the 224×224 image gives 210 valid collision-safe agent
centers. Four reference positions clear the largest cue markers. Geometry and
cues stay fixed while the agent position and reference goal change. Encoding
uses ImageNet normalization and float32 inference; distance arithmetic uses
float64. The archives also retain cosine similarities and raw features.

For display, eight-neighbor inverse-distance interpolation fills visible
floor pixels within each room. No energy is evaluated inside walls or at
invalid centers, and cue pixels are preserved. Yellow denotes low cost, with
a shared scale across goals within a model. Scales across models need not agree.

The geodesic reference tracks position and the locked destination portal.
It implements mandatory swept-radius triggering, exact landing offsets,
and the rearm boundary. Teleport jump length is excluded from movement cost.
The shortest paths are exact on a one-pixel finite graph with sixteen movement
directions, zero actions, and connections to exact queried positions. They are
an approximation to continuous-action travel distance. Goals are exact reference
points, unlike the planning success radius. The released reference is restricted
to the paper's two-leaf layout and pairing; unsupported settings fail explicitly.

## Limits

The transition-exposure utility audits how much cue-relevant dynamics the
objective sees. It does not reproduce the historical extra-transition training
pilot. That pilot remains inconclusive: brighter decoded cues did not produce
consistent improvements in linear cue recovery. This source release makes no
claim that increased exposure repairs the observed behavior.

Historical Color, Shape, and Palette checkpoints were produced across source
revisions. Equal nominal recipes do not establish a fully matched palette-only
intervention. The cleaned code preserves the retained model math, environment
contracts, and diagnostics, but exact numerical reproduction of every archived
run requires the corresponding original checkpoint and corpus. The paper's
single-seed and small evaluation-sample limitations still apply.
