# Research scope

## Question

When does a JEPA-based world model retain a persistent visual feature, and
when does average latent prediction make that feature expendable because its
dynamical consequence is rare?

## Native baseline contract

1. LeWM uses its actual CLS token, native action-conditioned predictor,
   prediction MSE, and SIGReg.
2. PLDM uses its actual CLS token, native prediction loss, VCReg terms, and
   temporal alignment. Its historical zero-weight inverse head is omitted.
3. LpWM uses its actual CLS token, native RepReLU projector, prediction MSE,
   and InfoReg with `mu=0`, `p=1`.
4. Sensorimotor World Model uses its actual CLS token and the published
   forward-dynamics plus `0.1 * inverse_dynamics` objective. This native term
   is not an auxiliary ablation.

Every baseline consumes the actual ViT CLS token plus actions; only SMWM has
an inverse objective because that term defines the method.

## Evidence contract

- Separate dominant approach/chase transitions, rare teleports, and ordinary
  non-approach transitions instead of forcing every frame into a binary label.
- Probe native encoder CLS, projected state, and predicted state separately.
- Measure geometry/motion and cue-conditioned identity on episode-disjoint
  splits.
- Pair decoder evidence with probes; an averaged reconstruction alone is not
  proof that all color information is absent.
- Report the realized event exposure under the exact training window and
  frameskip, not only the raw adjacent-frame rate.
- Do not infer representation use from planner success alone.
- Preserve dataset hashes, source commit, resolved configuration, seed,
  checkpoint, and W&B run ID for every result.

## Explicit LeWM + IDM ablation

The user authorized a matched Color/Shape/green-blue cue study with a uniform
auxiliary IDM weight of 3.125, paired with native no-IDM controls. The separate
`lewm_idm` training entry implements this historical adaptation; it must never
be described as native LeWM. It uses projected `[next, current]` states and a
512-wide LayerNorm/GELU head to predict the full normalized action block.
Keep `lewm.py` and its native prediction + SIGReg objective unchanged. Use the
same source revision, seed, data interfaces, budget, and CLS readout across the
six conditions. The user subsequently selected spatial aggregation plus IDM as the primary
workshop recipe, with CLS and no-IDM runs as explicit controls. Use the
existing canonical `feature_mode=global_spatial_8x8` adapter; preserve native
`global_cls` defaults and do not alter the native objective. A targeted
confirmation campaign covers spatial+IDM Color/Shape/green-blue and spatial
Color without IDM under the same source, seed, and budget.

Label readout and IDM weight for every reported condition. The historical
spatial Color gap occurs with and without IDM; native CLS without IDM fails
both location and cue probes. Do not describe this as a selective gap that
persists uniformly across every objective/readout, or as proof that IDM has
no influence on learned dynamics.

## Current workshop focus

The main workshop narrative and subsequent experiments focus on Tunnel Room
Color (green/orange). Preserve existing Shape and Palette results in the
appendix; do not launch new variant runs unless the user requests them.
Report their shared qualitative pattern without claiming statistical equivalence.
