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
