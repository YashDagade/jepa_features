# Sensorimotor World Model baseline

The `smwm` baseline follows *Sensorimotor World Models: Perception for Action
via Inverse Dynamics* and the authors' MIT-licensed implementation at commit
`9d22bcc15c6ad8af8cc93337a950df2b49a866b9`:

- paper: <https://arxiv.org/abs/2606.20104>
- code: <https://github.com/petr-ivashkov/sensorimotor-world-model>

## Objective and modules

SMWM minimizes forward-dynamics MSE plus weighted inverse-dynamics MSE:

`loss = fdm_loss + lambda_idm * idm_loss`.

There is no SIGReg, InfoReg, VCReg, or temporal-alignment training term. IDM is
the sole anti-collapse objective. Its input is the ordered pair
`[z_t, z_{t+1}]`; its head is exactly `Linear -> ReLU -> Linear -> ReLU ->
Linear` with hidden width 256. The default `lambda_idm=0.1` is the official
TwoRoom setting.

## Deliberate local integration choices

The encoder, projector, action encoder, and AdaLN predictor reuse the clean
global-CLS LeWM implementation. This keeps observation preprocessing, context
length, decoder diagnostics, dataset partitions, and optimization exposure
matched across the four baselines.

The inverse module uses the paper-faithful ordered pair `[z_t, z_{t+1}]` and
the action width resolved from the dataset. No optional inverse loss is exposed
to LeWM or LpWM; SMWM's inverse term is retained because it is the method's
defining anti-collapse objective.
