"""Laplace-regularized predictive world model."""

from stable_worldmodel.wm.lewm import LeWM


class LpWM(LeWM):
    """LeWM dynamics trained with rectified Product-Laplace InfoReg.

    LpWM deliberately reuses LeWM's actual CLS-token encoder and causal
    predictor.  It differs only in its rectified projection heads and
    distribution-matching objective.
    """


__all__ = ['LpWM']
