"""Inverse-dynamics-regularized Sensorimotor World Model."""

from stable_worldmodel.wm.lewm import LeWM


class SMWM(LeWM):
    """Native LeWM dynamics trained with forward and inverse dynamics."""


__all__ = ['SMWM']
