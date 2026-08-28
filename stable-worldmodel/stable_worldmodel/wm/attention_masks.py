"""Attention masks shared by patch-token world-model predictors."""

import torch


def frame_block_causal_mask(
    num_frames: int,
    num_patches: int,
) -> torch.Tensor:
    """Return the boolean mask for joint space-time causal attention.

    Tokens are ordered frame-major as ``(t0p0, ..., t0pP, t1p0, ...)``.
    Every query can attend to every patch in its own frame and in earlier
    frames, but never to a patch from a future frame.  The returned
    ``(1, 1, T*P, T*P)`` shape broadcasts over batch and attention heads in
    :func:`torch.nn.functional.scaled_dot_product_attention`.
    """
    if num_frames <= 0:
        raise ValueError(f'num_frames must be positive, got {num_frames}.')
    if num_patches <= 0:
        raise ValueError(f'num_patches must be positive, got {num_patches}.')

    # 1. Label every frame-major token with the frame it belongs to.
    token_frames = torch.arange(num_frames).repeat_interleave(num_patches)

    # 2. Permit keys from the query's own frame and all preceding frames.
    mask = token_frames[:, None] >= token_frames[None, :]

    # 3. Add singleton batch/head axes for attention-kernel broadcasting.
    return mask.unsqueeze(0).unsqueeze(0)
