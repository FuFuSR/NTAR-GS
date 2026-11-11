# -*- coding: utf-8 -*-
"""Pure SR-difficulty-aware patch sampling for NTAR-GS.

Training-only helper. For each crop request, K uniformly located candidate HR
patches are sampled. Each candidate is degraded to the LR size used by the
current training scale, bicubic-upsampled back to HR size, and scored by the
mean absolute reconstruction error.

No topology feature and no NTAR-GS model output is used here. Therefore this
module isolates the effect of SR reconstruction difficulty itself.
"""

import random
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


DIFFICULTY_NUM_CANDIDATES = 8
DIFFICULTY_TEMPERATURE = 0.50


@torch.no_grad()
def bicubic_reconstruction_difficulty(
    patches: torch.Tensor,
    lr_h: int,
    lr_w: int,
) -> torch.Tensor:
    """Return one bicubic SR reconstruction-difficulty score per HR patch.

    Args:
        patches: Tensor [K, C, H, W], normally in [0, 1].
        lr_h/lr_w: LR size used by the current training sample.

    Score:
        mean(abs(HR - BicubicUp(BicubicDown(HR))))

    Larger = more information is lost by bicubic degradation = harder SR patch.
    """
    if patches.dim() != 4:
        raise ValueError(f'patches must be [K,C,H,W], got {tuple(patches.shape)}')

    lr_h = max(1, int(lr_h))
    lr_w = max(1, int(lr_w))
    hr_h, hr_w = int(patches.shape[-2]), int(patches.shape[-1])

    x = patches.float()

    # At exact x1 there is no SR degradation, so all candidates should become
    # equivalent and the subsequent softmax becomes uniform.
    if lr_h == hr_h and lr_w == hr_w:
        return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

    lr = F.interpolate(
        x,
        size=(lr_h, lr_w),
        mode='bicubic',
        align_corners=False,
    )
    recon = F.interpolate(
        lr,
        size=(hr_h, hr_w),
        mode='bicubic',
        align_corners=False,
    )

    return (x - recon).abs().mean(dim=(1, 2, 3))


def _minmax(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    lo = v.min()
    hi = v.max()
    if float((hi - lo).abs()) < eps:
        return torch.zeros_like(v)
    return (v - lo) / (hi - lo + eps)


@torch.no_grad()
def choose_difficulty_crop_origin(
    img: torch.Tensor,
    crop_h: int,
    crop_w: int,
    lr_h: int,
    lr_w: int,
    num_candidates: int = DIFFICULTY_NUM_CANDIDATES,
    temperature: float = DIFFICULTY_TEMPERATURE,
) -> Tuple[int, int, Dict[str, float]]:
    """Choose one HR crop using pure SR reconstruction difficulty.

    Procedure:
      1. Uniformly sample K valid crop origins.
      2. Extract K HR candidate patches.
      3. Compute current-scale bicubic down-up reconstruction difficulty.
      4. Min-max normalize scores inside this K-candidate set.
      5. Sample stochastically with softmax(score / temperature).

    This is importance/informative patch sampling. It does not modify image
    content and it does not query the NTAR-GS model.
    """
    if img.dim() != 3:
        raise ValueError(f'img must be [C,H,W], got {tuple(img.shape)}')

    crop_h = int(crop_h)
    crop_w = int(crop_w)
    lr_h = int(lr_h)
    lr_w = int(lr_w)

    h, w = int(img.shape[-2]), int(img.shape[-1])
    if crop_h > h or crop_w > w:
        raise ValueError(
            f'crop ({crop_h},{crop_w}) is larger than image ({h},{w})'
        )

    k = max(1, int(num_candidates))
    max_x = h - crop_h
    max_y = w - crop_w

    origins = [
        (
            random.randint(0, max_x) if max_x > 0 else 0,
            random.randint(0, max_y) if max_y > 0 else 0,
        )
        for _ in range(k)
    ]

    patches = torch.stack([
        img[:, x0:x0 + crop_h, y0:y0 + crop_w]
        for x0, y0 in origins
    ], dim=0)

    difficulty = bicubic_reconstruction_difficulty(
        patches,
        lr_h=lr_h,
        lr_w=lr_w,
    )
    score = _minmax(difficulty)

    temp = max(float(temperature), 1e-4)
    prob = torch.softmax(score / temp, dim=0)
    chosen = int(torch.multinomial(prob, num_samples=1).item())
    x0, y0 = origins[chosen]

    meta = {
        'difficulty_raw': float(difficulty[chosen].item()),
        'difficulty_norm': float(score[chosen].item()),
        'prob': float(prob[chosen].item()),
        'num_candidates': float(k),
    }
    return int(x0), int(y0), meta
