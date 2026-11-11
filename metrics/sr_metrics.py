from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import cv2
import numpy as np
import torch

ScaleLike = Union[float, int, Sequence[float], torch.Tensor]



def _check_image_tensor(x: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(x):
        raise TypeError(f'{name} must be a torch.Tensor, got {type(x).__name__}')
    if x.ndim != 4:
        raise ValueError(f'{name} must have shape [B,C,H,W], got {tuple(x.shape)}')
    if x.shape[1] != 3:
        raise ValueError(f'{name} must be RGB with C=3, got C={x.shape[1]}')


def _check_pair(sr: torch.Tensor, gt: torch.Tensor) -> None:
    _check_image_tensor(sr, 'sr')
    _check_image_tensor(gt, 'gt')
    if sr.shape != gt.shape:
        raise ValueError(f'sr and gt must have the same shape, got {tuple(sr.shape)} and {tuple(gt.shape)}')


def _quantize_8bit(x: torch.Tensor) -> torch.Tensor:
    return torch.round(x.clamp(0.0, 1.0) * 255.0) / 255.0


def _prepare_rgb(sr: torch.Tensor, gt: Optional[torch.Tensor] = None, quantize: bool = True):
    sr = sr.clamp(0.0, 1.0)
    if quantize:
        sr = _quantize_8bit(sr)
    if gt is None:
        return sr
    gt = gt.clamp(0.0, 1.0)
    if quantize:
        gt = _quantize_8bit(gt)
    return sr, gt


def _modcrop_pair(sr: torch.Tensor, gt: torch.Tensor, scale: int):
    if scale <= 0:
        raise ValueError(f'mod_scale must be positive, got {scale}')
    h = min(sr.shape[-2], gt.shape[-2])
    w = min(sr.shape[-1], gt.shape[-1])
    h = (h // scale) * scale
    w = (w // scale) * scale
    if h <= 0 or w <= 0:
        raise ValueError(f'Image is too small for modcrop scale={scale}: {(sr.shape[-2], sr.shape[-1])}')
    return sr[..., :h, :w], gt[..., :h, :w]


def _crop_pair(sr: torch.Tensor, gt: torch.Tensor, border: int):
    border = int(border)
    if border < 0:
        raise ValueError(f'crop_border must be >= 0, got {border}')
    if border == 0:
        return sr, gt
    if 2 * border >= sr.shape[-2] or 2 * border >= sr.shape[-1]:
        raise ValueError(f'crop_border={border} is too large for image size {tuple(sr.shape[-2:])}')
    return sr[..., border:-border, border:-border], gt[..., border:-border, border:-border]


def _rgb_to_matlab_y_255(x: torch.Tensor) -> torch.Tensor:
    """Convert RGB [0,1] to MATLAB-style Y in the nominal [16,235] range."""
    rgb255 = x * 255.0
    r = rgb255[:, 0:1]
    g = rgb255[:, 1:2]
    b = rgb255[:, 2:3]
    return (65.481 * r + 128.553 * g + 24.966 * b) / 255.0 + 16.0


def _normalize_color_space(color_space: str) -> str:
    value = str(color_space).strip().lower()
    if value not in {'y', 'rgb'}:
        raise ValueError(f"color_space must be 'y' or 'rgb', got {color_space!r}")
    return value


def _scale_values(scale: Optional[ScaleLike], batch_size: int) -> list[float]:

    if scale is None:
        raise ValueError('scale is required when crop=True')

    if torch.is_tensor(scale):
        values = scale.detach().float().cpu()
        if values.ndim == 0:
            return [float(values.item())] * batch_size
        if values.ndim == 1:
            if batch_size == 1:
                return [float(values.max().item())]
            if values.numel() == batch_size:
                return [float(v) for v in values.tolist()]
            if values.numel() == 1:
                return [float(values.item())] * batch_size
            raise ValueError(
                f'Cannot map scale tensor of shape {tuple(values.shape)} to batch size {batch_size}'
            )
        if values.shape[0] == batch_size:
            return [float(row.max().item()) for row in values.reshape(batch_size, -1)]
        if values.numel() == 1:
            return [float(values.item())] * batch_size
        raise ValueError(
            f'Cannot map scale tensor of shape {tuple(values.shape)} to batch size {batch_size}'
        )

    if isinstance(scale, Sequence) and not isinstance(scale, (str, bytes)):
        values = [float(v) for v in scale]
        if batch_size == 1:
            return [max(values)]
        if len(values) == batch_size:
            return values
        if len(values) == 1:
            return values * batch_size
        raise ValueError(f'Cannot map {len(values)} scale values to batch size {batch_size}')

    return [float(scale)] * batch_size


def resolve_crop_border(crop: bool, scale: Optional[ScaleLike], batch_size: int = 1) -> list[int]:

    if not bool(crop):
        return [0] * int(batch_size)

    borders = []
    for value in _scale_values(scale, int(batch_size)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'scale must be a finite positive value, got {value}')
        nearest = round(value)
        if math.isclose(value, nearest, rel_tol=0.0, abs_tol=1e-6):
            border = int(nearest)
        else:
            border = int(math.ceil(value))
        borders.append(max(1, border))
    return borders


def _paired_metric_images(
    sr: torch.Tensor,
    gt: torch.Tensor,
    mod_scale: int,
    crop: bool,
    scale: Optional[ScaleLike],
    quantize: bool,
):
    _check_pair(sr, gt)
    sr, gt = _prepare_rgb(sr, gt, quantize=quantize)
    sr, gt = _modcrop_pair(sr, gt, int(mod_scale))
    borders = resolve_crop_border(crop, scale, batch_size=sr.shape[0])
    return [
        _crop_pair(sr[i:i + 1], gt[i:i + 1], border)
        for i, border in enumerate(borders)
    ]


def calculate_psnr(
    sr: torch.Tensor,
    gt: torch.Tensor,
    color_space: str = 'y',
    mod_scale: int = 4,
    crop: bool = False,
    scale: Optional[ScaleLike] = None,
    quantize: bool = True,
) -> torch.Tensor:
    """Image-wise mean PSNR in MATLAB-style Y or RGB space."""
    color_space = _normalize_color_space(color_space)
    values = []
    for sr_i, gt_i in _paired_metric_images(sr, gt, mod_scale, crop, scale, quantize):
        if color_space == 'y':
            eval_sr = _rgb_to_matlab_y_255(sr_i)
            eval_gt = _rgb_to_matlab_y_255(gt_i)
        else:
            eval_sr = sr_i * 255.0
            eval_gt = gt_i * 255.0
        mse = (eval_sr - eval_gt).pow(2).mean()
        if mse.item() == 0.0:
            values.append(torch.full((), float('inf'), device=mse.device, dtype=mse.dtype))
        else:
            values.append(20.0 * torch.log10(255.0 / torch.sqrt(mse)))
    return torch.stack(values).mean()


def _ssim_2d(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.shape != img2.shape:
        raise ValueError(f'SSIM inputs must have identical shapes, got {img1.shape} and {img2.shape}')
    if img1.ndim != 2:
        raise ValueError(f'SSIM expects a 2-D channel image, got ndim={img1.ndim}')
    if min(img1.shape) < 11:
        raise ValueError(f'SSIM requires both image dimensions >= 11, got {img1.shape}')

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean())


def calculate_ssim(
    sr: torch.Tensor,
    gt: torch.Tensor,
    color_space: str = 'y',
    mod_scale: int = 4,
    crop: bool = False,
    scale: Optional[ScaleLike] = None,
    quantize: bool = True,
) -> float:

    color_space = _normalize_color_space(color_space)
    values = []
    for sr_i, gt_i in _paired_metric_images(sr, gt, mod_scale, crop, scale, quantize):
        if color_space == 'y':
            eval_sr = _rgb_to_matlab_y_255(sr_i).detach().cpu().numpy()[0, 0]
            eval_gt = _rgb_to_matlab_y_255(gt_i).detach().cpu().numpy()[0, 0]
            values.append(_ssim_2d(eval_sr, eval_gt))
        else:
            eval_sr = (sr_i * 255.0).detach().cpu().numpy()[0]
            eval_gt = (gt_i * 255.0).detach().cpu().numpy()[0]
            values.append(float(np.mean([
                _ssim_2d(eval_sr[channel], eval_gt[channel])
                for channel in range(3)
            ])))
    return float(np.mean(values))
