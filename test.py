import argparse
import copy
import csv
import os
from os import PathLike
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

import datasets
import models
from metrics.sr_metrics import calculate_psnr, calculate_ssim

# Keep the same full-raster setting used by the current NTAR-GS evaluation path.
torch.backends.cudnn.enabled = False
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:2048')


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _scale_modify_from_batch(batch, b, inp, gt=None):
    if 'scale_modify' in batch and torch.is_tensor(batch['scale_modify']):
        sm = batch['scale_modify'][b]
        return sm.to(device=inp.device, dtype=inp.dtype)
    if gt is not None:
        return torch.tensor(
            [gt.shape[-2] / inp.shape[-2], gt.shape[-1] / inp.shape[-1]],
            device=inp.device,
            dtype=inp.dtype,
        )
    if 'scale' in batch:
        s = batch['scale']
        if torch.is_tensor(s):
            sv = float(s.flatten()[0].item())
        else:
            sv = float(s)
        return torch.tensor([sv, sv], device=inp.device, dtype=inp.dtype)
    return torch.ones(2, device=inp.device, dtype=inp.dtype)


def _gt_size_from_batch(batch, b, inp, gt=None, scale_modify=None):
    if 'gt_size' in batch and torch.is_tensor(batch['gt_size']):
        return batch['gt_size'][b].to(device=inp.device)
    if gt is not None:
        return torch.tensor(
            [gt.shape[-2], gt.shape[-1]],
            device=inp.device,
            dtype=torch.long,
        )
    if scale_modify is None:
        scale_modify = _scale_modify_from_batch(batch, b, inp, gt=None)
    h = int(round(inp.shape[-2] * float(scale_modify[0].item())))
    w = int(round(inp.shape[-1] * float(scale_modify[1].item())))
    return torch.tensor([h, w], device=inp.device, dtype=torch.long)


@torch.no_grad()
def forward_full(model, inp, batch, b=0, gt=None):
    sm = _scale_modify_from_batch(batch, b, inp, gt=gt).view(1, 2)
    gt_size = _gt_size_from_batch(
        batch, b, inp, gt=gt, scale_modify=sm[0]
    ).view(1, 2)
    scale = sm.mean(dim=1)
    return model(inp, scale, gt_size=gt_size, scale_modify=sm)


def build_fixed_scale_loader(base_spec, scale):
    """Build deterministic loader for either online-downsampled or paired data."""
    spec = copy.deepcopy(base_spec)

    wrapper = spec.setdefault('wrapper', {})
    wrapper_name = wrapper.get('name', '')
    wrapper_args = wrapper.setdefault('args', {})

    if wrapper_name == 'sr-implicit-downsampled':
        # 原来的在线生成 LR 模式
        wrapper_args['scale_min'] = float(scale)
        wrapper_args['scale_max'] = float(scale)

        wrapper_args['inp_size'] = None
        wrapper_args['augment'] = False
        wrapper_args['batch_per_gpu'] = 1
        wrapper_args['random_crop'] = False
        wrapper_args['round_mode'] = 'round'

    elif wrapper_name == 'sr-implicit-paired':
        # 预生成 LR/GT：绝对不能再设置 scale_min / scale_max
        wrapper_args['inp_size'] = None
        wrapper_args['augment'] = False

        # 防止旧配置残留不支持的参数
        for key in (
            'scale_min',
            'scale_max',
            'batch_per_gpu',
            'random_crop',
            'round_mode',
        ):
            wrapper_args.pop(key, None)

    else:
        raise ValueError(
            f'Unsupported test wrapper: {wrapper_name}'
        )

    spec['batch_size'] = 1
    spec['num_workers'] = int(spec.get('num_workers', 0))

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(
        spec['wrapper'],
        args={'dataset': dataset}
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=spec['num_workers'],
        pin_memory=True,
        shuffle=False,
    )

    return loader



def _scale_tag(scale):
    value = float(scale)
    if abs(value - round(value)) < 1e-9:
        body = str(int(round(value)))
    else:
        body = ("%g" % value).replace(".", "_")
    return f"x{body}"


def get_dataset_filenames(dataset):
    """
    Recover the exact reference-image filenames used for SR saving.

    For paired-image-folders:
        dataset_1 = LR
        dataset_2 = GT/HR
    Therefore SR naming follows dataset_2 (GT/HR), including the original
    filename extension.

    For a single image-folder dataset, use its own filenames.
    """
    base = dataset
    visited = set()

    # Unwrap sr-implicit-* wrappers until reaching the underlying dataset.
    while hasattr(base, "dataset") and id(base) not in visited:
        visited.add(id(base))
        base = base.dataset

    # paired-image-folders: use GT/HR names, NOT LR names.
    dataset_2 = getattr(base, "dataset_2", None)
    if dataset_2 is not None and hasattr(dataset_2, "files"):
        files = getattr(dataset_2, "files")
        if all(isinstance(f, (str, PathLike)) for f in files):
            return [Path(f).name for f in files]

    # Fallback for ordinary image-folder / online-downsampled evaluation.
    if hasattr(base, "files"):
        files = getattr(base, "files")
        if all(isinstance(f, (str, PathLike)) for f in files):
            return [Path(f).name for f in files]

    return None


@torch.no_grad()
def evaluate_one_scale(
    loader, model, data_norm, metric_cfg, nominal_scale,
    scale_output_dir=None, filenames=None, max_save=None,
):
    model.eval()
    device = next(model.parameters()).device

    data_norm = data_norm or {
        "inp": {"sub": [0.0], "div": [1.0]},
        "gt": {"sub": [0.0], "div": [1.0]},
    }

    inp_sub = torch.tensor(
        data_norm["inp"]["sub"], dtype=torch.float32, device=device
    ).view(1, -1, 1, 1)
    inp_div = torch.tensor(
        data_norm["inp"]["div"], dtype=torch.float32, device=device
    ).view(1, -1, 1, 1)
    gt_sub = torch.tensor(
        data_norm["gt"]["sub"], dtype=torch.float32, device=device
    ).view(1, -1, 1, 1)
    gt_div = torch.tensor(
        data_norm["gt"]["div"], dtype=torch.float32, device=device
    ).view(1, -1, 1, 1)

    mod_scale = int(metric_cfg.get("mod_scale", 4))
    crop = bool(metric_cfg.get("crop", False))
    quantize = bool(metric_cfg.get("quantize", True))

    psnr_sum = 0.0
    ssim_sum = 0.0
    realized_scale_sum = 0.0
    count = 0
    saved = 0
    per_image_rows = []

    # ---------------------------------------------------------
    # Output directories
    # ---------------------------------------------------------
    scale_output_dir = Path(scale_output_dir)
    scale_output_dir.mkdir(parents=True, exist_ok=True)

    lr_dir = scale_output_dir / "LR"
    sr_dir = scale_output_dir / "SR"
    gt_dir = scale_output_dir / "GT"

    if max_save != 0:
        lr_dir.mkdir(parents=True, exist_ok=True)
        sr_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(loader, leave=False, desc=f"x{nominal_scale:g}")

    for batch in pbar:
        # -----------------------------------------------------
        # Move tensors to GPU
        # -----------------------------------------------------
        for key, value in batch.items():
            if torch.is_tensor(value):
                batch[key] = value.to(device, non_blocking=True)

        # -----------------------------------------------------
        # Keep the ORIGINAL LR for saving.
        #
        # lr_all:
        #     actual [0, 1] LR image used for visualization
        #
        # inp_all:
        #     normalized input used by the network
        # -----------------------------------------------------
        lr_all = batch["inp"].clamp(0.0, 1.0)

        inp_all = (batch["inp"] - inp_sub) / inp_div

        gt_all = batch["gt"].clamp(0.0, 1.0)

        # -----------------------------------------------------
        # Forward
        # -----------------------------------------------------
        preds = []

        for b in range(inp_all.shape[0]):
            pred_b = forward_full(
                model,
                inp_all[b:b + 1],
                batch,
                b=b,
                gt=gt_all[b:b + 1],
            )
            preds.append(pred_b)

        pred = torch.cat(preds, dim=0)

        # -----------------------------------------------------
        # De-normalize SR
        # -----------------------------------------------------
        pred = (pred * gt_div + gt_sub).clamp(0.0, 1.0)

        # Ensure SR and GT spatial sizes are identical
        pred = pred[
            ...,
            :gt_all.shape[-2],
            :gt_all.shape[-1]
        ]

        # -----------------------------------------------------
        # Per-image evaluation
        # -----------------------------------------------------
        for b in range(pred.shape[0]):

            actual_scale = _scale_modify_from_batch(
                batch,
                b,
                inp_all[b:b + 1],
                gt=gt_all[b:b + 1],
            )

            scale_h = float(actual_scale[0].item())
            scale_w = float(actual_scale[1].item())
            realized_scale = float(actual_scale.mean().item())

            # -------------------------------------------------
            # PSNR-Y
            # -------------------------------------------------
            psnr_y = float(
                calculate_psnr(
                    pred[b:b + 1],
                    gt_all[b:b + 1],
                    color_space="y",
                    mod_scale=mod_scale,
                    crop=crop,
                    scale=actual_scale,
                    quantize=quantize,
                ).item()
            )

            # -------------------------------------------------
            # SSIM-Y
            # -------------------------------------------------
            ssim_y = float(
                calculate_ssim(
                    pred[b:b + 1],
                    gt_all[b:b + 1],
                    color_space="y",
                    mod_scale=mod_scale,
                    crop=crop,
                    scale=actual_scale,
                    quantize=quantize,
                )
            )

            idx = count

            if filenames is not None and idx < len(filenames):
                original_filename = filenames[idx]
                name = Path(original_filename).stem
            else:
                name = f"{idx:05d}"

            # 统一无损保存为 PNG，避免 JPG 二次压缩影响评价
            filename = f"{name}.png"

            # -------------------------------------------------
            # Record image-level metrics
            # -------------------------------------------------
            per_image_rows.append({
                "name": name,
                "nominal_scale": float(nominal_scale),
                "realized_scale_h": scale_h,
                "realized_scale_w": scale_w,
                "realized_scale_mean": realized_scale,
                "psnr_y": psnr_y,
                "ssim_y": ssim_y,
            })

            # -------------------------------------------------
            # Save LR / SR / GT
            # -------------------------------------------------
            if (
                max_save != 0
                and (max_save is None or saved < max_save)
            ):
                save_image(
                    lr_all[b],
                    lr_dir / filename,
                )

                save_image(
                    pred[b],
                    sr_dir / filename,
                )

                save_image(
                    gt_all[b],
                    gt_dir / filename,
                )

                saved += 1

            # -------------------------------------------------
            # Statistics
            # -------------------------------------------------
            psnr_sum += psnr_y
            ssim_sum += ssim_y
            realized_scale_sum += realized_scale
            count += 1

        if count:
            pbar.set_postfix(
                psnr_y=f"{psnr_sum / count:.4f}",
                ssim_y=f"{ssim_sum / count:.4f}",
            )

    if count == 0:
        raise RuntimeError(
            f"No test images were produced at scale {nominal_scale}."
        )

    # ---------------------------------------------------------
    # Scale summary
    # ---------------------------------------------------------
    result = {
        "scale": float(nominal_scale),
        "realized_scale_mean": realized_scale_sum / count,
        "num_images": int(count),
        "psnr_y": psnr_sum / count,
        "ssim_y": ssim_sum / count,
    }

    # ---------------------------------------------------------
    # Per-image CSV
    # ---------------------------------------------------------
    with (
        scale_output_dir / "per_image_metrics.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "nominal_scale",
                "realized_scale_h",
                "realized_scale_w",
                "realized_scale_mean",
                "psnr_y",
                "ssim_y",
            ],
        )

        writer.writeheader()
        writer.writerows(per_image_rows)

    # ---------------------------------------------------------
    # Scale summary CSV
    # ---------------------------------------------------------
    with (
        scale_output_dir / "scale_summary.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scale",
                "realized_scale_mean",
                "num_images",
                "psnr_y",
                "ssim_y",
            ],
        )

        writer.writeheader()
        writer.writerow(result)

    return result


def main():
    parser = argparse.ArgumentParser(
        description='One-shot fixed multi-scale evaluation for NTAR-GS ablations.'
    )
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--output', default='./multiscale_results')
    parser.add_argument('--max_save', type=int, default=None,
                        help='Max SR images saved per scale; default all, 0 disables saving.')
    parser.add_argument(
        '--scales',
        nargs='*',
        type=float,
        default=None,
        help='Optional override, e.g. --scales 4  8 ',
    )
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    seed = int(config.get('seed', 42))
    set_seed(seed)

    config_scales = config.get('multi_scale', {}).get(
        'scales', [ 4.0, 8.0]
    )
    scales = args.scales if args.scales else config_scales
    scales = [float(s) for s in scales]

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required by the current NTAR-GS/gsplat evaluation path.')

    checkpoint = torch.load(args.model, map_location='cpu')
    if 'model' not in checkpoint:
        raise KeyError('Checkpoint does not contain the expected "model" entry.')
    model_spec = checkpoint['model']
    model = models.make(model_spec, load_sd=True).cuda().eval()

    base_spec = config.get('test_dataset', config.get('val_dataset'))
    if base_spec is None:
        raise KeyError('Config must contain test_dataset (or val_dataset as fallback).')

    metric_cfg = dict(config.get('test_metrics', {}) or {})

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    print('============================================================')
    print('NTAR-GS fixed multi-scale evaluation')
    print('checkpoint:', args.model)
    print('scales    :', scales)
    print('============================================================')

    for scale in scales:
        # Reset seed before every deterministic fixed-scale build.
        set_seed(seed)
        loader = build_fixed_scale_loader(base_spec, scale)
        filenames = get_dataset_filenames(loader.dataset)

        if filenames is None:
            print(
                "[warning] Could not recover GT/HR filenames; "
                "SR files will use numeric fallback names."
            )
        else:
            print(
                f"[info] recovered {len(filenames)} GT/HR filenames; "
                "SR files will keep exactly the same names."
            )

        scale_output_dir = output_dir / _scale_tag(scale)
        result = evaluate_one_scale(
            loader,
            model,
            data_norm=config.get('data_norm'),
            metric_cfg=metric_cfg,
            nominal_scale=scale,
            scale_output_dir=scale_output_dir,
            filenames=filenames,
            max_save=args.max_save,
        )
        rows.append(result)
        print(
            'x{scale:g}: PSNR-Y={psnr_y:.6f}, SSIM-Y={ssim_y:.6f}, '
            'realized_scale_mean={realized_scale_mean:.6f}, images={num_images}'.format(
                **result
            )
        )

    csv_path = output_dir / 'multiscale_summary.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'scale',
                'realized_scale_mean',
                'num_images',
                'psnr_y',
                'ssim_y',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print('')
    print('================ Multi-scale summary ================')
    print(' scale |   PSNR-Y |   SSIM-Y | realized scale')
    for row in rows:
        print(
            ' x{:>3g} | {:>8.5f} | {:>8.6f} | {:>14.6f}'.format(
                row['scale'],
                row['psnr_y'],
                row['ssim_y'],
                row['realized_scale_mean'],
            )
        )
    print('CSV:', csv_path)
    print('=====================================================')


if __name__ == '__main__':
    main()
