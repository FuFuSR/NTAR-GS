import argparse
import copy
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
from metrics.sr_metrics import calculate_psnr, calculate_ssim


torch.backends.cudnn.enabled = False
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:2048')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset_spec, scale):
    spec = copy.deepcopy(dataset_spec)
    wrapper = spec['wrapper']
    args = wrapper.setdefault('args', {})

    if wrapper['name'] == 'sr-implicit-downsampled':
        args.update(
            inp_size=None, scale_min=scale, scale_max=scale,
            augment=False, batch_per_gpu=1, random_crop=False,
            round_mode='round',
        )
    elif wrapper['name'] == 'sr-implicit-paired':
        args['inp_size'] = None
        args['augment'] = False
        for key in ('scale_min', 'scale_max', 'batch_per_gpu',
                    'random_crop', 'round_mode'):
            args.pop(key, None)
    else:
        raise ValueError(f"Unsupported test wrapper: {wrapper['name']}")

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(wrapper, args={'dataset': dataset})
    return DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=spec.get('num_workers', 0), pin_memory=True,
    )


def norm_params(spec, device):
    sub = torch.tensor(spec['sub'], dtype=torch.float32, device=device).view(1, -1, 1, 1)
    div = torch.tensor(spec['div'], dtype=torch.float32, device=device).view(1, -1, 1, 1)
    return sub, div


@torch.no_grad()
def evaluate(loader, model, data_norm, scale):
    model.eval()
    device = next(model.parameters()).device
    inp_sub, inp_div = norm_params(data_norm['inp'], device)
    gt_sub, gt_div = norm_params(data_norm['gt'], device)

    psnr_sum = 0.0
    ssim_sum = 0.0
    count = 0

    for batch in tqdm(loader, desc=f'x{scale:g}', leave=False):
        batch = {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        inp = (batch['inp'] - inp_sub) / inp_div
        gt = batch['gt'].clamp(0.0, 1.0)
        h, w = gt.shape[-2:]

        scale_modify = batch.get('scale_modify')
        if scale_modify is None:
            scale_modify = torch.tensor(
                [[h / inp.shape[-2], w / inp.shape[-1]]],
                device=device, dtype=inp.dtype,
            )
        else:
            scale_modify = scale_modify.to(device=device, dtype=inp.dtype).reshape(1, 2)
        gt_size = batch.get('gt_size')
        if gt_size is None:
            gt_size = torch.tensor([[h, w]], device=device, dtype=torch.long)
        else:
            gt_size = gt_size.to(device=device, dtype=torch.long).reshape(1, 2)

        pred = model(
            inp, scale_modify.mean(dim=1),
            gt_size=gt_size, scale_modify=scale_modify,
        )
        pred = (pred * gt_div + gt_sub).clamp(0.0, 1.0)[..., :h, :w]

        psnr = calculate_psnr(
            pred, gt, color_space='y', mod_scale=4,
            crop=True, scale=scale_modify[0], quantize=True,
        )
        ssim = calculate_ssim(
            pred, gt, color_space='y', mod_scale=4,
            crop=True, scale=scale_modify[0], quantize=True,
        )
        psnr_sum += float(psnr)
        ssim_sum += float(ssim)
        count += 1

    if count == 0:
        raise RuntimeError('Test dataset is empty.')
    return psnr_sum / count, ssim_sum / count, count


def main():
    parser = argparse.ArgumentParser(description='NTAR-GS PSNR-Y / SSIM-Y evaluation')
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--scales', nargs='+', type=float, default=None)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for NTAR-GS evaluation.')

    checkpoint = torch.load(args.model, map_location='cpu')
    model = models.make(checkpoint['model'], load_sd=True).cuda().eval()

    dataset_spec = config['test_dataset']
    data_norm = config['data_norm']
    scales = args.scales or config.get('multi_scale', {}).get('scales', [4.0])
    seed = int(config.get('seed', 42))

    for scale in scales:
        set_seed(seed)
        loader = make_loader(dataset_spec, float(scale))
        psnr, ssim, count = evaluate(
            loader, model, data_norm, float(scale),
        )
        print(f'x{scale:g} | images: {count} | PSNR-Y: {psnr:.4f} | SSIM-Y: {ssim:.4f}')


if __name__ == '__main__':
    main()
