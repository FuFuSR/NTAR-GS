"""Single-image NTAR-GS super-resolution inference .

Example:
    python inference.py \
        --input examples/input/intersection_346.png \
        --model NTAR-GS.pth \
        --scale 4 \
        --output examples/output/intersection_346_x4.png

"""

import argparse
import math
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--scale', type=float, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    if not math.isfinite(args.scale) or args.scale <= 0:
        parser.error('--scale must be a positive finite number')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:2048')

    import torch
    import models
    from PIL import Image, ImageOps
    from torchvision.transforms.functional import to_tensor, to_pil_image

    torch.backends.cudnn.enabled = False
    device = torch.device('cuda')

    checkpoint = torch.load(args.model, map_location='cpu')
    model = models.make(checkpoint['model'], load_sd=True).to(device).eval()

    with Image.open(args.input) as image:
        inp = to_tensor(ImageOps.exif_transpose(image).convert('RGB'))
    inp = inp.unsqueeze(0).to(device)

    h, w = inp.shape[-2:]
    out_h = max(1, round(h * args.scale))
    out_w = max(1, round(w * args.scale))
    gt_size = torch.tensor([[out_h, out_w]], device=device)
    scale_modify = inp.new_tensor([[out_h / h, out_w / w]])
    scale = scale_modify.mean(dim=1)

    with torch.no_grad():
        sr = model(inp * 2 - 1, scale, gt_size=gt_size, scale_modify=scale_modify)
        sr = ((sr + 1) / 2).clamp(0, 1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(sr[0].cpu()).save(args.output, format='PNG')
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
