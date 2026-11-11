import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from datasets import register
from utils import to_pixel_samples


def resize_fn(img, size):
    """Bicubic resize tensor/PIL image to (h, w), returning tensor in [0,1]."""
    if not isinstance(img, Image.Image):
        img = transforms.ToPILImage()(img)
    img = transforms.Resize(size, InterpolationMode.BICUBIC)(img)
    return transforms.ToTensor()(img)


def _scale_modify_from_pair(lr, hr):
    """Return true H/W scale after round/crop: [hr_h/lr_h, hr_w/lr_w]."""
    h_lr, w_lr = lr.shape[-2:]
    h_hr, w_hr = hr.shape[-2:]
    return torch.tensor([float(h_hr) / float(h_lr), float(w_hr) / float(w_lr)], dtype=torch.float32)


def _gt_size_tensor(hr):
    return torch.tensor([int(hr.shape[-2]), int(hr.shape[-1])], dtype=torch.long)


@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):
    """Paired LR/HR wrapper with STAG-GSASR scale_modify metadata."""

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        if self.inp_size is None:
            # 对预生成 paired LR/GT，完整保留二者尺寸。
            # 不使用整数倍率，不重新裁 GT。
            crop_lr = img_lr
            crop_hr = img_hr
        else:
            w_lr = int(self.inp_size)
            h_lr = int(self.inp_size)
            x0 = random.randint(0, img_lr.shape[-2] - h_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + h_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            h_hr = h_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + h_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        scale_modify = _scale_modify_from_pair(crop_lr, crop_hr)
        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': crop_hr,
            'scale': float(scale_modify.mean().item()),
            'gt_size': _gt_size_tensor(crop_hr),
            'scale_modify': scale_modify,
        }


@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):
    """Online bicubic downsample wrapper for direct-splat arbitrary scale.

    Compared with the old STAG wrapper, this returns two extra keys:
      - gt_size: exact rendered target size [H_gt, W_gt]
      - scale_modify: exact realized scale [H_gt/H_lr, W_gt/W_lr]

    This is the key GSASR-style fix: the renderer no longer infers the target
    size only from a scalar scale, which avoids round/floor mismatch under 1-4x.
    """

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None, batch_per_gpu=16,
                 random_crop=True, round_mode='round'):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = float(scale_min)
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = float(scale_max)
        self.augment = augment
        self.last_s = random.uniform(self.scale_min, self.scale_max)
        self.batch_per_gpu = int(batch_per_gpu)
        self.call_count = -1
        self.random_crop = random_crop
        self.round_mode = round_mode

    def __len__(self):
        return len(self.dataset)

    def _round_size(self, x):
        if self.round_mode == 'ceil':
            return int(math.ceil(x))
        if self.round_mode == 'floor':
            return int(math.floor(x + 1e-9))
        return int(round(x))

    def __getitem__(self, idx):
        self.call_count += 1
        # Keep one scale inside a batch so default PyTorch collation can stack
        # HR tensors.  This matches the original STAG/GaussianSR wrapper style.
        if self.call_count % self.batch_per_gpu == 0:
            s = random.uniform(self.scale_min, self.scale_max)
            self.last_s = s
        else:
            s = self.last_s

        img = self.dataset[idx]

        if self.inp_size is None:
            h_lr = int(math.floor(img.shape[-2] / s + 1e-9))
            w_lr = int(math.floor(img.shape[-1] / s + 1e-9))
            h_hr = self._round_size(h_lr * s)
            w_hr = self._round_size(w_lr * s)
            img = img[:, :h_hr, :w_hr]
            crop_lr = resize_fn(img, (h_lr, w_lr))
            crop_hr = img
        else:
            h_lr = int(self.inp_size)
            w_lr = int(self.inp_size)
            h_hr = self._round_size(h_lr * s)
            w_hr = self._round_size(w_lr * s)
            if h_hr > img.shape[-2] or w_hr > img.shape[-1]:
                raise ValueError(
                    f'HR crop ({h_hr}, {w_hr}) is larger than image '
                    f'{tuple(img.shape[-2:])}. Reduce inp_size or scale_max.'
                )
            if self.random_crop:
                x0 = random.randint(0, img.shape[-2] - h_hr)
                y0 = random.randint(0, img.shape[-1] - w_hr)
            else:
                x0 = (img.shape[-2] - h_hr) // 2
                y0 = (img.shape[-1] - w_hr) // 2
            crop_hr = img[:, x0: x0 + h_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, (h_lr, w_lr))

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        scale_modify = _scale_modify_from_pair(crop_lr, crop_hr)
        return {
            'inp': crop_lr,
            'gt': crop_hr,
            # Legacy scalar scale.  The model uses scale_modify preferentially.
            'scale': float(scale_modify.mean().item()),
            'gt_size': _gt_size_tensor(crop_hr),
            'scale_modify': scale_modify,
        }


@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        p = idx / (len(self.dataset) - 1)
        w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
        img_hr = resize_fn(img_hr, w_hr)

        if self.augment:
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)

        if self.gt_resize is not None:
            img_hr = resize_fn(img_hr, self.gt_resize)

        hr_coord, hr_rgb = to_pixel_samples(img_hr)

        if self.sample_q is not None:
            sample_lst = np.random.choice(len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / img_hr.shape[-2]
        cell[:, 1] *= 2 / img_hr.shape[-1]

        scale_modify = _scale_modify_from_pair(img_lr, img_hr)
        return {
            'inp': img_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'gt_size': _gt_size_tensor(img_hr),
            'scale_modify': scale_modify,
            'scale': float(scale_modify.mean().item()),
        }
