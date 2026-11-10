import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torchvision.utils import save_image
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils
from metrics.sr_metrics import calculate_psnr

try:
    import models.ar_gaussian_splatter  # noqa: F401
except ImportError:
    pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_duration(seconds):
    seconds = float(seconds)
    if seconds < 3600:
        return f'{seconds / 60.0:.1f}m'
    return f'{seconds / 3600.0:.1f}h'


def make_data_loader(spec, tag):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log(f'{tag} dataset: size={len(dataset)}')
    first = dataset[0]
    for k, v in first.items():
        if torch.is_tensor(v):
            log(f'  {k}: shape={tuple(v.shape)} dtype={v.dtype}')
        else:
            log(f'  {k}: type={type(v).__name__} value={v}')

    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        shuffle=(tag == 'train'),
        num_workers=spec.get('num_workers', 0),
        pin_memory=True,
    )
    return loader


def _load_checkpoint(path):
    """Load checkpoints containing optimizer/RNG metadata across PyTorch versions."""
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        # PyTorch < 2.6 has no weights_only argument.
        return torch.load(path, map_location='cpu')


def _build_scheduler(optimizer):
    """Build exactly one configured LR scheduler, or return None."""
    multi_step_cfg = config.get('multi_step_lr')
    cosine_cfg = config.get('cosine_lr')

    if multi_step_cfg is not None and cosine_cfg is not None:
        raise ValueError(
            'Configure only one LR scheduler: multi_step_lr or cosine_lr.'
        )

    if cosine_cfg is not None:
        return CosineAnnealingLR(
            optimizer,
            T_max=int(cosine_cfg.get('T_max', config.get('epoch_max', 300))),
            eta_min=float(cosine_cfg.get('eta_min', 1e-6)),
        )

    if multi_step_cfg is not None:
        return MultiStepLR(optimizer, **multi_step_cfg)

    return None


def _fast_forward_scheduler(optimizer, epoch_start):
    """Reconstruct scheduler position for legacy checkpoints without scheduler state.

    A checkpoint saved after epoch N resumes at epoch N+1.  The LR scheduler in
    this training loop is stepped once at the end of every completed epoch, so
    before starting epoch N+1 it must already have received N scheduler steps.
    """
    # When a scheduler was used previously, PyTorch normally leaves initial_lr
    # in each optimizer param group.  For older checkpoints without it, the
    # stored LR is the best available base LR (and is the original LR for the
    # legacy constant-LR checkpoints used by this project).
    for group in optimizer.param_groups:
        base_lr = float(group.get('initial_lr', group['lr']))
        group['lr'] = base_lr
        group['initial_lr'] = base_lr

    scheduler = _build_scheduler(optimizer)
    if scheduler is None:
        return None

    # Replaying scheduler steps is only for state reconstruction.  Suppress the
    # harmless "scheduler.step before optimizer.step" warning in this branch.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        for _ in range(max(0, int(epoch_start) - 1)):
            scheduler.step()
    return scheduler


def _capture_rng_state():
    """Capture RNG states needed for deterministic single-process resume."""
    state = {
        'python_rng_state': random.getstate(),
        'numpy_rng_state': np.random.get_state(),
        'torch_rng_state': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(ckpt):
    """Restore every RNG state available in a checkpoint.

    Returns True only when all CPU-side states and CUDA state (when CUDA is
    available) were present.  Legacy checkpoints remain fully usable; they just
    start a new random continuation from the resumed model/optimizer state.
    """
    restored = []
    missing = []

    if 'python_rng_state' in ckpt:
        random.setstate(ckpt['python_rng_state'])
        restored.append('python')
    else:
        missing.append('python')

    if 'numpy_rng_state' in ckpt:
        np.random.set_state(ckpt['numpy_rng_state'])
        restored.append('numpy')
    else:
        missing.append('numpy')

    if 'torch_rng_state' in ckpt:
        torch.set_rng_state(ckpt['torch_rng_state'])
        restored.append('torch')
    else:
        missing.append('torch')

    if torch.cuda.is_available():
        cuda_state = ckpt.get('cuda_rng_state_all', ckpt.get('cuda_rng_state'))
        if cuda_state is not None:
            try:
                if isinstance(cuda_state, (list, tuple)):
                    torch.cuda.set_rng_state_all(cuda_state)
                else:
                    torch.cuda.set_rng_state(cuda_state)
                restored.append('cuda')
            except Exception as exc:
                missing.append(f'cuda({exc})')
        else:
            missing.append('cuda')

    if restored:
        log('resume RNG restored: ' + ', '.join(restored))
    if missing:
        log(
            'resume RNG warning: checkpoint has no complete RNG state for '
            + ', '.join(missing)
            + '; continuation is valid but will not reproduce the old random '
              'trajectory bit-for-bit.'
        )

    return len(missing) == 0


def prepare_training():
    resume_ckpt = None

    if config.get('resume') is not None:
        resume_ckpt = _load_checkpoint(config['resume'])
        ckpt_model = resume_ckpt['model']
        model = models.make(ckpt_model, load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), resume_ckpt['optimizer'], load_sd=True
        )
        epoch_start = int(resume_ckpt['epoch']) + 1
    else:
        model = models.make(config['model']).cuda()

        # Optional architecture-expansion initialization.  This loads every
        # checkpoint tensor whose name and shape still match, then copies the
        # overlapping prefix of layer-sized vectors (for example release logits)
        # when expanding L1 to L2.  The optimizer is intentionally new, so this
        # is initialization rather than an exact resume.
        init_from = config.get('init_from')
        if init_from:
            ckpt = _load_checkpoint(init_from)
            source = ckpt.get('model', ckpt)
            source_sd = source.get('sd', source) if isinstance(source, dict) else source
            source_sd = {
                (k[7:] if k.startswith('module.') else k): v
                for k, v in source_sd.items()
            }
            target_sd = model.state_dict()
            compatible = {
                k: v for k, v in source_sd.items()
                if k in target_sd and tuple(v.shape) == tuple(target_sd[k].shape)
            }
            missing, unexpected = model.load_state_dict(compatible, strict=False)

            # Copy overlapping entries for vectors whose length grows with the
            # number of AR levels.  This preserves the learned G1 entry and keeps
            # the newly added G2 entry at its stable constructor initialization.
            expandable = [
                'ar_offset_layer_scale_raw',
                'ar_child_topology_release_logit',
                'ar_sibling_release_logit',
            ]
            with torch.no_grad():
                named_params = dict(model.named_parameters())
                for name in expandable:
                    if name not in source_sd or name not in named_params:
                        continue
                    src = source_sd[name].to(named_params[name].device)
                    dst = named_params[name]
                    if src.ndim == dst.ndim == 1:
                        count = min(src.numel(), dst.numel())
                        dst[:count].copy_(src[:count])

            log(
                'init_from: loaded {} exact tensors from {}; '
                '{} target tensors remain newly initialized'.format(
                    len(compatible), init_from, len(missing)
                )
            )
            if unexpected:
                log('init_from: ignored unexpected tensors={}'.format(len(unexpected)))

        optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
        epoch_start = 1

    # LR scheduler: prefer the exact saved scheduler state.  For legacy
    # checkpoints that predate scheduler-state saving, reconstruct the scheduler
    # position from the completed epoch count.
    if resume_ckpt is not None and resume_ckpt.get('scheduler') is not None:
        scheduler = _build_scheduler(optimizer)
        if scheduler is None:
            raise ValueError(
                'Checkpoint contains scheduler state, but the current YAML has '
                'no multi_step_lr/cosine_lr scheduler configured.'
            )
        scheduler.load_state_dict(resume_ckpt['scheduler'])
        log(
            'resume scheduler: loaded saved state at epoch {}, lr={}'.format(
                resume_ckpt['epoch'],
                ','.join(f'{g["lr"]:.8g}' for g in optimizer.param_groups),
            )
        )
    elif resume_ckpt is not None:
        scheduler = _fast_forward_scheduler(optimizer, epoch_start)
        if scheduler is not None:
            log(
                'resume scheduler: legacy checkpoint had no scheduler state; '
                'reconstructed to epoch {}, lr={}'.format(
                    resume_ckpt['epoch'],
                    ','.join(f'{g["lr"]:.8g}' for g in optimizer.param_groups),
                )
            )
    else:
        scheduler = _build_scheduler(optimizer)

    # Restore RNG only after model/optimizer/scheduler construction, because
    # constructors may themselves consume random numbers.  This makes future
    # checkpoints with saved RNG states resume from the correct next sample.
    if resume_ckpt is not None:
        _restore_rng_state(resume_ckpt)

    resume_meta = {
        'best_psnr': float(resume_ckpt.get('best_psnr', -1e18))
        if resume_ckpt is not None else -1e18,
        'bad_epochs': int(resume_ckpt.get('bad_epochs', 0))
        if resume_ckpt is not None else 0,
    }

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    if resume_ckpt is not None:
        log(
            'resume: checkpoint epoch={} -> next epoch={}'.format(
                resume_ckpt['epoch'], epoch_start
            )
        )
    return model, optimizer, epoch_start, scheduler, resume_meta


def normalize_img(x, norm, device):
    sub = torch.FloatTensor(norm['sub']).view(1, -1, 1, 1).to(device)
    div = torch.FloatTensor(norm['div']).view(1, -1, 1, 1).to(device)
    return (x - sub) / div


def denormalize_img(x, norm, device):
    sub = torch.FloatTensor(norm['sub']).view(1, -1, 1, 1).to(device)
    div = torch.FloatTensor(norm['div']).view(1, -1, 1, 1).to(device)
    return x * div + sub


def _normalize_per_image(x, eps=1e-6):
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def _gaussian_blur_5x5(x):
    """Small separable Gaussian blur used to turn sparse node targets into soft disks."""
    if x is None:
        return None
    dtype, device = x.dtype, x.device
    k = torch.tensor([1., 4., 6., 4., 1.], dtype=dtype, device=device)
    k = k / k.sum()
    ky = k.view(1, 1, 5, 1)
    kx = k.view(1, 1, 1, 5)
    c = x.shape[1]
    x = F.conv2d(x, ky.expand(c, 1, 5, 1), padding=(2, 0), groups=c)
    x = F.conv2d(x, kx.expand(c, 1, 1, 5), padding=(0, 2), groups=c)
    return x


def _soft_dice_loss(pred, target, eps=1e-6):
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    dims = tuple(range(1, pred.dim()))
    inter = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return (1.0 - dice).mean()


def _balanced_bce_loss(pred, target, max_pos_weight=20.0, eps=1e-6):
    """BCE for sparse soft targets with per-image positive reweighting."""
    pred = pred.clamp(eps, 1.0 - eps)
    target = target.clamp(0, 1)
    pos = target.mean(dim=(-2, -1), keepdim=True).detach()
    pos_weight = ((1.0 - pos) / (pos + eps)).clamp(1.0, float(max_pos_weight))
    weight = 1.0 + target.detach() * (pos_weight - 1.0)
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    return (bce * weight).mean()


def _weighted_region_mean(x, weight, eps=1e-6):
    return (x * weight).sum() / (weight.sum() + eps)


def make_topology_primitive_targets(img, out_hw):
    """Create weak topology-primitive targets from HR/GT images.

    Topology-density prior keeps the successful edge/link/orientation targets, but replaces
    the old needle-like node target with a soft junction disk.  The previous
    target was too sparse, so topology_node could minimize its loss by becoming
    almost all black.  Here a structure-tensor corner response is dilated and
    softly blurred, giving the node head a learnable local region around
    corners/endpoints/junctions while still remaining sparse.
    """
    gray = img.mean(dim=1, keepdim=True)
    dtype, device = gray.dtype, gray.device
    kx = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]], device=device, dtype=dtype).view(1, 1, 3, 3) / 16.0
    ky = kx.transpose(-1, -2)
    dx = F.conv2d(gray, kx, padding=1)
    dy = F.conv2d(gray, ky, padding=1)
    grad = torch.sqrt(dx * dx + dy * dy + 1e-12)
    local_grad = F.avg_pool2d(grad, kernel_size=3, stride=1, padding=1)
    edge = _normalize_per_image(grad + (grad - local_grad).abs())

    # Structure tensor / Harris-like junction response.
    jxx = F.avg_pool2d(dx * dx, kernel_size=3, stride=1, padding=1)
    jyy = F.avg_pool2d(dy * dy, kernel_size=3, stride=1, padding=1)
    jxy = F.avg_pool2d(dx * dy, kernel_size=3, stride=1, padding=1)
    det = jxx * jyy - jxy * jxy
    trace = jxx + jyy
    raw_node = F.relu(det - 0.04 * trace * trace)

    # Anti-collapse soft node target: normalize, dilate, then blur.  This keeps
    # the target local and sparse, but avoids single-pixel positives that are
    # easily ignored by a reconstruction-dominated SR objective.
    node_peak = _normalize_per_image(raw_node)
    node_disk = F.max_pool2d(node_peak, kernel_size=7, stride=1, padding=3)
    node = _normalize_per_image(_gaussian_blur_5x5(node_disk))

    edge_smooth = F.avg_pool2d(edge, kernel_size=5, stride=1, padding=2)
    node_smooth = F.avg_pool2d(node, kernel_size=5, stride=1, padding=2)
    link = _normalize_per_image(F.relu(edge_smooth - 0.5 * node_smooth))

    orient = torch.cat([dx, -dy], dim=1)
    orient = F.normalize(orient, dim=1, eps=1e-6)

    if edge.shape[-2:] != tuple(out_hw):
        edge = F.interpolate(edge, size=out_hw, mode='bilinear', align_corners=False)
        node = F.interpolate(node, size=out_hw, mode='bilinear', align_corners=False)
        link = F.interpolate(link, size=out_hw, mode='bilinear', align_corners=False)
        orient = F.interpolate(orient, size=out_hw, mode='bilinear', align_corners=False)
        orient = F.normalize(orient, dim=1, eps=1e-6)

    # A soft edge-neighborhood target used only for node-edge consistency: nodes
    # should appear near line primitives, not as isolated random bright pixels.
    edge_near = F.max_pool2d(edge.clamp(0, 1), kernel_size=5, stride=1, padding=2)

    return {
        'edge': edge.clamp(0, 1),
        'node': node.clamp(0, 1),
        'link': link.clamp(0, 1),
        'edge_near': edge_near.clamp(0, 1),
        'orientation': orient,
    }


def primitive_topology_loss_from_gt(aux, gt_img, loss_cfg):
    """Compute the topology objective without diagnostic statistics."""
    if not aux or aux.get('topology_edge') is None or gt_img is None:
        return gt_img.new_tensor(0.0) if torch.is_tensor(gt_img) else None

    pred_edge = aux.get('topology_edge')
    pred_node = aux.get('topology_node')
    pred_link = aux.get('topology_link')
    pred_ori = aux.get('orientation')
    targets = make_topology_primitive_targets(
        gt_img.detach(), pred_edge.shape[-2:]
    )

    edge_w = float(loss_cfg.get('primitive_edge_w', 1.0))
    node_w = float(loss_cfg.get('primitive_node_w', 1.0))
    link_w = float(loss_cfg.get('primitive_link_w', 0.35))
    dir_w = float(loss_cfg.get('primitive_dir_w', 0.2))
    node_dice_w = float(loss_cfg.get('primitive_node_dice_w', 0.5))
    node_edge_w = float(loss_cfg.get('primitive_node_edge_w', 0.2))

    edge_loss = F.binary_cross_entropy(
        pred_edge.clamp(1e-6, 1 - 1e-6), targets['edge']
    )

    if pred_node is not None:
        pred_node_c = pred_node.clamp(1e-6, 1 - 1e-6)
        node_target = targets['node'].to(pred_node_c.dtype)
        node_bce = _balanced_bce_loss(pred_node_c, node_target)
        node_dice = _soft_dice_loss(pred_node_c, node_target)
        edge_near = targets['edge_near'].to(pred_node_c.dtype).detach()
        node_edge = (pred_node_c * (1.0 - edge_near)).mean()
        node_loss = (
            node_bce + node_dice_w * node_dice + node_edge_w * node_edge
        )
    else:
        node_loss = edge_loss.new_tensor(0.0)

    link_loss = (
        F.binary_cross_entropy(
            pred_link.clamp(1e-6, 1 - 1e-6), targets['link']
        )
        if pred_link is not None else edge_loss.new_tensor(0.0)
    )

    if pred_ori is not None:
        pred_ori = F.normalize(pred_ori, dim=1, eps=1e-6)
        tgt_ori = targets['orientation'].to(pred_ori.dtype)
        cos = (pred_ori * tgt_ori).sum(
            dim=1, keepdim=True
        ).abs().clamp(0, 1)
        mask = targets['edge'].to(pred_ori.dtype).detach()
        dir_loss = ((1.0 - cos) * mask).sum() / (mask.sum() + 1e-6)
    else:
        dir_loss = edge_loss.new_tensor(0.0)

    return (
        edge_w * edge_loss
        + node_w * node_loss
        + link_w * link_loss
        + dir_w * dir_loss
    )


def sr_gradient_loss(pred, gt):
    """L1 gradient loss on SR output.

    This directly supervises reconstructed edges and line structures instead of
    only supervising topology maps.  It is kept as a compact auxiliary term and
    can be interpreted together with the topology primitive loss as the
    structure reconstruction objective.
    """
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]
    return F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)


def model_forward(model, inp, batch, return_aux=False):
    """Forward helper carrying GSASR-style scale_modify/gt_size metadata."""
    return model(
        inp,
        batch['scale'],
        gt_size=batch.get('gt_size'),
        scale_modify=batch.get('scale_modify'),
        return_aux=return_aux,
    )


def compute_losses(model, batch, epoch=None):
    """Compute only terms that participate in the active objective."""
    del epoch
    inp = normalize_img(
        batch['inp'], config['data_norm']['inp'], batch['inp'].device
    )
    gt = normalize_img(
        batch['gt'], config['data_norm']['gt'], batch['gt'].device
    )
    loss_cfg = config.get('loss', {}) or {}
    primitive_w = float(loss_cfg.get('primitive_topology_w', 1.5e-3))

    aux_mode = 'loss' if primitive_w > 0 else False
    out = model_forward(model, inp, batch, return_aux=aux_mode)
    if isinstance(out, tuple):
        pred, aux = out
    else:
        pred, aux = out, {}

    pred = pred[..., :gt.shape[-2], :gt.shape[-1]]
    total = F.l1_loss(pred, gt)

    if primitive_w > 0 and aux.get('topology_edge') is not None:
        topology_loss = primitive_topology_loss_from_gt(
            aux, batch.get('gt'), loss_cfg
        )
        total = total + primitive_w * topology_loss

    sr_grad_w = float(loss_cfg.get('sr_gradient_w', 0.0))
    if sr_grad_w > 0:
        total = total + sr_grad_w * sr_gradient_loss(pred, gt)

    return total


def train_one_epoch(loader, model, optimizer, epoch):
    model.train()
    loss_sum = None
    sample_count = 0

    iterator = tqdm(loader, leave=False, desc='train')
    for batch in iterator:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.cuda(non_blocking=True)

        loss = compute_losses(model, batch, epoch=epoch)

        optimizer.zero_grad()
        loss.backward()
        grad_clip = config.get('optimizer', {}).get(
            'grad_clip', config.get('grad_clip', None)
        )
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()

        batch_size = int(batch['inp'].shape[0])
        weighted = loss.detach().to(torch.float64) * batch_size
        loss_sum = weighted if loss_sum is None else loss_sum + weighted
        sample_count += batch_size

    if loss_sum is None or sample_count == 0:
        return 0.0
    # A single GPU-to-CPU synchronization per epoch replaces one .item() call
    # per training batch.  This changes reporting overhead only, not gradients.
    return (loss_sum / float(sample_count)).item()


def _make_single_sample_batch(dataset, index, device):
    sample = dataset[index]
    batch = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            batch[k] = v.unsqueeze(0).to(device, non_blocking=True)
        elif isinstance(v, (int, float)):
            batch[k] = torch.tensor([v], device=device)
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v):
            batch[k] = torch.tensor([v], device=device)
        else:
            batch[k] = v
    return batch


def _save_tensor_image(x, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_image(x.detach().cpu().clamp(0, 1), path)


def _normalize_map_for_save(x):
    x = x.detach()
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + 1e-6)


@torch.no_grad()
def save_val_visualizations(loader, model, save_path, epoch, vis_cfg):
    """
    Save fixed validation SR images using the config format:

    val_vis:
      enabled: true
      save_every: 1
      indices: [30, 460, 568, 963]
      save_lr: false
      save_compare: true
      save_prior: true
      overwrite: false
      subdir: val_vis_selected

    Compare images are always saved as LR-SR-HR triplets when save_compare is
    true.  save_lr only controls whether the separate LR_bicubic.png file is
    saved in addition to the triplet.
    """
    if loader is None or not vis_cfg or not vis_cfg.get('enabled', False):
        return 0

    save_every = int(vis_cfg.get('save_every', 1))
    if save_every <= 0 or epoch % save_every != 0:
        return 0

    indices = vis_cfg.get('indices', [])
    if indices is None:
        indices = []
    indices = [int(i) for i in indices]
    if len(indices) == 0:
        indices = list(range(min(4, len(loader.dataset))))

    save_lr = bool(vis_cfg.get('save_lr', False))
    save_compare = bool(vis_cfg.get('save_compare', True))
    save_prior = bool(vis_cfg.get('save_prior', True))
    overwrite = bool(vis_cfg.get('overwrite', False))
    subdir = str(vis_cfg.get('subdir', 'val_vis_selected'))

    root = os.path.join(save_path, subdir)
    if not overwrite:
        root = os.path.join(root, f'epoch_{epoch:04d}')
    os.makedirs(root, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()

    saved = 0
    dataset_len = len(loader.dataset)
    for index in indices:
        if index < 0 or index >= dataset_len:
            continue

        batch = _make_single_sample_batch(loader.dataset, index, device)
        inp = normalize_img(batch['inp'], config['data_norm']['inp'], device)

        out = model_forward(
            model, inp, batch, return_aux=('prior' if save_prior else False)
        )
        if isinstance(out, tuple):
            pred, aux = out
        else:
            pred, aux = out, {}

        pred = denormalize_img(pred, config['data_norm']['gt'], device).clamp(0, 1)
        hr = batch['gt'].clamp(0, 1)
        pred = pred[..., :hr.shape[-2], :hr.shape[-1]]

        lr_up = F.interpolate(
            batch['inp'], size=hr.shape[-2:], mode='bicubic', align_corners=False
        ).clamp(0, 1)

        idx_prefix = f'idx_{index:05d}'
        _save_tensor_image(pred[0], os.path.join(root, f'{idx_prefix}_SR.png'))
        _save_tensor_image(hr[0], os.path.join(root, f'{idx_prefix}_HR.png'))

        if save_lr:
            _save_tensor_image(lr_up[0], os.path.join(root, f'{idx_prefix}_LR_bicubic.png'))

        if save_compare:
            compare = torch.stack([lr_up[0], pred[0], hr[0]], dim=0)
            _save_tensor_image(compare, os.path.join(root, f'{idx_prefix}_LR_SR_HR.png'))

        if save_prior and aux:
            prior_dir = os.path.join(root, 'prior_maps')
            core_prior_keys = [
                'complexity', 'structure_complexity',
                'density_raw', 'density_blend', 'density',
                'scaffold_score', 'seed_prob',
                'alpha_floor_map', 'sigma_floor_map',
                'topology_node', 'topology_edge',
                'topology_link', 'topology_importance',
            ]
            for key in core_prior_keys:
                m = aux.get(key)
                if m is None:
                    continue
                if m.dim() == 4:
                    m = F.interpolate(
                        m,
                        size=hr.shape[-2:],
                        mode='bilinear',
                        align_corners=False,
                    )[0]
                elif m.dim() == 3:
                    m = m[0:1]
                m = _normalize_map_for_save(m)
                _save_tensor_image(
                    m,
                    os.path.join(prior_dir, f'{idx_prefix}_{key}.png'),
                )

        saved += 1

    return saved


@torch.no_grad()
def evaluate(loader, model):
    """Validate with one configured PSNR color space.

    The default is MATLAB-style Y PSNR. Optional border cropping is resolved
    automatically from the actual per-image scale metadata.
    """
    model.eval()
    if loader is None:
        return None

    eval_cfg = config.get('evaluation', {}) or {}
    color_space = str(eval_cfg.get('color_space', 'y')).lower()
    mod_scale = int(eval_cfg.get('mod_scale', 4))
    crop = bool(eval_cfg.get('crop', False))
    quantize = bool(eval_cfg.get('quantize', True))

    psnr_sum = None
    sample_count = 0
    for batch in tqdm(loader, leave=False, desc=f'val-psnr-{color_space}'):
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.cuda(non_blocking=True)

        inp = normalize_img(
            batch['inp'], config['data_norm']['inp'], batch['inp'].device
        )
        pred = model_forward(model, inp, batch, return_aux=False)
        pred = denormalize_img(
            pred, config['data_norm']['gt'], batch['inp'].device
        ).clamp(0, 1)
        gt = batch['gt'].clamp(0, 1)
        pred = pred[..., :gt.shape[-2], :gt.shape[-1]]
        scale_meta = batch.get('scale_modify', batch.get('scale'))
        psnr = calculate_psnr(
            pred, gt,
            color_space=color_space,
            mod_scale=mod_scale,
            crop=crop,
            scale=scale_meta,
            quantize=quantize,
        )
        batch_size = int(inp.shape[0])
        psnr_tensor = (
            psnr.detach().to(torch.float64)
            if torch.is_tensor(psnr)
            else torch.tensor(float(psnr), device=inp.device, dtype=torch.float64)
        )
        weighted = psnr_tensor * batch_size
        psnr_sum = weighted if psnr_sum is None else psnr_sum + weighted
        sample_count += batch_size

    if psnr_sum is None or sample_count == 0:
        return None
    # One synchronization after the complete validation set instead of one per
    # image/batch.  The PSNR formula and averaging weights are unchanged.
    return (psnr_sum / float(sample_count)).item()


def save_checkpoint(
    model,
    optimizer,
    epoch,
    save_path,
    filename,
    scheduler=None,
    best_psnr=-1e18,
    bad_epochs=0,
):
    """Save full training state, including scheduler and RNG states."""
    model_ = model.module if isinstance(model, torch.nn.DataParallel) else model

    # Copy only the top-level dicts so adding state_dict entries does not mutate
    # the active config object used by the rest of training.
    model_spec = dict(config['model'])
    model_spec['sd'] = model_.state_dict()
    optimizer_spec = dict(config['optimizer'])
    optimizer_spec['sd'] = optimizer.state_dict()

    payload = {
        'model': model_spec,
        'optimizer': optimizer_spec,
        'epoch': int(epoch),
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
        'best_psnr': float(best_psnr),
        'bad_epochs': int(bad_epochs),
    }
    payload.update(_capture_rng_state())

    torch.save(payload, os.path.join(save_path, filename))


def should_validate_epoch(epoch, config):
    """Return whether validation should run at the current epoch.

    Preferred current YAML format:
        val_schedule:
          switch_epoch: 350
          before_every: 100
          after_every: 1

    With the values above:
      - epochs < 350: validate at 100, 200, 300, ...
      - epoch 350: validate at the transition
      - epochs > 350: validate every epoch

    The older validation_schedule/epoch_val formats are also supported.
    """
    schedule = config.get('val_schedule')
    if schedule is not None:
        schedule = schedule or {}
        switch_epoch = int(schedule.get('switch_epoch', 350))
        before_every = int(schedule.get('before_every', 100))
        after_every = int(schedule.get('after_every', 1))

        if switch_epoch < 1:
            raise ValueError('val_schedule.switch_epoch must be >= 1.')
        if before_every < 1:
            raise ValueError('val_schedule.before_every must be >= 1.')
        if after_every < 1:
            raise ValueError('val_schedule.after_every must be >= 1.')

        if epoch < switch_epoch:
            return epoch % before_every == 0
        if epoch == switch_epoch:
            return True
        return (epoch - switch_epoch) % after_every == 0

    # Backward compatibility with an older naming convention used by some
    # experiment configs.
    schedule = config.get('validation_schedule')
    if schedule is not None:
        schedule = schedule or {}
        switch_epoch = int(schedule.get('switch_epoch', 150))
        early_interval = int(schedule.get('early_interval', 20))
        late_interval = int(schedule.get('late_interval', 1))

        if switch_epoch < 1 or early_interval < 1 or late_interval < 1:
            raise ValueError(
                'validation_schedule values must all be positive integers.'
            )
        if epoch < switch_epoch:
            return epoch % early_interval == 0
        if epoch == switch_epoch:
            return True
        return (epoch - switch_epoch) % late_interval == 0

    # Legacy fixed validation interval.
    epoch_val = config.get('epoch_val')
    if epoch_val is None:
        return False
    epoch_val = int(epoch_val)
    if epoch_val < 1:
        raise ValueError('epoch_val must be >= 1 when validation is enabled.')
    return epoch % epoch_val == 0


def main(config_, save_path):
    global config, log, writer
    config = config_

    if config.get('seed') is not None:
        set_seed(config['seed'])

    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]},
        }

    train_loader = make_data_loader(config.get('train_dataset'), 'train')
    val_loader = make_data_loader(config.get('val_dataset'), 'val')
    model, optimizer, epoch_start, scheduler, resume_meta = prepare_training()

    n_gpus = len(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(','))
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)

    best_psnr = float(resume_meta.get('best_psnr', -1e18))
    bad_epochs = int(resume_meta.get('bad_epochs', 0))
    epoch_max = int(config['epoch_max'])
    train_t0 = time.time()
    total_epochs = max(0, epoch_max - epoch_start + 1)

    if epoch_start > epoch_max:
        log(
            f'nothing to train: checkpoint starts at epoch {epoch_start}, '
            f'but epoch_max={epoch_max}'
        )
        return

    for epoch in range(epoch_start, epoch_max + 1):
        epoch_t0 = time.time()
        train_loss = train_one_epoch(train_loader, model, optimizer, epoch)
        if scheduler is not None:
            scheduler.step()

        # Only the optimized total loss is stored and displayed for training.
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

        val_psnr = None
        eval_color_space = str(
            (config.get('evaluation', {}) or {}).get('color_space', 'y')
        ).lower()

        if should_validate_epoch(epoch, config):
            model_for_eval = (
                model.module if isinstance(model, torch.nn.DataParallel) else model
            )
            val_psnr = evaluate(val_loader, model_for_eval)
            if val_psnr is not None:
                writer.add_scalar(
                    f'val/psnr_{eval_color_space}', val_psnr, epoch
                )

            # Preserve the original visualization behavior.  Full diagnostics
            # are calculated only for explicitly selected visualization samples.
            vis_cfg = config.get('val_vis', {}) or {}
            if vis_cfg.get('enabled', False):
                save_val_visualizations(
                    val_loader, model_for_eval, save_path, epoch, vis_cfg
                )

            early_cfg = config.get('early_stop', {}) or {}
            min_delta = float(early_cfg.get('min_delta', 0.0))
            if val_psnr is not None and val_psnr > best_psnr + min_delta:
                best_psnr = float(val_psnr)
                bad_epochs = 0
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    save_path,
                    'epoch-best.pth',
                    scheduler=scheduler,
                    best_psnr=best_psnr,
                    bad_epochs=bad_epochs,
                )
            elif val_psnr is not None:
                bad_epochs += 1

            should_stop = False
            if early_cfg.get('enabled', False):
                patience = int(early_cfg.get('patience', 10))
                start_epoch = int(early_cfg.get('start_epoch', 1))
                if epoch >= start_epoch and bad_epochs >= patience:
                    should_stop = True
        else:
            should_stop = False

        # Save regular checkpoints AFTER validation/visualization.  This is
        # important for exact future resume because validation can consume RNG
        # state (for example through dataset scale bookkeeping).
        save_checkpoint(
            model,
            optimizer,
            epoch,
            save_path,
            'epoch-last.pth',
            scheduler=scheduler,
            best_psnr=best_psnr,
            bad_epochs=bad_epochs,
        )
        if config.get('epoch_save') is not None and epoch % int(config['epoch_save']) == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                save_path,
                f'epoch-{epoch}.pth',
                scheduler=scheduler,
                best_psnr=best_psnr,
                bad_epochs=bad_epochs,
            )

        epoch_time = time.time() - epoch_t0
        elapsed = time.time() - train_t0
        done_epochs = epoch - epoch_start + 1
        total_est = elapsed / max(done_epochs, 1) * max(total_epochs, 1)

        info = [
            f'epoch {epoch}/{epoch_max}',
            f'train: loss={train_loss:.4f}',
            f'lr={optimizer.param_groups[0]["lr"]:.8g}',
        ]
        if val_psnr is not None:
            info.append(f'val: psnr_{eval_color_space}={val_psnr:.4f}')
        info.append(
            f'{format_duration(epoch_time)} '
            f'{format_duration(elapsed)}/{format_duration(total_est)}'
        )
        log(', '.join(info))
        writer.flush()
        if should_stop:
            log(
                f'early stop at epoch {epoch}: bad_epochs={bad_epochs}, '
                f'best_psnr={best_psnr:.6f}'
            )
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    save_name = args.name
    if save_name is None:
        save_name = '_' + os.path.basename(args.config)[:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path)
