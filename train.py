import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils
from metrics.sr_metrics import calculate_psnr


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_data_loader(spec, train=False):
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    return DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        shuffle=train,
        num_workers=spec.get('num_workers', 0),
        pin_memory=True,
    )


def normalize_img(x, norm, device):
    sub = torch.FloatTensor(norm['sub']).view(1, -1, 1, 1).to(device)
    div = torch.FloatTensor(norm['div']).view(1, -1, 1, 1).to(device)
    return (x - sub) / div


def denormalize_img(x, norm, device):
    sub = torch.FloatTensor(norm['sub']).view(1, -1, 1, 1).to(device)
    div = torch.FloatTensor(norm['div']).view(1, -1, 1, 1).to(device)
    return x * div + sub


def _normalize_per_image(x, eps=1e-06):
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


def _gaussian_blur_5x5(x):
    if x is None:
        return None
    dtype, device = (x.dtype, x.device)
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=dtype, device=device)
    k = k / k.sum()
    ky = k.view(1, 1, 5, 1)
    kx = k.view(1, 1, 1, 5)
    c = x.shape[1]
    x = F.conv2d(x, ky.expand(c, 1, 5, 1), padding=(2, 0), groups=c)
    x = F.conv2d(x, kx.expand(c, 1, 1, 5), padding=(0, 2), groups=c)
    return x


def _soft_dice_loss(pred, target, eps=1e-06):
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    dims = tuple(range(1, pred.dim()))
    inter = (pred * target).sum(dim=dims)
    denom = pred.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return (1.0 - dice).mean()


def _balanced_bce_loss(pred, target, max_pos_weight=20.0, eps=1e-06):
    pred = pred.clamp(eps, 1.0 - eps)
    target = target.clamp(0, 1)
    pos = target.mean(dim=(-2, -1), keepdim=True).detach()
    pos_weight = ((1.0 - pos) / (pos + eps)).clamp(1.0, float(max_pos_weight))
    weight = 1.0 + target.detach() * (pos_weight - 1.0)
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    return (bce * weight).mean()


# Weak supervision targets for local topology.
def make_topology_primitive_targets(img, out_hw):
    gray = img.mean(dim=1, keepdim=True)
    dtype, device = (gray.dtype, gray.device)
    kx = torch.tensor(
        [[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]],
        device=device, dtype=dtype,
    ).view(1, 1, 3, 3) / 16.0
    ky = kx.transpose(-1, -2)
    dx = F.conv2d(gray, kx, padding=1)
    dy = F.conv2d(gray, ky, padding=1)
    grad = torch.sqrt(dx * dx + dy * dy + 1e-12)
    local_grad = F.avg_pool2d(grad, kernel_size=3, stride=1, padding=1)
    edge = _normalize_per_image(grad + (grad - local_grad).abs())
    jxx = F.avg_pool2d(dx * dx, kernel_size=3, stride=1, padding=1)
    jyy = F.avg_pool2d(dy * dy, kernel_size=3, stride=1, padding=1)
    jxy = F.avg_pool2d(dx * dy, kernel_size=3, stride=1, padding=1)
    det = jxx * jyy - jxy * jxy
    trace = jxx + jyy
    raw_node = F.relu(det - 0.04 * trace * trace)
    node_peak = _normalize_per_image(raw_node)
    node_disk = F.max_pool2d(node_peak, kernel_size=7, stride=1, padding=3)
    node = _normalize_per_image(_gaussian_blur_5x5(node_disk))
    edge_smooth = F.avg_pool2d(edge, kernel_size=5, stride=1, padding=2)
    node_smooth = F.avg_pool2d(node, kernel_size=5, stride=1, padding=2)
    link = _normalize_per_image(F.relu(edge_smooth - 0.5 * node_smooth))
    orient = torch.cat([dx, -dy], dim=1)
    orient = F.normalize(orient, dim=1, eps=1e-06)
    if edge.shape[-2:] != tuple(out_hw):
        edge = F.interpolate(edge, size=out_hw, mode='bilinear', align_corners=False)
        node = F.interpolate(node, size=out_hw, mode='bilinear', align_corners=False)
        link = F.interpolate(link, size=out_hw, mode='bilinear', align_corners=False)
        orient = F.interpolate(orient, size=out_hw, mode='bilinear', align_corners=False)
        orient = F.normalize(orient, dim=1, eps=1e-06)
    edge_near = F.max_pool2d(edge.clamp(0, 1), kernel_size=5, stride=1, padding=2)
    return {
        'edge': edge.clamp(0, 1),
        'node': node.clamp(0, 1),
        'link': link.clamp(0, 1),
        'edge_near': edge_near.clamp(0, 1),
        'orientation': orient,
    }


def primitive_topology_loss_from_gt(aux, gt_img, loss_cfg):
    if not aux or aux.get('topology_edge') is None or gt_img is None:
        return gt_img.new_tensor(0.0) if torch.is_tensor(gt_img) else None
    pred_edge = aux.get('topology_edge')
    pred_node = aux.get('topology_node')
    pred_link = aux.get('topology_link')
    pred_ori = aux.get('orientation')
    targets = make_topology_primitive_targets(gt_img.detach(), pred_edge.shape[-2:])
    edge_w = float(loss_cfg.get('primitive_edge_w', 1.0))
    node_w = float(loss_cfg.get('primitive_node_w', 1.0))
    link_w = float(loss_cfg.get('primitive_link_w', 0.35))
    dir_w = float(loss_cfg.get('primitive_dir_w', 0.2))
    node_dice_w = float(loss_cfg.get('primitive_node_dice_w', 0.5))
    node_edge_w = float(loss_cfg.get('primitive_node_edge_w', 0.2))
    edge_loss = F.binary_cross_entropy(pred_edge.clamp(1e-06, 1 - 1e-06), targets['edge'])
    if pred_node is not None:
        pred_node_c = pred_node.clamp(1e-06, 1 - 1e-06)
        node_target = targets['node'].to(pred_node_c.dtype)
        node_bce = _balanced_bce_loss(pred_node_c, node_target)
        node_dice = _soft_dice_loss(pred_node_c, node_target)
        edge_near = targets['edge_near'].to(pred_node_c.dtype).detach()
        node_edge = (pred_node_c * (1.0 - edge_near)).mean()
        node_loss = node_bce + node_dice_w * node_dice + node_edge_w * node_edge
    else:
        node_loss = edge_loss.new_tensor(0.0)
    link_loss = (
        F.binary_cross_entropy(pred_link.clamp(1e-6, 1 - 1e-6), targets['link'])
        if pred_link is not None else edge_loss.new_tensor(0.0)
    )
    if pred_ori is not None:
        pred_ori = F.normalize(pred_ori, dim=1, eps=1e-06)
        tgt_ori = targets['orientation'].to(pred_ori.dtype)
        cos = (pred_ori * tgt_ori).sum(dim=1, keepdim=True).abs().clamp(0, 1)
        mask = targets['edge'].to(pred_ori.dtype).detach()
        dir_loss = ((1.0 - cos) * mask).sum() / (mask.sum() + 1e-06)
    else:
        dir_loss = edge_loss.new_tensor(0.0)
    return edge_w * edge_loss + node_w * node_loss + link_w * link_loss + dir_w * dir_loss


def sr_gradient_loss(pred, gt):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]
    return F.l1_loss(pred_dx, gt_dx) + F.l1_loss(pred_dy, gt_dy)


def model_forward(model, inp, batch, return_aux=False):
    return model(
        inp, batch['scale'],
        gt_size=batch.get('gt_size'),
        scale_modify=batch.get('scale_modify'),
        return_aux=return_aux,
    )


# Reconstruction, topology supervision, and gradient consistency.
def compute_losses(model, batch):
    inp = normalize_img(batch['inp'], config['data_norm']['inp'], batch['inp'].device)
    gt = normalize_img(batch['gt'], config['data_norm']['gt'], batch['gt'].device)
    loss_cfg = config.get('loss', {}) or {}
    primitive_w = float(loss_cfg.get('primitive_topology_w', 0.0015))
    aux_mode = 'loss' if primitive_w > 0 else False
    out = model_forward(model, inp, batch, return_aux=aux_mode)
    if isinstance(out, tuple):
        pred, aux = out
    else:
        pred, aux = (out, {})
    pred = pred[..., :gt.shape[-2], :gt.shape[-1]]
    total = F.l1_loss(pred, gt)
    if primitive_w > 0 and aux.get('topology_edge') is not None:
        topology_loss = primitive_topology_loss_from_gt(aux, batch.get('gt'), loss_cfg)
        total = total + primitive_w * topology_loss
    sr_grad_w = float(loss_cfg.get('sr_gradient_w', 0.0))
    if sr_grad_w > 0:
        total = total + sr_grad_w * sr_gradient_loss(pred, gt)
    return total


def train_one_epoch(loader, model, optimizer):
    model.train()
    loss_sum = None
    sample_count = 0
    iterator = tqdm(loader, leave=False, desc='train')
    for batch in iterator:
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.cuda(non_blocking=True)
        loss = compute_losses(model, batch)
        optimizer.zero_grad()
        loss.backward()
        grad_clip = config.get('optimizer', {}).get('grad_clip', config.get('grad_clip', None))
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()
        batch_size = int(batch['inp'].shape[0])
        weighted = loss.detach().to(torch.float64) * batch_size
        loss_sum = weighted if loss_sum is None else loss_sum + weighted
        sample_count += batch_size
    if loss_sum is None or sample_count == 0:
        return 0.0
    return (loss_sum / float(sample_count)).item()


@torch.no_grad()
def evaluate(loader, model):
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
        inp = normalize_img(batch['inp'], config['data_norm']['inp'], batch['inp'].device)
        pred = model_forward(model, inp, batch, return_aux=False)
        pred = denormalize_img(pred, config['data_norm']['gt'], batch['inp'].device).clamp(0, 1)
        gt = batch['gt'].clamp(0, 1)
        pred = pred[..., :gt.shape[-2], :gt.shape[-1]]
        scale_meta = batch.get('scale_modify', batch.get('scale'))
        psnr = calculate_psnr(
            pred, gt, color_space=color_space, mod_scale=mod_scale,
            crop=crop, scale=scale_meta, quantize=quantize,
        )
        batch_size = int(inp.shape[0])
        psnr_tensor = (
            psnr.detach().to(torch.float64) if torch.is_tensor(psnr)
            else torch.tensor(float(psnr), device=inp.device, dtype=torch.float64)
        )
        weighted = psnr_tensor * batch_size
        psnr_sum = weighted if psnr_sum is None else psnr_sum + weighted
        sample_count += batch_size
    if psnr_sum is None or sample_count == 0:
        return None
    return (psnr_sum / float(sample_count)).item()


def should_validate_epoch(epoch, cfg):
    interval = int(cfg.get('epoch_val', 1))
    if interval < 1:
        raise ValueError('epoch_val must be a positive integer')
    return epoch % interval == 0


def save_checkpoint(model, save_path, filename):
    model = model.module if isinstance(model, torch.nn.DataParallel) else model
    model_spec = dict(config['model'])
    model_spec['sd'] = model.state_dict()
    torch.save({'model': model_spec}, os.path.join(save_path, filename))


def main(cfg, save_path):
    global config
    config = cfg

    set_seed(config.get('seed', 42))
    log, writer = utils.set_save_path(save_path, remove=False)
    with open(os.path.join(save_path, 'config.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, sort_keys=False)

    train_loader = make_data_loader(config['train_dataset'], train=True)
    val_loader = make_data_loader(config['val_dataset'])
    model = models.make(config['model']).cuda()
    optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
    cosine_cfg = config['cosine_lr']
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(cosine_cfg['T_max']),
        eta_min=float(cosine_cfg['eta_min']),
    )

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:
        model = torch.nn.DataParallel(model)

    best_psnr = float('-inf')
    last_epoch = 0
    for epoch in range(1, int(config['epoch_max']) + 1):
        train_loss = train_one_epoch(train_loader, model, optimizer)
        scheduler.step()
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

        val_psnr = None
        if should_validate_epoch(epoch, config):
            eval_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            val_psnr = evaluate(val_loader, eval_model)
            if val_psnr is not None:
                color = config.get('evaluation', {}).get('color_space', 'y')
                writer.add_scalar(f'val/psnr_{color}', val_psnr, epoch)
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    save_checkpoint(model, save_path, 'epoch-best.pth')

        save_every = config.get('epoch_save')
        if save_every and epoch % int(save_every) == 0:
            save_checkpoint(model, save_path, f'epoch-{epoch}.pth')

        message = (
            f"epoch {epoch}/{config['epoch_max']}, "
            f"loss={train_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.8g}"
        )
        if val_psnr is not None:
            message += f', val_psnr={val_psnr:.4f}'
        log(message)
        writer.flush()
        last_epoch = epoch

    if last_epoch:
        save_checkpoint(model, save_path, 'epoch-last.pth')
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--name', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    save_name = args.name or os.path.splitext(os.path.basename(args.config))[0]
    main(config, os.path.join('save', save_name))
