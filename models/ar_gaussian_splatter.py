import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register

try:
    from gsplat.project_gaussians_2d import project_gaussians_2d
    from gsplat.rasterize_sum import rasterize_gaussians_sum
except ImportError:
    project_gaussians_2d = None
    rasterize_gaussians_sum = None


def make_coord(shape, ranges=None, flatten=True):
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    try:
        ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    except TypeError:
        ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def normalized_yx_to_gsplat_xy(mu_yx, out_h, out_w):
    out_h = int(out_h)
    out_w = int(out_w)
    if out_h <= 0 or out_w <= 0:
        raise ValueError(f'Output size must be positive, got {(out_h, out_w)}.')
    if mu_yx.shape[-1] != 2:
        raise ValueError(f'Expected coordinates ending in 2 values [y, x], got {tuple(mu_yx.shape)}.')

    means_xy = mu_yx[..., [1, 0]]
    half_pixel_shift_xy = means_xy.new_tensor([
        1.0 / float(out_w),
        1.0 / float(out_h),
    ])
    return means_xy - half_pixel_shift_xy


class LocalTopologyGraphBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.edge_proj = nn.Conv2d(5, 9, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(1, channels)
        self.scale = channels ** -0.5

    @staticmethod
    def _shift(x, dy, dx):
        h, w = x.shape[-2:]
        pad_l = max(dx, 0)
        pad_r = max(-dx, 0)
        pad_t = max(dy, 0)
        pad_b = max(-dy, 0)
        xp = F.pad(x, (pad_l, pad_r, pad_t, pad_b), mode='replicate')
        y0 = pad_b
        x0 = pad_r
        return xp[..., y0:y0 + h, x0:x0 + w]

    def forward(self, feat, topology_importance, topology_link, density, orientation):
        if topology_link is None:
            topology_link = topology_importance
        if orientation is None:
            orientation = feat.new_zeros(feat.shape[0], 2, feat.shape[-2], feat.shape[-1])
        topo = torch.cat([
            topology_importance,
            topology_link,
            density,
            orientation,
        ], dim=1).to(dtype=feat.dtype)
        edge_bias = self.edge_proj(topo)

        q = self.q_proj(feat)
        k = self.k_proj(feat)
        v = self.v_proj(feat)
        shifts = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 0), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ]
        scores = []
        values = []
        for idx, (dy, dx) in enumerate(shifts):
            ks = self._shift(k, dy, dx)
            vs = self._shift(v, dy, dx)
            score = (q * ks).sum(dim=1, keepdim=True) * self.scale + edge_bias[:, idx:idx + 1]
            scores.append(score)
            values.append(vs)
        attn = torch.softmax(torch.cat(scores, dim=1), dim=1)
        msg = 0.0
        for idx, vs in enumerate(values):
            msg = msg + attn[:, idx:idx + 1] * vs
        return feat + self.out_proj(self.norm(msg))


class SiblingInteractionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)
        self.scale = channels ** -0.5

    def forward(self, child_feat):
        # Child features: [B, N, P, K, C].
        q = self.q_proj(child_feat)
        k = self.k_proj(child_feat)
        v = self.v_proj(child_feat)
        attn = torch.softmax(
            torch.matmul(q, k.transpose(-1, -2)) * self.scale,
            dim=-1,
        )
        msg = torch.matmul(attn, v)
        return self.norm(child_feat + self.out_proj(msg))


def _reshape_head(x, num_groups, group_dim):
    """[B, num_groups*group_dim, H, W] -> [B, H*W, num_groups, group_dim]."""
    b, _, h, w = x.shape
    x = x.view(b, num_groups, group_dim, h, w)
    x = x.permute(0, 3, 4, 1, 2).contiguous().view(b, h * w, num_groups, group_dim)
    return x


@register('td-ar-gs-sr')
class TDARGaussianSplatter(nn.Module):
    def __init__(
        self,
        encoder_spec,
        hidden_dim=128,
        lattice_stride=1,
        slots_l0=2,
        out_dim=3,
        rgb_activation='tanh',
        splat_block_h=16,
        splat_block_w=16,
        splat_background=-1.0,
        splat_opacity_scale=1.0,
        splat_opacity_min=0.0,
        splat_coord_order='xy',
        rgb_from_lr=True,
        # Parent-conditioned autoregressive Gaussian refinement.
        ar_layers=1,
        split_factor=2,
        ar_context_feat_dim=0,
        # Spatially varying topology-density controls.
        density_blend_init=0.15,
        init_low: float = 0.1,
        init_high: float = 0.3,
    ):
        super().__init__()
        self.encoder = models.make(encoder_spec)
        self.enc_dim = self.encoder.out_dim
        self.hidden_dim = int(hidden_dim)
        self.lattice_stride = int(lattice_stride)
        self.slots_l0 = int(slots_l0)
        self.out_dim = int(out_dim)
        self.rgb_activation = rgb_activation
        self.splat_block_h = int(splat_block_h)
        self.splat_block_w = int(splat_block_w)
        self.splat_background = float(splat_background)
        self.splat_opacity_scale = float(splat_opacity_scale)
        self.splat_opacity_min = float(splat_opacity_min)
        self.splat_coord_order = splat_coord_order
        self.rgb_from_lr = bool(rgb_from_lr)
        requested_ar_layers = int(ar_layers)
        requested_split_factor = int(split_factor)
        if requested_ar_layers < 1:
            raise ValueError('ar_layers must be at least 1.')
        if requested_split_factor < 2:
            raise ValueError('split_factor must be at least 2.')
        self.ar_layers = requested_ar_layers
        self.split_factor = requested_split_factor
        self.ar_context_feat_dim = max(0, int(ar_context_feat_dim))
        # Keep the Cholesky covariance numerically valid.
        self.rho_limit = 1.0 - 1e-3
        self.ar_color_feat_dim = max(self.out_dim, self.enc_dim // 8)
        self.coarse_alpha_logit_bias = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.density_blend_init = float(max(1e-4, min(1.0 - 1e-4, density_blend_init)))
        self.sigma_init_low = float(init_low)
        self.sigma_init_high = float(init_high)
        if not (0.0 < self.sigma_init_low < self.sigma_init_high < 1.0):
            raise ValueError('init_low and init_high must satisfy 0 < init_low < init_high < 1.')
        self.current_scale_norm = None

        mid_dim = max(self.enc_dim, 64)
        self.seed_reduce = nn.Sequential(
            nn.Conv2d(self.enc_dim, mid_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_dim, self.enc_dim, 3, padding=1),
            nn.GELU(),
        )

        self.scale_to_feat = nn.Sequential(
            nn.Linear(1, self.enc_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.enc_dim, self.enc_dim),
        )

        self.prior_stem = nn.Sequential(
            nn.Conv2d(self.enc_dim, self.hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.GELU(),
        )
        # Nonstationary density controls local Gaussian support.
        self.prior_refine = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.GELU(),
        )


        self.complexity_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.density_delta_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.prior_to_seed = nn.Conv2d(self.hidden_dim, self.enc_dim, 1)
        blend_hidden = max(16, self.hidden_dim // 2)
        self.density_blend_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim + 2, blend_hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(blend_hidden, 1, 1),
        )

        # Learnable scaffold opacity bounds.
        self.alpha_floor_center_logit = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.alpha_floor_radius_raw = nn.Parameter(torch.zeros((), dtype=torch.float32))

        # Learnable Gaussian support bounds.
        sigma_low_init = self.sigma_init_low
        sigma_high_init = self.sigma_init_high

        self.sigma_floor_low_param = nn.Parameter(torch.tensor(
            self._inverse_sigmoid_value(
                sigma_low_init, 0.0, 1.0
            ),
            dtype=torch.float32,
        ))
        sigma_range_frac = (sigma_high_init - sigma_low_init) / (1.0 - sigma_low_init)
        self.sigma_floor_range_param = nn.Parameter(torch.tensor(
            self._inverse_sigmoid_value(sigma_range_frac, 0.0, 1.0),
            dtype=torch.float32,
        ))

        # Topology heads: edge, node, link, and orientation.
        self.topology_node_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.topology_edge_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.topology_link_head = nn.Conv2d(self.hidden_dim, 1, 1)
        self.orientation_head = nn.Conv2d(self.hidden_dim, 2, 1)

        self.topology_prior_gain = nn.Parameter(torch.zeros(1))
        self.topology_density_gain = nn.Parameter(torch.zeros(1))
        self.topology_sigma_gain = nn.Parameter(torch.zeros(1))
        self.orientation_rho_gain = nn.Parameter(torch.zeros(1))
        self.topology_gate_gain = nn.Parameter(torch.zeros(1))

        # Gaussian parameter prediction and autoregressive refinement.
        self.base_offset_head = nn.Conv2d(self.enc_dim, self.slots_l0 * 2, 1)
        self.base_cov_head = nn.Conv2d(self.enc_dim, self.slots_l0 * 3, 1)
        self.base_alpha_head = nn.Conv2d(self.enc_dim, self.slots_l0, 1)
        self.base_rgb_head = nn.Conv2d(self.enc_dim, self.slots_l0 * self.out_dim, 1)
        self.base_ar_context_head = nn.Conv2d(self.enc_dim, self.slots_l0 * self.ar_context_feat_dim, 1) if self.ar_context_feat_dim > 0 else None

        attr_dim = 2 + 2 + 1 + 1 + self.out_dim + self.ar_context_feat_dim
        self.gaussian_attr_embed = nn.Sequential(
            nn.Linear(attr_dim, self.enc_dim),
            nn.GELU(),
            nn.Linear(self.enc_dim, self.enc_dim),
        )
        self.ar_graph_blocks = nn.ModuleList([
            LocalTopologyGraphBlock(self.enc_dim)
            for _ in range(self.ar_layers)
        ])
        self.ar_parent_feat_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.enc_dim, self.enc_dim),
                nn.GELU(),
                nn.Linear(self.enc_dim, self.enc_dim),
            )
            for _ in range(self.ar_layers)
        ])

        # Child token generation and sibling interaction.
        self.ar_child_token_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.enc_dim, self.enc_dim),
                nn.GELU(),
                nn.Linear(self.enc_dim, self.split_factor * self.enc_dim),
            )
            for _ in range(self.ar_layers)
        ])
        self.ar_child_topology_blocks = nn.ModuleList([
            LocalTopologyGraphBlock(self.enc_dim)
            for _ in range(self.ar_layers)
        ])
        self.ar_sibling_blocks = nn.ModuleList([
            SiblingInteractionBlock(self.enc_dim)
            for _ in range(self.ar_layers)
        ])

        child_attr_dim = (
            2 + 3
            + self.ar_color_feat_dim
            + self.ar_context_feat_dim
        )
        self.ar_child_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.enc_dim, self.enc_dim),
                nn.GELU(),
                nn.Linear(self.enc_dim, child_attr_dim),
            )
            for _ in range(self.ar_layers)
        ])

        # Parent-relative covariance refinement.
        covariance_condition_dim = self.enc_dim + 6
        self.ar_cov_relative_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(covariance_condition_dim, self.enc_dim),
                nn.GELU(),
                nn.Linear(self.enc_dim, 3),
            )
            for _ in range(self.ar_layers)
        ])
        self.ar_cov_inherit_gate_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(covariance_condition_dim, self.enc_dim),
                nn.GELU(),
                nn.Linear(self.enc_dim, 1),
            )
            for _ in range(self.ar_layers)
        ])

        self.ar_alpha_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.enc_dim, self.enc_dim),
                nn.GELU(),
                nn.Linear(self.enc_dim, 1),
            )
            for _ in range(self.ar_layers)
        ])

        # Decode opacity and branch continuation independently.
        continuation_in_dim = self.enc_dim + 3
        self.ar_continuation_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(continuation_in_dim, self.enc_dim),
                nn.GELU(),
                nn.Linear(self.enc_dim, 1),
            )
            for _ in range(max(0, self.ar_layers - 1))
        ])

        self.ar_rgb_decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.enc_dim + self.ar_color_feat_dim, self.enc_dim),
                nn.GELU(),
                nn.Linear(self.enc_dim, self.out_dim),
            )
            for _ in range(self.ar_layers)
        ])

        # Spatially varying appearance-residual gain.
        self.ar_rgb_energy_gain_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.enc_dim + self.ar_color_feat_dim, self.enc_dim),
                nn.GELU(),
                nn.Linear(self.enc_dim, 1),
            )
            for _ in range(self.ar_layers)
        ])

        self.seed_gate_head = nn.Conv2d(self.enc_dim, 1, 1)
        self.seed_temperature_param = nn.Parameter(torch.tensor(
            math.log(math.expm1(1.0)), dtype=torch.float32
        ))

        release_init = 1.0 / float(max(2, self.enc_dim))
        release_logit = math.log(release_init / (1.0 - release_init))
        release_tensor = torch.full(
            (self.ar_layers,), release_logit, dtype=torch.float32
        )

        self.ar_offset_layer_scale_raw = nn.Parameter(torch.zeros(self.ar_layers))
        self.ar_child_topology_release_logit = nn.Parameter(release_tensor.clone())
        self.ar_sibling_release_logit = nn.Parameter(release_tensor.clone())

        self.offset_gain_l0 = nn.Parameter(torch.zeros(1))
        self.rgb_residual_gain = nn.Parameter(torch.zeros(1))
        self.structure_residual_gain = nn.Parameter(torch.zeros(1))
        self.structure_gain = nn.Parameter(torch.zeros(1))
        self.complexity_bias = nn.Parameter(torch.zeros(1))


        self._init_stable_heads()

        self.inp = None
        self.feat = None
        self.seed_feat = None
        self.gauss_layer_mu = None
        self.gauss_layer_sigma = None
        self.gauss_layer_rho = None
        self.gauss_layer_alpha = None
        self.gauss_layer_rgb = None
        self.gauss_layer_rgb_residual = None

        self.complexity_map = None
        self.structure_complexity_map = None
        self.density_raw_map = None
        self.density_blend_map = None
        self.density_map = None
        self.scaffold_score_map = None
        self.seed_prob_map = None
        self.alpha_floor_map = None
        self.sigma_floor_prior_map = None
        self.topology_node_map = None
        self.topology_edge_map = None
        self.topology_link_map = None
        self.topology_importance_map = None
        self.orientation_map = None



    def _zero_conv(self, module):
        if module is None:
            return
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Sequential):
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    @staticmethod
    def _inverse_sigmoid_value(value, lo=0.0, hi=1.0):
        value = float(value)
        lo = float(lo)
        hi = float(hi)
        if hi <= lo:
            raise ValueError('hi must be larger than lo')
        x = (value - lo) / (hi - lo)
        x = max(1e-4, min(1.0 - 1e-4, x))
        return math.log(x / (1.0 - x))

    def _bounded_scalar(self, param, lo, hi):
        return float(lo) + (float(hi) - float(lo)) * torch.sigmoid(param)

    def _adaptive_bounds_low_high(self, low_param, range_param, lo, hi):
        low = self._bounded_scalar(low_param, lo, hi)
        high = low + (float(hi) - low) * torch.sigmoid(range_param)
        return low, high

    def _init_stable_heads(self):
        for m in [
            self.prior_to_seed,
            self.density_delta_head,
            self.base_offset_head, self.base_cov_head, self.base_alpha_head,
            self.base_rgb_head, self.base_ar_context_head,
            self.seed_gate_head,
        ]:
            self._zero_conv(m)

        if getattr(self, 'density_blend_head', None) is not None:
            last = self.density_blend_head[-1]
            if isinstance(last, nn.Conv2d):
                nn.init.zeros_(last.weight)
                if last.bias is not None:
                    nn.init.constant_(last.bias, self._safe_logit(torch.tensor(self.density_blend_init)).item())

        for decoder in getattr(self, 'ar_child_decoders', []):
            if isinstance(decoder[-1], nn.Linear):
                nn.init.zeros_(decoder[-1].weight)
                if decoder[-1].bias is not None:
                    nn.init.zeros_(decoder[-1].bias)
        alpha_init = 0.5 / float(max(1, self.split_factor))
        alpha_init_logit = math.log(alpha_init / (1.0 - alpha_init))
        for head in getattr(self, 'ar_alpha_heads', []):
            if isinstance(head[-1], nn.Linear):
                nn.init.zeros_(head[-1].weight)
                if head[-1].bias is not None:
                    nn.init.constant_(head[-1].bias, alpha_init_logit)
        for head in getattr(self, 'ar_continuation_heads', []):
            if isinstance(head[-1], nn.Linear):
                nn.init.zeros_(head[-1].weight)
                if head[-1].bias is not None:
                    nn.init.zeros_(head[-1].bias)
        for head in getattr(self, 'ar_cov_relative_heads', []):
            if isinstance(head[-1], nn.Linear):
                nn.init.zeros_(head[-1].weight)
                if head[-1].bias is not None:
                    nn.init.zeros_(head[-1].bias)
        for head in getattr(self, 'ar_cov_inherit_gate_heads', []):
            if isinstance(head[-1], nn.Linear):
                nn.init.zeros_(head[-1].weight)
                if head[-1].bias is not None:
                    nn.init.zeros_(head[-1].bias)

        for decoder in getattr(self, 'ar_rgb_decoders', []):
            if isinstance(decoder[-1], nn.Linear):
                nn.init.zeros_(decoder[-1].weight)
                if decoder[-1].bias is not None:
                    nn.init.zeros_(decoder[-1].bias)
        for head in getattr(self, 'ar_rgb_energy_gain_heads', []):
            if isinstance(head[-1], nn.Linear):
                nn.init.zeros_(head[-1].weight)
                if head[-1].bias is not None:
                    nn.init.zeros_(head[-1].bias)
    def _parse_encoder_out(self, out):
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    def _seed_lattice_pool(self, feat):
        b, c, h, w = feat.shape
        s = self.lattice_stride
        pad_h = (s - h % s) % s
        pad_w = (s - w % s) % s
        if pad_h > 0 or pad_w > 0:
            feat = F.pad(feat, (0, pad_w, 0, pad_h), mode='replicate')
        feat = F.avg_pool2d(feat, kernel_size=s, stride=s)
        return feat, h, w

    def _scale_to_tensor(self, scale, batch_size, device, dtype):
        if scale is None:
            return torch.ones(batch_size, 1, device=device, dtype=dtype)
        if not torch.is_tensor(scale):
            scale = torch.tensor(scale, device=device, dtype=dtype)
        else:
            scale = scale.to(device=device, dtype=dtype)

        if scale.dim() == 0:
            scale = scale.view(1, 1).expand(batch_size, 1)
        elif scale.dim() == 1:
            if scale.numel() == 1:
                scale = scale.view(1, 1).expand(batch_size, 1)
            elif scale.numel() == batch_size:
                scale = scale.view(batch_size, 1)
            else:
                scale = scale.reshape(-1)[0].view(1, 1).expand(batch_size, 1)
        else:
            if scale.shape[-1] == 2:
                scale = scale.reshape(batch_size, -1, 2).mean(dim=(1, 2), keepdim=False).view(batch_size, 1)
            else:
                scale = scale.reshape(batch_size, -1).mean(dim=1, keepdim=True)
        return scale.clamp_min(1e-6)

    def _scale_modify_to_tensor(self, scale_modify, scale, batch_size, device, dtype):
        """Return exact H/W scale as [B,2].  Falls back to scalar scale."""
        if scale_modify is None:
            s = self._scale_to_tensor(scale, batch_size, device, dtype)
            return s.expand(batch_size, 2).contiguous()
        if not torch.is_tensor(scale_modify):
            scale_modify = torch.tensor(scale_modify, device=device, dtype=dtype)
        else:
            scale_modify = scale_modify.to(device=device, dtype=dtype)
        if scale_modify.dim() == 0:
            scale_modify = scale_modify.view(1, 1).expand(batch_size, 2)
        elif scale_modify.dim() == 1:
            if scale_modify.numel() == 1:
                scale_modify = scale_modify.view(1, 1).expand(batch_size, 2)
            elif scale_modify.numel() == 2:
                scale_modify = scale_modify.view(1, 2).expand(batch_size, 2)
            elif scale_modify.numel() == batch_size:
                scale_modify = scale_modify.view(batch_size, 1).expand(batch_size, 2)
            else:
                scale_modify = scale_modify.reshape(-1, 2)[:batch_size]
        else:
            scale_modify = scale_modify.reshape(batch_size, -1)
            if scale_modify.shape[1] == 1:
                scale_modify = scale_modify.expand(batch_size, 2)
            else:
                scale_modify = scale_modify[:, :2]
        return scale_modify.clamp_min(1e-6)

    def _gt_size_to_hw(self, gt_size):
        if gt_size is None:
            return None
        if torch.is_tensor(gt_size):
            g = gt_size.detach().reshape(-1)
            if g.numel() >= 2:
                return int(g[0].item()), int(g[1].item())
        if isinstance(gt_size, (list, tuple)) and len(gt_size) >= 2:
            return int(gt_size[0]), int(gt_size[1])
        return None

    def _activate_rgb(self, x):
        if self.rgb_activation == 'sigmoid':
            return torch.sigmoid(x)
        if self.rgb_activation == 'identity':
            return x
        if self.rgb_activation == 'tanh':
            return torch.tanh(x)
        raise ValueError(f'Unsupported rgb_activation: {self.rgb_activation}')

    def _spatial_center_sigmoid(self, logits):
        centered = logits - logits.mean(dim=(2, 3), keepdim=True)
        return torch.sigmoid(centered)

    def _normalize_spatial_logits(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (x - mean) / std

    def _make_observed_structure_complexity(self, out_hw):
        if self.inp is None:
            return None, None

        gray = self.inp.mean(dim=1, keepdim=True)
        dy = gray[:, :, 1:, :] - gray[:, :, :-1, :]
        dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
        dx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
        dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
        grad = torch.sqrt(dx * dx + dy * dy + 1e-12)

        gray_pad = F.pad(gray, (1, 1, 1, 1), mode='replicate')
        local_mean = F.avg_pool2d(gray_pad, kernel_size=3, stride=1, padding=0)
        high = (gray - local_mean).abs()

        energy = grad + high
        if energy.shape[-2:] != tuple(out_hw):
            energy = F.interpolate(energy, size=out_hw, mode='bilinear', align_corners=False)

        structure_logits = self._normalize_spatial_logits(energy)
        structure_complexity = self._spatial_center_sigmoid(structure_logits)
        return structure_complexity, structure_logits


    def _safe_logit(self, x, eps=1e-4):
        x = x.clamp(eps, 1.0 - eps)
        return torch.log(x / (1.0 - x))

    def _center_map(self, x):
        return x - x.mean(dim=(2, 3), keepdim=True)

    def _make_scaffold_score(self, density, topology_importance):
        if topology_importance is None:
            return density.clamp(0.0, 1.0)
        return (1.0 - (1.0 - density.clamp(0.0, 1.0)) *
                (1.0 - topology_importance.clamp(0.0, 1.0))).clamp(0.0, 1.0)

    def _scale_norm_like(self, ref):
        scale_norm = self.current_scale_norm
        if scale_norm is None:
            scale_norm = ref.new_ones(ref.shape[0], *([1] * (ref.dim() - 1)))
        else:
            scale_norm = scale_norm.to(device=ref.device, dtype=ref.dtype)
            while scale_norm.dim() < ref.dim():
                scale_norm = scale_norm.unsqueeze(1)
        return scale_norm

    def _alpha_floor_bounds(self):
        radius = F.softplus(self.alpha_floor_radius_raw)
        low = torch.sigmoid(self.alpha_floor_center_logit - radius)
        high = torch.sigmoid(self.alpha_floor_center_logit + radius)
        return low, high

    def _alpha_floor_radius(self):
        return F.softplus(self.alpha_floor_radius_raw)

    def _alpha_floor_from_scaffold(self, scaffold):
        low, high = self._alpha_floor_bounds()
        scaffold = scaffold.clamp(0.0, 1.0)
        return high * (1.0 - scaffold) + low * scaffold

    def _sigma_floor_from_scaffold(self, scaffold):
        low, high = self._adaptive_bounds_low_high(
            self.sigma_floor_low_param,
            self.sigma_floor_range_param,
            0.0,
            1.0,
        )
        s_norm = self._scale_norm_like(scaffold).clamp(0.0, 1.0)
        scaffold = scaffold.clamp(0.0, 1.0)
        scale_base = high * (1.0 - s_norm) + low * s_norm
        return scale_base * (1.0 - scaffold) + low * scaffold

    def _make_observed_topology_cues(self, out_hw):
        if self.inp is None:
            return None

        gray = self.inp.mean(dim=1, keepdim=True)
        dtype = gray.dtype
        device = gray.device

        # Scharr gradients for local structure.
        kx = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]], device=device, dtype=dtype).view(1, 1, 3, 3) / 16.0
        ky = kx.transpose(-1, -2)
        dx = F.conv2d(gray, kx, padding=1)
        dy = F.conv2d(gray, ky, padding=1)
        grad = torch.sqrt(dx * dx + dy * dy + 1e-12)

        local_grad = F.avg_pool2d(grad, kernel_size=3, stride=1, padding=1)
        line = grad + (grad - local_grad).abs()

        jxx = F.avg_pool2d(dx * dx, kernel_size=3, stride=1, padding=1)
        jyy = F.avg_pool2d(dy * dy, kernel_size=3, stride=1, padding=1)
        jxy = F.avg_pool2d(dx * dy, kernel_size=3, stride=1, padding=1)
        det = jxx * jyy - jxy * jxy
        trace = jxx + jyy
        corner = F.relu(det - 0.04 * trace * trace)

        line_smooth = F.avg_pool2d(line, kernel_size=5, stride=1, padding=2)
        corner_smooth = F.avg_pool2d(corner, kernel_size=5, stride=1, padding=2)
        linking = F.relu(line_smooth - corner_smooth)

        # Tangent orientation follows the [y, x] convention.
        orient = torch.cat([dx, -dy], dim=1)
        orient = F.normalize(orient, dim=1, eps=1e-6)

        if line.shape[-2:] != tuple(out_hw):
            line = F.interpolate(line, size=out_hw, mode='bilinear', align_corners=False)
            corner = F.interpolate(corner, size=out_hw, mode='bilinear', align_corners=False)
            linking = F.interpolate(linking, size=out_hw, mode='bilinear', align_corners=False)
            orient = F.interpolate(orient, size=out_hw, mode='bilinear', align_corners=False)
            orient = F.normalize(orient, dim=1, eps=1e-6)

        return {
            'line_logits': self._normalize_spatial_logits(line),
            'corner_logits': self._normalize_spatial_logits(corner),
            'linking_logits': self._normalize_spatial_logits(linking),
            'orientation': orient,
        }

    def _make_topology_primitive_prior(self, prior_feat, structure_logits, observed=None):
        b, _, h, w = prior_feat.shape
        zeros = prior_feat.new_zeros((b, 1, h, w))

        if observed is None:
            observed = self._make_observed_topology_cues((h, w))
        if observed is None:
            line_logits = structure_logits.detach()
            corner_logits = structure_logits.detach()
            linking_logits = structure_logits.detach()
            observed_orientation = prior_feat.new_zeros((b, 2, h, w))
        else:
            line_logits = observed['line_logits']
            corner_logits = observed['corner_logits']
            linking_logits = observed['linking_logits']
            observed_orientation = observed['orientation']

        topology_edge_logits = self.topology_edge_head(prior_feat) + line_logits
        topology_node_logits = self.topology_node_head(prior_feat) + corner_logits
        topology_link_logits = self.topology_link_head(prior_feat) + linking_logits

        topology_edge = torch.sigmoid(topology_edge_logits)
        topology_node = torch.sigmoid(topology_node_logits)
        topology_link = torch.sigmoid(topology_link_logits)

        topology_importance = 1.0 - (1.0 - topology_edge) * (1.0 - topology_node) * (1.0 - topology_link)
        topology_importance = topology_importance.clamp(0.0, 1.0)

        learned_orientation = self.orientation_head(prior_feat)
        orientation = F.normalize(learned_orientation + observed_orientation, dim=1, eps=1e-6)

        return {
            'topology_node': topology_node,
            'topology_edge': topology_edge,
            'topology_link': topology_link,
            'topology_importance': topology_importance,
            'orientation': orientation,
        }

    def _make_deep_prior(self, seed_feat):
        stem = self.prior_stem(seed_feat)
        prior_feat = self.prior_refine(stem)

        observed_topology = self._make_observed_topology_cues(prior_feat.shape[-2:])

        raw_complexity = self.complexity_head(prior_feat)
        learned_logits = self._normalize_spatial_logits(raw_complexity)
        learned_complexity = torch.sigmoid(learned_logits)

        structure_complexity, structure_logits = self._make_observed_structure_complexity(seed_feat.shape[-2:])
        if structure_complexity is None:
            structure_complexity = learned_complexity.detach()
            structure_logits = learned_logits.detach()

        topology_prior = self._make_topology_primitive_prior(prior_feat, structure_logits, observed=observed_topology)

        topology_centered = self._center_map(topology_prior['topology_importance'])

        structure_strength = F.softplus(self.structure_gain)
        residual_strength = F.softplus(self.structure_residual_gain)
        fused_logits = (
            structure_strength * structure_logits
            + residual_strength * learned_logits
            + self.topology_prior_gain.view(1, 1, 1, 1) * topology_centered
            + self.complexity_bias.view(1, 1, 1, 1)
        )
        complexity = torch.sigmoid(fused_logits)
        density_logits = (
            fused_logits
            + self.density_delta_head(prior_feat)
            + self.topology_density_gain.view(1, 1, 1, 1) * topology_centered
        )
        density_raw = torch.sigmoid(density_logits)
        blend_in = torch.cat([
            prior_feat,
            structure_complexity,
            topology_prior['topology_importance'],
        ], dim=1)
        density_blend = torch.sigmoid(self.density_blend_head(blend_in))
        density = ((1.0 - density_blend) * density_raw + density_blend * structure_complexity).clamp(0.0, 1.0)
        scaffold_score = self._make_scaffold_score(density, topology_prior['topology_importance'])
        alpha_floor_map = self._alpha_floor_from_scaffold(scaffold_score)
        sigma_floor_map = self._sigma_floor_from_scaffold(scaffold_score)
        prior_seed = self.prior_to_seed(prior_feat)

        out = {
            'prior_feat': prior_feat,
            'prior_seed': prior_seed,
            'complexity': complexity,
            'structure_complexity': structure_complexity,
            'density_raw': density_raw,
            'density_blend': density_blend,
            'density': density,
            'scaffold_score': scaffold_score,
            'alpha_floor_map': alpha_floor_map,
            'alpha_floor_low': self._alpha_floor_bounds()[0].detach().view(1),
            'alpha_floor_high': self._alpha_floor_bounds()[1].detach().view(1),
            'alpha_floor_radius': self._alpha_floor_radius().detach().view(1),
            'sigma_floor_map': sigma_floor_map,
        }
        out.update(topology_prior)
        return out


    def _make_seed_probability(self, fused_feat, prior):
        scaffold = prior['scaffold_score'].clamp(0.0, 1.0)
        temp = F.softplus(self.seed_temperature_param).view(1, 1, 1, 1).clamp_min(1e-4)
        residual = torch.tanh(self.seed_gate_head(fused_feat) / temp)
        seed_prob = torch.sigmoid(self._safe_logit(scaffold) + residual)
        return seed_prob.clamp(0.0, 1.0)

    def _make_covariance_from_cholesky(
        self,
        raw_chol,
        step,
        density_seed,
        topology_seed=None,
        orientation_seed=None,
    ):

        step_y = step[..., 0:1]
        step_x = step[..., 1:2]
        base_complexity = density_seed

        raw_sy = raw_chol[..., 0:1]
        raw_rho = raw_chol[..., 1]
        raw_sx = raw_chol[..., 2:3]

        detail_logits = self._safe_logit(base_complexity)
        if topology_seed is not None:
            detail_logits = detail_logits + self.topology_sigma_gain.view(1, 1, 1, 1) * topology_seed
        detail = torch.sigmoid(detail_logits)

        # Maintain a minimum Gaussian size for stable rendering.
        sigma_floor = self._sigma_floor_from_scaffold(detail).to(dtype=raw_sy.dtype, device=raw_sy.device)
        detail_shrink = (1.0 - detail).clamp_min(1e-6)
        sy = step_y * (sigma_floor + (1.0 - sigma_floor) * torch.sigmoid(raw_sy) * detail_shrink)
        sx = step_x * (sigma_floor + (1.0 - sigma_floor) * torch.sigmoid(raw_sx) * detail_shrink)

        rho_logits = raw_rho
        if orientation_seed is not None:
            oy = orientation_seed[..., 0]
            ox = orientation_seed[..., 1]
            diagonal_orientation = 2.0 * oy * ox
            rho_logits = rho_logits + self.orientation_rho_gain.view(1, 1, 1) * diagonal_orientation
        rho = (
            self.rho_limit
            * torch.tanh(rho_logits)
            * torch.sqrt(detail.squeeze(-1).clamp_min(1e-6))
        )

        sigma = torch.cat([sy.clamp_min(1e-6), sx.clamp_min(1e-6)], dim=-1)
        return sigma, rho

    def _sample_lr_rgb_at_mu(self, mu):
        b = mu.shape[0]
        leading = mu.shape[:-1]
        grid = mu.contiguous().view(b, -1, 1, 2)[..., [1, 0]]
        sampled = F.grid_sample(
            self.inp, grid, mode='bilinear', padding_mode='border', align_corners=False
        )
        sampled = sampled[:, :, :, 0].permute(0, 2, 1).contiguous()
        sampled = sampled.view(*leading, sampled.shape[-1])
        return sampled[..., :self.out_dim]


    def _expand_seed_map(self, x, b, n, slots, extra_dim=None):
        """Flatten [B,C,H,W] maps to [B,N,slots,C] or [B,N,slots,extra]."""
        x = x.permute(0, 2, 3, 1).contiguous().view(b, n, 1, x.shape[1])
        return x.expand(b, n, slots, x.shape[-1])

    def _num_child_slots(self):
        if self.ar_layers <= 0:
            return 0
        total = 0
        parents = max(1, int(self.slots_l0))
        split_factor = max(1, int(self.split_factor))
        for _ in range(int(self.ar_layers)):
            parents *= split_factor
            total += parents
        return int(total)

    def _num_total_slots(self):
        """Return seed + all autoregressive child candidates per seed."""
        return int(max(1, int(self.slots_l0)) + self._num_child_slots())

    def _split_mlp_output(self, raw, parent_shape):
        del parent_shape
        q = 0
        offset = raw[..., q:q + 2]
        q += 2
        cov_raw = raw[..., q:q + 3]
        q += 3
        color_feat_raw = raw[..., q:q + self.ar_color_feat_dim]
        q += self.ar_color_feat_dim
        if self.ar_context_feat_dim > 0:
            ar_context_raw = raw[..., q:q + self.ar_context_feat_dim]
        else:
            ar_context_raw = raw.new_zeros(raw.shape[:-1] + (0,))
        return {
            'offset': offset,
            'cov_raw': cov_raw,
            'color_feat_raw': color_feat_raw,
            'ar_context_raw': ar_context_raw,
        }

    def _make_feature_decoded_child_covariance(
        self,
        cov_raw,
        step_ar,
        density_seed,
        topology_seed_centered,
        orientation_seed,
        layer_idx,
    ):
        density_child = density_seed.unsqueeze(3)
        topology_child = topology_seed_centered.unsqueeze(3)
        orientation_child = orientation_seed.unsqueeze(3)
        child_sigma, child_rho = self._make_covariance_from_cholesky(
            cov_raw,
            step_ar,
            density_child,
            topology_seed=topology_child,
            orientation_seed=orientation_child,
        )
        if int(layer_idx) > 0:
            layer_scale = float(self.split_factor) ** (-0.5 * float(layer_idx))
            child_sigma = child_sigma * layer_scale
        return child_sigma, child_rho

    def _make_parent_relative_child_covariance(
        self,
        sigma_absolute,
        rho_absolute,
        parent_sigma,
        parent_rho,
        child_feature,
        structure_seed,
        topology_seed,
        density_seed,
        scaffold_seed,
        orientation_seed,
        layer_idx,
    ):
        b, n, parent_slots, child_slots, _ = child_feature.shape

        def expand_scalar(seed):
            return seed.expand(
                b, n, parent_slots, child_slots
            ).unsqueeze(-1)

        orientation_child = orientation_seed.unsqueeze(3).expand(
            b, n, parent_slots, child_slots, 2
        )
        structure_child = expand_scalar(structure_seed)
        topology_child = expand_scalar(topology_seed)
        density_child = expand_scalar(density_seed)
        scaffold_child = expand_scalar(scaffold_seed)

        structural = torch.cat([
            structure_child,
            topology_child,
            density_child,
            scaffold_child,
            orientation_child,
        ], dim=-1)
        condition = torch.cat([child_feature, structural], dim=-1)

        relative_raw = self.ar_cov_relative_heads[layer_idx](condition)
        inherit_delta = torch.tanh(
            self.ar_cov_inherit_gate_heads[layer_idx](condition)
        )

        # Prevent reconstruction gradients from altering the detached structure prior.
        detail_prior = torch.sqrt(
            structure_child.detach().clamp_min(1e-6)
            * topology_child.detach().clamp_min(1e-6)
        ).clamp(0.0, 1.0)

        contraction_delta = torch.tanh(relative_raw[..., 0:1])
        contraction = (
            detail_prior
            + detail_prior * (1.0 - detail_prior) * contraction_delta
        ).clamp(0.0, 1.0)

        gate = (
            detail_prior
            + detail_prior * (1.0 - detail_prior) * inherit_delta
        ).clamp(0.0, 1.0)

        log_k = math.log(float(max(1, self.split_factor)))
        base_decay = 0.5 * contraction * log_k
        anisotropy_signal = torch.tanh(relative_raw[..., 1:2])
        anisotropy = base_decay * anisotropy_signal

        # Sigma axes follow [sy, sx] and contract relative to the parent.
        ratio_y = torch.exp(-base_decay + anisotropy)
        ratio_x = torch.exp(-base_decay - anisotropy)
        sigma_ratio_relative = torch.cat([ratio_y, ratio_x], dim=-1)

        parent_sigma_e = parent_sigma.unsqueeze(3).clamp_min(1e-6)
        sigma_relative = parent_sigma_e * sigma_ratio_relative

        parent_rho_e = parent_rho.unsqueeze(3)
        rho_unit = (
            parent_rho_e / float(self.rho_limit)
        ).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        parent_fisher = torch.atanh(rho_unit)
        delta_fisher = torch.tanh(relative_raw[..., 2])
        rho_relative = float(self.rho_limit) * torch.tanh(
            parent_fisher + delta_fisher
        )

        effective_contraction = (gate * contraction).clamp(0.0, 1.0)
        effective_base_decay = 0.5 * effective_contraction * log_k
        effective_anisotropy = (
            effective_base_decay * anisotropy_signal
        )
        final_ratio_y = torch.exp(
            -effective_base_decay + effective_anisotropy
        )
        final_ratio_x = torch.exp(
            -effective_base_decay - effective_anisotropy
        )
        sigma_ratio_final = torch.cat(
            [final_ratio_y, final_ratio_x], dim=-1
        )
        sigma_final = parent_sigma_e * sigma_ratio_final

        rho_gate = gate.squeeze(-1)
        rho_final = (
            (1.0 - rho_gate) * rho_absolute
            + rho_gate * rho_relative
        ).clamp(-self.rho_limit, self.rho_limit)

        return (
            sigma_final,
            rho_final,
            gate,
            sigma_relative,
            rho_relative,
            detail_prior,
            effective_contraction,
            anisotropy_signal,
            sigma_ratio_relative,
        )

    @staticmethod
    def _make_child_alpha(alpha_logit, ancestor_continuation):
        """Decode current-layer opacity independently of continuation."""
        alpha_raw = torch.sigmoid(alpha_logit)
        return (
            alpha_raw * ancestor_continuation.unsqueeze(3)
        ).clamp(0.0, 1.0)

    @staticmethod
    def _make_continuation_gate(
        continuation_delta,
        seed_prior,
        ancestor_continuation,
    ):
        eps = torch.finfo(continuation_delta.dtype).eps
        prior = seed_prior.clamp(eps, 1.0 - eps).expand_as(continuation_delta)
        local = torch.sigmoid(torch.logit(prior) + continuation_delta)
        return ancestor_continuation.unsqueeze(3) * local

    def _apply_child_topology_inheritance(
        self,
        child_feat,
        topology_block,
        topology_importance,
        topology_link,
        density,
        orientation,
        height,
        width,
    ):
        b, n, p, k, c = child_feat.shape
        if n != int(height) * int(width):
            raise ValueError(
                f'Child topology reshape mismatch: n={n}, HxW={height}x{width}.'
            )
        lineage_count = p * k
        x = child_feat.view(b, height, width, p, k, c)
        x = x.permute(0, 3, 4, 5, 1, 2).contiguous()
        x = x.view(b * lineage_count, c, height, width)

        def repeat_map(value):
            if value is None:
                return None
            return (
                value.unsqueeze(1)
                .expand(b, lineage_count, *value.shape[1:])
                .contiguous()
                .view(b * lineage_count, *value.shape[1:])
            )

        y = topology_block(
            x,
            repeat_map(topology_importance),
            repeat_map(topology_link),
            repeat_map(density),
            repeat_map(orientation),
        )
        y = y.view(b, p, k, c, height, width)
        y = y.permute(0, 4, 5, 1, 2, 3).contiguous()
        return y.view(b, n, p, k, c)

    @staticmethod
    def _make_topology_guided_child_offset(
        raw_offset,
        step,
        layer_gain,
        orientation,
        topology_importance,
    ):
        free_unit = torch.tanh(raw_offset)

        orientation_norm = torch.linalg.vector_norm(
            orientation, dim=-1, keepdim=True
        )
        tangent = orientation / orientation_norm.clamp_min(1e-6)

        parallel_unit = (free_unit * tangent).sum(
            dim=-1, keepdim=True
        ) * tangent
        normal_unit = free_unit - parallel_unit

        direction_validity = orientation_norm.clamp(0.0, 1.0)
        guide = topology_importance.clamp(0.0, 1.0) * direction_validity
        guided_unit = parallel_unit + (1.0 - guide) * normal_unit

        scale = step * layer_gain
        return guided_unit * scale

    def _embed_gaussian_attrs(self, mu, sigma, rho, alpha, rgb, ar_context_feat):
        if ar_context_feat is None:
            ar_context_feat = rgb.new_zeros(rgb.shape[:-1] + (self.ar_context_feat_dim,))
        attr = torch.cat([
            mu,
            sigma,
            rho.unsqueeze(-1),
            alpha.unsqueeze(-1),
            rgb,
            ar_context_feat,
        ], dim=-1)
        return self.gaussian_attr_embed(attr)

    def _predict_ar_gaussians(self, seed_feat, feat_h, feat_w):
        b, c, ha, wa = seed_feat.shape
        n = ha * wa
        device = seed_feat.device

        prior = self._make_deep_prior(seed_feat)
        fused_feat = seed_feat + prior['prior_seed']
        seed_prob_map = self._make_seed_probability(fused_feat, prior)

        seed_pos0 = make_coord((ha, wa)).to(device).unsqueeze(0).expand(b, -1, -1)
        step_y = 2.0 * self.lattice_stride / float(feat_h)
        step_x = 2.0 * self.lattice_stride / float(feat_w)
        step = seed_pos0.new_tensor([step_y, step_x]).view(1, 1, 1, 2)
        step_ar = seed_pos0.new_tensor([step_y, step_x]).view(1, 1, 1, 1, 2)

        density_seed = prior['density'].permute(0, 2, 3, 1).contiguous().view(b, n, 1, 1)
        structure_seed = prior['structure_complexity'].permute(0, 2, 3, 1).contiguous().view(b, n, 1, 1)
        topology_seed = prior['topology_importance'].permute(0, 2, 3, 1).contiguous().view(b, n, 1, 1)
        scaffold_seed = prior['scaffold_score'].permute(0, 2, 3, 1).contiguous().view(b, n, 1, 1)
        seed_gate = seed_prob_map.permute(0, 2, 3, 1).contiguous().view(b, n, 1, 1)
        topology_seed_c = topology_seed - topology_seed.mean(dim=1, keepdim=True)
        orientation_seed = prior['orientation'].permute(0, 2, 3, 1).contiguous().view(b, n, 1, 2)

        # Base Gaussian layer G0.
        base_slots = max(1, self.slots_l0)
        offset0 = _reshape_head(self.base_offset_head(fused_feat), base_slots, 2)
        cov0_raw = _reshape_head(self.base_cov_head(fused_feat), base_slots, 3)
        alpha0_raw = _reshape_head(self.base_alpha_head(fused_feat), base_slots, 1).squeeze(-1)
        rgb0_raw = _reshape_head(self.base_rgb_head(fused_feat), base_slots, self.out_dim)
        if self.base_ar_context_head is not None:
            context0_raw = _reshape_head(self.base_ar_context_head(fused_feat), base_slots, self.ar_context_feat_dim)
        else:
            context0_raw = rgb0_raw.new_zeros(b, n, base_slots, 0)

        offset0_gain = F.softplus(self.offset_gain_l0) + 1e-6
        offset0 = torch.tanh(offset0) * step * offset0_gain
        sigma0, rho0 = self._make_covariance_from_cholesky(
            cov0_raw, step, density_seed,
            topology_seed=topology_seed_c,
            orientation_seed=orientation_seed,
        )
        # Coarse-layer opacity retains structure-dependent coverage.
        alpha0_learned = torch.sigmoid(alpha0_raw + self.coarse_alpha_logit_bias)
        alpha0_density = 0.5 + 0.5 * density_seed.squeeze(-1)
        alpha_floor_seed = self._alpha_floor_from_scaffold(scaffold_seed).squeeze(-1)
        alpha0 = (
            alpha_floor_seed
            + (1.0 - alpha_floor_seed) * alpha0_learned * alpha0_density
        ).clamp(0.0, 1.0)
        mu0 = seed_pos0.unsqueeze(2) + offset0

        if self.rgb_from_lr and self.inp is not None:
            base_rgb0 = self._sample_lr_rgb_at_mu(mu0)
            res_gain = torch.sigmoid(self.rgb_residual_gain)
            rgb0 = (base_rgb0 + res_gain * self._activate_rgb(rgb0_raw)).clamp(-1.0, 1.0)
        else:
            rgb0 = self._activate_rgb(rgb0_raw)
        context0 = torch.tanh(context0_raw) if self.ar_context_feat_dim > 0 else context0_raw

        # G0 renders absolute RGB; finer layers render RGB residuals.
        layer_mu = [mu0]
        layer_sigma = [sigma0]
        layer_rho = [rho0]
        layer_alpha = [alpha0]
        layer_rgb = [rgb0]
        layer_rgb_residual = []

        parent_mu, parent_sigma, parent_rho = mu0, sigma0, rho0
        parent_alpha, parent_rgb, parent_context = alpha0, rgb0, context0
        parent_feat = self._expand_seed_map(fused_feat, b, n, base_slots)

        # Inherited continuation controls child-layer contribution.
        parent_continuation = alpha0.new_ones(b, n, base_slots)

        graph_feat = fused_feat

        for layer_idx in range(self.ar_layers):
            graph_feat = self.ar_graph_blocks[layer_idx](
                graph_feat,
                prior['topology_importance'],
                prior.get('topology_link'),
                prior['density'],
                prior['orientation'],
            )
            ctx = graph_feat.permute(0, 2, 3, 1).contiguous().view(
                b, n, 1, self.enc_dim
            )
            ctx = ctx.expand(b, n, parent_mu.shape[2], self.enc_dim)
            attr_embed = self._embed_gaussian_attrs(
                parent_mu,
                parent_sigma,
                parent_rho,
                parent_alpha,
                parent_rgb,
                parent_context,
            )
            parent_state = parent_feat + attr_embed
            h = parent_state + self.ar_parent_feat_proj[layer_idx](ctx)

            token_delta = self.ar_child_token_mlps[layer_idx](h)
            child_feat = token_delta.view(
                b,
                n,
                parent_mu.shape[2],
                self.split_factor,
                self.enc_dim,
            )
            child_feat = child_feat + h.unsqueeze(3)

            child_topology_candidate = self._apply_child_topology_inheritance(
                child_feat,
                self.ar_child_topology_blocks[layer_idx],
                prior['topology_importance'],
                prior.get('topology_link'),
                prior['density'],
                prior['orientation'],
                ha,
                wa,
            )
            topology_release = torch.sigmoid(
                self.ar_child_topology_release_logit[layer_idx]
            )
            child_feat = child_feat + topology_release * (
                child_topology_candidate - child_feat
            )

            sibling_candidate = self.ar_sibling_blocks[layer_idx](child_feat)
            sibling_release = torch.sigmoid(
                self.ar_sibling_release_logit[layer_idx]
            )
            child_feat = child_feat + sibling_release * (
                sibling_candidate - child_feat
            )

            decoded = self._split_mlp_output(
                self.ar_child_decoders[layer_idx](child_feat),
                parent_mu.shape[:3],
            )

            offset_layer_scale = (
                2.0 * torch.sigmoid(
                    self.ar_offset_layer_scale_raw[layer_idx]
                )
            ).clamp(max=1.0).view(1, 1, 1, 1, 1)
            orientation_ar = orientation_seed.unsqueeze(3)
            topology_guide_ar = topology_seed.unsqueeze(3)
            # Deeper offsets use the direct parent support as reference.
            if layer_idx == 0:
                offset_reference = step_ar
            else:
                offset_reference = (
                    parent_sigma.unsqueeze(3)
                    / math.sqrt(float(self.split_factor))
                )
            child_offset = self._make_topology_guided_child_offset(
                decoded['offset'],
                offset_reference,
                offset_layer_scale,
                orientation_ar,
                topology_guide_ar,
            )
            child_mu = parent_mu.unsqueeze(3) + child_offset

            child_sigma_absolute, child_rho_absolute = (
                self._make_feature_decoded_child_covariance(
                    decoded['cov_raw'],
                    step_ar,
                    density_seed,
                    topology_seed_c,
                    orientation_seed,
                    layer_idx,
                )
            )
            child_sigma, child_rho, *_ = self._make_parent_relative_child_covariance(
                child_sigma_absolute,
                child_rho_absolute,
                parent_sigma,
                parent_rho,
                child_feat,
                structure_seed,
                topology_seed,
                density_seed,
                scaffold_seed,
                orientation_seed,
                layer_idx,
            )

            alpha_logit = self.ar_alpha_heads[layer_idx](child_feat).squeeze(-1)
            child_alpha = self._make_child_alpha(
                alpha_logit,
                parent_continuation,
            )

            child_continuation = None
            if layer_idx < self.ar_layers - 1:
                parent_slots = parent_mu.shape[2]
                structural = torch.cat([
                    topology_seed.expand(
                        b, n, parent_slots, self.split_factor
                    ).unsqueeze(-1),
                    density_seed.expand(
                        b, n, parent_slots, self.split_factor
                    ).unsqueeze(-1),
                    scaffold_seed.expand(
                        b, n, parent_slots, self.split_factor
                    ).unsqueeze(-1),
                ], dim=-1)
                continuation_input = torch.cat(
                    [child_feat, structural], dim=-1
                )
                continuation_delta = self.ar_continuation_heads[layer_idx](
                    continuation_input
                ).squeeze(-1)
                continuation_prior = seed_gate.expand(
                    b, n, parent_slots, self.split_factor
                )
                child_continuation = self._make_continuation_gate(
                    continuation_delta,
                    continuation_prior,
                    parent_continuation,
                )

            # Child appearance inherits parent color and predicts local residuals.
            child_color_feat = torch.tanh(decoded['color_feat_raw'])
            rgb_decoder_input = torch.cat(
                [child_feat, child_color_feat], dim=-1
            )
            rgb_residual = self.ar_rgb_decoders[layer_idx](
                rgb_decoder_input
            )
            rgb_energy_gain_raw = self.ar_rgb_energy_gain_heads[layer_idx](
                rgb_decoder_input.detach()
            )
            rgb_energy_gain = F.softplus(rgb_energy_gain_raw) / math.log(2.0)
            parent_rgb_anchor = parent_rgb.unsqueeze(3)
            res_gain = torch.sigmoid(self.rgb_residual_gain)
            child_rgb_residual = (
                res_gain
                * rgb_energy_gain
                * self._activate_rgb(rgb_residual)
            )
            child_rgb = (
                parent_rgb_anchor + child_rgb_residual
            ).clamp(-1.0, 1.0)

            child_context = (
                torch.tanh(decoded['ar_context_raw'])
                if self.ar_context_feat_dim > 0
                else decoded['ar_context_raw']
            )

            child_mu_f = child_mu.view(b, n, -1, 2)
            child_sigma_f = child_sigma.view(b, n, -1, 2)
            child_rho_f = child_rho.view(b, n, -1)
            child_alpha_f = child_alpha.view(b, n, -1)
            child_rgb_f = child_rgb.view(b, n, -1, self.out_dim)
            child_rgb_residual_f = child_rgb_residual.view(
                b, n, -1, self.out_dim
            )
            if self.ar_context_feat_dim > 0:
                child_context_f = child_context.view(
                    b, n, -1, self.ar_context_feat_dim
                )
            else:
                child_context_f = child_rgb_f.new_zeros(
                    b, n, child_rgb_f.shape[2], 0
                )
            layer_mu.append(child_mu_f)
            layer_sigma.append(child_sigma_f)
            layer_rho.append(child_rho_f)
            layer_alpha.append(child_alpha_f)
            layer_rgb.append(child_rgb_f)
            layer_rgb_residual.append(child_rgb_residual_f)
            if child_continuation is not None:
                parent_continuation = child_continuation.view(b, n, -1)

            parent_mu, parent_sigma, parent_rho = (
                child_mu_f,
                child_sigma_f,
                child_rho_f,
            )
            parent_alpha, parent_rgb, parent_context = (
                child_alpha_f,
                child_rgb_f,
                child_context_f,
            )
            parent_feat = child_feat.view(b, n, -1, self.enc_dim)

        gauss_layer_mu = [value.reshape(b, -1, 2) for value in layer_mu]
        gauss_layer_sigma = [
            value.reshape(b, -1, 2) for value in layer_sigma
        ]
        gauss_layer_rho = [value.reshape(b, -1) for value in layer_rho]
        gauss_layer_alpha = [
            value.reshape(b, -1) for value in layer_alpha
        ]
        gauss_layer_rgb = [
            value.reshape(b, -1, self.out_dim) for value in layer_rgb
        ]
        gauss_layer_rgb_residual = [
            value.reshape(b, -1, self.out_dim)
            for value in layer_rgb_residual
        ]
        return {
            'gauss_layer_mu': gauss_layer_mu,
            'gauss_layer_sigma': gauss_layer_sigma,
            'gauss_layer_rho': gauss_layer_rho,
            'gauss_layer_alpha': gauss_layer_alpha,
            'gauss_layer_rgb': gauss_layer_rgb,
            'gauss_layer_rgb_residual': gauss_layer_rgb_residual,
            'complexity': prior['complexity'],
            'structure_complexity': prior['structure_complexity'],
            'density_raw': prior.get('density_raw'),
            'density_blend': prior.get('density_blend'),
            'density': prior['density'],
            'scaffold_score': prior.get('scaffold_score'),
            'seed_prob': seed_prob_map,
            'alpha_floor_map': prior.get('alpha_floor_map'),
            'sigma_floor_map': prior.get('sigma_floor_map'),
            'topology_node': prior['topology_node'],
            'topology_edge': prior['topology_edge'],
            'topology_link': prior.get('topology_link'),
            'topology_importance': prior['topology_importance'],
            'orientation': prior['orientation'],
        }

    def _predict_gaussian_params(self, seed_feat, feat_h, feat_w):
        return self._predict_ar_gaussians(seed_feat, feat_h, feat_w)

    def gen_feat(self, inp, scale=None, scale_modify=None):
        self.inp = inp
        feat = self._parse_encoder_out(self.encoder(inp))
        self.feat = feat

        seed_feat, feat_h, feat_w = self._seed_lattice_pool(feat)
        seed_feat = self.seed_reduce(seed_feat)

        # Use the realized height/width scales for conditioning.
        scale_modify_value = self._scale_modify_to_tensor(scale_modify, scale, inp.shape[0], inp.device, seed_feat.dtype)
        scale_value = scale_modify_value.mean(dim=1, keepdim=True)
        log_scale = torch.log(scale_value.clamp_min(1e-6))
        self.current_scale_norm = torch.sigmoid(log_scale).view(inp.shape[0], 1, 1, 1)
        scale_code = self.scale_to_feat(log_scale).view(inp.shape[0], self.enc_dim, 1, 1)
        seed_feat = seed_feat + scale_code
        self.seed_feat = seed_feat

        out = self._predict_gaussian_params(seed_feat, feat_h, feat_w)
        self.gauss_layer_mu = out['gauss_layer_mu']
        self.gauss_layer_sigma = out['gauss_layer_sigma']
        self.gauss_layer_rho = out['gauss_layer_rho']
        self.gauss_layer_alpha = out['gauss_layer_alpha']
        self.gauss_layer_rgb = out['gauss_layer_rgb']
        self.gauss_layer_rgb_residual = out['gauss_layer_rgb_residual']

        self.complexity_map = out['complexity']
        self.structure_complexity_map = out.get('structure_complexity')
        self.density_raw_map = out.get('density_raw')
        self.density_blend_map = out.get('density_blend')
        self.density_map = out['density']
        self.scaffold_score_map = out.get('scaffold_score')
        self.seed_prob_map = out.get('seed_prob')
        self.alpha_floor_map = out.get('alpha_floor_map')
        self.sigma_floor_prior_map = out.get('sigma_floor_map')
        self.topology_node_map = out['topology_node']
        self.topology_edge_map = out['topology_edge']
        self.topology_link_map = out.get('topology_link')
        self.topology_importance_map = out['topology_importance']
        self.orientation_map = out['orientation']
        return self.feat

    def _scale_to_hw(self, inp, scale, gt_size=None, scale_modify=None):
        # Prefer the target image size provided by the dataset.
        hw = self._gt_size_to_hw(gt_size)
        if hw is not None:
            return hw

        if scale_modify is not None:
            sm = self._scale_modify_to_tensor(scale_modify, scale, inp.shape[0], inp.device, inp.dtype)
            scale_h = float(sm[0, 0].detach().item())
            scale_w = float(sm[0, 1].detach().item())
        elif torch.is_tensor(scale):
            scale_detached = scale.detach()
            if scale_detached.dim() >= 2 and scale_detached.shape[-1] == 2:
                scale_h = float(scale_detached.reshape(-1, 2)[0, 0].item())
                scale_w = float(scale_detached.reshape(-1, 2)[0, 1].item())
            else:
                scale_h = scale_w = float(scale_detached.flatten()[0].item())
        else:
            scale_h = scale_w = float(scale)

        h, w = inp.shape[-2:]
        out_h = round(h * scale_h)
        out_w = round(w * scale_w)
        return int(out_h), int(out_w)

    def _sigma_rho_to_l_elements(self, sigma, rho, out_h, out_w):
        """Convert normalized [sigma_y, sigma_x, rho] to gsplat Cholesky in pixel units."""
        sy = sigma[:, 0].clamp_min(1e-4) * (float(out_h) / 2.0)
        sx = sigma[:, 1].clamp_min(1e-4) * (float(out_w) / 2.0)
        rho = rho.clamp(-0.999, 0.999)

        l_xx = sx
        l_xy = rho * sy
        l_yy = sy * torch.sqrt((1.0 - rho * rho).clamp_min(1e-4))
        return torch.stack([l_xx, l_xy, l_yy], dim=-1)

    def _render_splat_single(self, mu, sigma, rho, alpha, rgb, out_h, out_w, scale=None, scale_modify=None, background_value=None):
        if project_gaussians_2d is None or rasterize_gaussians_sum is None:
            raise ImportError(
                'NS-AGS splat rendering requires the ContinuousSR/GaussianImage gsplat package. '
                'Install the XingtongGe/gsplat package used by ContinuousSR first.'
            )

        # Convert model [y, x] coordinates to gsplat [x, y] centers.
        if self.splat_coord_order != 'xy':
            raise ValueError(
                'The unified pixel-center path requires splat_coord_order="xy". '
                'Internal Gaussian centers remain [y, x], while gsplat inputs '
                'are converted to corrected [x, y] coordinates at render time.'
            )
        means2d = normalized_yx_to_gsplat_xy(mu, out_h, out_w)

        l_elements = self._sigma_rho_to_l_elements(sigma, rho, out_h, out_w)
        opacity = alpha.unsqueeze(-1) * self.splat_opacity_scale
        if self.splat_opacity_min > 0:
            opacity = opacity.clamp_min(self.splat_opacity_min)
        opacity = opacity.clamp(0.0, 1.0)

        tile_bounds = (
            (out_w + self.splat_block_w - 1) // self.splat_block_w,
            (out_h + self.splat_block_h - 1) // self.splat_block_h,
            1,
        )
        xys, depths, radii, conics, num_tiles_hit = project_gaussians_2d(
            means2d, l_elements, out_h, out_w, tile_bounds
        )
        bg_value = self.splat_background if background_value is None else float(background_value)

        # Render attributes in three-channel chunks for RGB-only gsplat kernels.
        attr_dim = int(rgb.shape[-1])
        if attr_dim <= 0:
            raise ValueError('Gaussian attributes must have at least one channel.')

        rendered_chunks = []
        for start in range(0, attr_dim, 3):
            end = min(start + 3, attr_dim)
            chunk = rgb[:, start:end]
            valid_dim = int(chunk.shape[-1])
            if valid_dim < 3:
                pad = chunk.new_zeros((chunk.shape[0], 3 - valid_dim))
                chunk_rgb = torch.cat([chunk, pad], dim=-1)
            else:
                chunk_rgb = chunk

            background = chunk_rgb.new_full((3,), bg_value)
            out_chunk = rasterize_gaussians_sum(
                xys, depths, radii, conics, num_tiles_hit, chunk_rgb, opacity,
                out_h, out_w, self.splat_block_h, self.splat_block_w,
                background=background, return_alpha=False
            )
            rendered_chunks.append(out_chunk[..., :valid_dim])

        out_img = torch.cat(rendered_chunks, dim=-1)
        return out_img.permute(2, 0, 1).contiguous()

    def render_splat_rgb(
        self,
        inp,
        scale,
        gt_size=None,
        scale_modify=None,
    ):
        out_h, out_w = self._scale_to_hw(
            inp, scale, gt_size=gt_size, scale_modify=scale_modify
        )
        sm = self._scale_modify_to_tensor(
            scale_modify, scale, inp.shape[0], inp.device, inp.dtype
        )

        if not self.gauss_layer_mu or not self.gauss_layer_rgb:
            raise RuntimeError(
                'Layer-separated Gaussian parameters are unavailable. '
                'Call gen_feat() before rendering.'
            )

        final_images = []
        for i in range(inp.shape[0]):
            base_num = self._render_splat_single(
                self.gauss_layer_mu[0][i],
                self.gauss_layer_sigma[0][i],
                self.gauss_layer_rho[0][i],
                self.gauss_layer_alpha[0][i],
                self.gauss_layer_rgb[0][i],
                out_h,
                out_w,
                scale=scale,
                scale_modify=sm[i],
                background_value=0.0,
            )
            base_ones = self.gauss_layer_rgb[0].new_ones(
                (self.gauss_layer_rgb[0].shape[1], 1)
            )
            base_coverage = self._render_splat_single(
                self.gauss_layer_mu[0][i],
                self.gauss_layer_sigma[0][i],
                self.gauss_layer_rho[0][i],
                self.gauss_layer_alpha[0][i],
                base_ones,
                out_h,
                out_w,
                scale=scale,
                scale_modify=sm[i],
                background_value=0.0,
            )
            current = base_num / base_coverage.clamp_min(1e-6)

            for layer_idx, residual_attr in enumerate(
                self.gauss_layer_rgb_residual, start=1
            ):
                residual_image = self._render_splat_single(
                    self.gauss_layer_mu[layer_idx][i],
                    self.gauss_layer_sigma[layer_idx][i],
                    self.gauss_layer_rho[layer_idx][i],
                    self.gauss_layer_alpha[layer_idx][i],
                    residual_attr[i],
                    out_h,
                    out_w,
                    scale=scale,
                    scale_modify=sm[i],
                    background_value=0.0,
                )
                current = current + residual_image


            final_images.append(current)

        pred = torch.stack(final_images, dim=0)

        return pred

    def _sample_splat_at_coord(self, image, coord):
        # Coordinates are normalized in [y, x] order.
        grid = coord[..., [1, 0]].view(coord.shape[0], coord.shape[1], 1, 2)
        sampled = F.grid_sample(
            image, grid, mode='bilinear',
            padding_mode='border', align_corners=False
        )
        sampled = sampled[:, :, :, 0].permute(0, 2, 1).contiguous()
        return sampled

    def query_rgb(
        self,
        coord,
        scale,
        cell=None,
        return_aux=False,
        gt_size=None,
        scale_modify=None,
    ):
        del cell
        prior_aux = return_aux is True or return_aux in ('prior', 'full')
        image = self.render_splat_rgb(
            self.inp,
            scale,
            gt_size=gt_size,
            scale_modify=scale_modify,
        )
        pred = self._sample_splat_at_coord(image, coord)
        if return_aux == 'loss':
            aux = self._loss_aux_dict()
            aux['cover'] = pred.new_ones(pred.shape[:2])
            return pred, aux
        if prior_aux:
            aux = self._prior_aux_dict()
            aux['cover'] = pred.new_ones(pred.shape[:2])
            return pred, aux
        return pred

    def _loss_aux_dict(self):
        """Return tensors required by the topology supervision losses."""
        return {
            'topology_edge': self.topology_edge_map,
            'topology_node': self.topology_node_map,
            'topology_link': self.topology_link_map,
            'orientation': self.orientation_map,
        }

    def _prior_aux_dict(self):
        """Return the learned topology and density prior maps."""
        return {
            'complexity': self.complexity_map,
            'structure_complexity': self.structure_complexity_map,
            'density_raw': self.density_raw_map,
            'density_blend': self.density_blend_map,
            'density': self.density_map,
            'scaffold_score': self.scaffold_score_map,
            'seed_prob': self.seed_prob_map,
            'alpha_floor_map': self.alpha_floor_map,
            'sigma_floor_map': self.sigma_floor_prior_map,
            'topology_node': self.topology_node_map,
            'topology_edge': self.topology_edge_map,
            'topology_link': self.topology_link_map,
            'topology_importance': self.topology_importance_map,
            'orientation': self.orientation_map,
        }

    def render_rgb(self, inp, scale, bsize=None, gt_size=None, scale_modify=None):
        del bsize
        self.gen_feat(inp, scale=scale, scale_modify=scale_modify)
        return self.render_splat_rgb(
            inp,
            scale,
            gt_size=gt_size,
            scale_modify=scale_modify,
        )

    def forward(self, inp, scale, gt_size=None, scale_modify=None, return_aux=False):
        self.gen_feat(inp, scale=scale, scale_modify=scale_modify)
        pred = self.render_splat_rgb(
            inp,
            scale,
            gt_size=gt_size,
            scale_modify=scale_modify,
        )
        if return_aux == 'loss':
            return pred, self._loss_aux_dict()
        if return_aux is True or return_aux in ('prior', 'full'):
            return pred, self._prior_aux_dict()
        return pred
