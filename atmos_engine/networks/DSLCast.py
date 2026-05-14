import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Positional Utilities
# =========================
def get_2d_sincos_pos_embed(dim: int, hw, cls_token: bool = False):
    """Return numpy array of shape [H*W, dim] (or [1+H*W, dim] if cls_token)."""
    H, W = hw
    yy = np.arange(H, dtype=np.float32)
    xx = np.arange(W, dtype=np.float32)
    grid = np.stack(np.meshgrid(xx, yy), axis=0).reshape(2, 1, H, W)  # (2,1,H,W)
    pe = _pos_from_grid(dim, grid)
    if cls_token:
        pe = np.concatenate([np.zeros([1, dim], dtype=np.float32), pe], axis=0)
    return pe

def _pos_from_grid(dim: int, grid):
    assert dim % 2 == 0
    a = _pos_1d(dim // 2, grid[0])
    b = _pos_1d(dim // 2, grid[1])
    return np.concatenate([a, b], axis=1)

def _pos_1d(dim: int, coords):
    assert dim % 2 == 0
    omega = 1.0 / (10000 ** (np.arange(dim // 2, dtype=np.float64) / (dim / 2.0)))
    coords = coords.reshape(-1)
    out = np.einsum('m,d->md', coords, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1).astype(np.float32)


# =========================
# Patch Embedding (conv2d as patchifier)
# =========================
class PatchEmbed2D(nn.Module):
    """[B,C,H,W] -> [B, L, D] with L = (H/ps0)*(W/ps1)."""
    def __init__(self, img_size=(120, 240), patch_size=(2, 2), in_chans=97, embed_dim=1024):
        super().__init__()
        self.img_size = tuple(img_size)
        self.patch_size = tuple(patch_size)
        # 2x2 走非重叠；>2 则半步重叠，但把有效下采样倍率封顶到 2×
        # 自适应 stride（按 patch 大小映射）：小 patch 稳精度，大 patch 明确降显存
        kh, kw = self.patch_size
        # 关键：非重叠分块，token≈H/kh × W/kw，显存与patch直接挂钩
        self.stride = (kh, kw)

        
        # 去掉原来的整除断言，改为基于卷积输出尺寸的计算
        # 输出尺寸（无 padding）：out = floor((in - kernel) / stride) + 1
        # SAME-like padding：p = floor((k - s)/2)，逐维计算，保证 token 尺寸只由 stride 决定
        ph = max(0, math.ceil((kh - self.stride[0]) / 2))
        pw = max(0, math.ceil((kw - self.stride[1]) / 2))
        self.padding = (ph, pw)
        
        H, W = self.img_size
        self.h = (H + 2*ph - kh) // self.stride[0] + 1
        self.w = (W + 2*pw - kw) // self.stride[1] + 1

        self.num_patches = self.h * self.w

        # 改 stride：由非重叠（=kernel）改为重叠（=kernel/2）
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(kh, kw), stride=self.stride, padding=self.padding, bias=True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        assert (H, W) == self.img_size, f"Input size ({H},{W}) != configured img_size {self.img_size}"
        x = self.proj(x)                      # [B, D, h, w]
        x = x.permute(0, 2, 3, 1).contiguous()# [B, h, w, D]
        x = x.view(B, self.h * self.w, -1)    # [B, L, D]
        return x


# =========================
# DropPath (stochastic depth)
# =========================
class DropPath(nn.Module):
    """Per-sample stochastic depth on residual branches."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x * mask / keep


# =========================
# Axially-Gated Depthwise-Separable Convolution Block (AGB)
# =========================
class AxiallyGatedDepthwiseSeparableBlock(nn.Module):
    """
    Depthwise axial convs (1×k & k×1) + pointwise mix + sigmoid gate, then channel MLP (1×1), with two residuals.
    """
    def __init__(self, dim, mlp_ratio=4., dropout=0., path_drop=0., kernel_size=7):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(1, kernel_size),
                              padding=(0, kernel_size // 2), groups=dim, bias=True)
        self.dw_v = nn.Conv2d(dim, dim, kernel_size=(kernel_size, 1),
                              padding=(kernel_size // 2, 0), groups=dim, bias=True)
        self.mix   = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.gproj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.drop_path = DropPath(path_drop) if path_drop > 0. else nn.Identity()

        self.norm2 = nn.GroupNorm(1, dim)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hid, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hid, dim, kernel_size=1, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        u = self.norm1(x)
        axial = self.dw_h(u) + self.dw_v(u)
        gate = torch.sigmoid(self.gproj(u))
        y = self.mix(axial) * gate
        x = res + self.drop_path(y)

        z = self.norm2(x)
        x = x + self.drop_path(self.mlp(z))
        return x


# =========================
# Earth-Aware Latitudinal Positional Encoding (ELPE)
# =========================
class EarthAwareLatitudinalPositionalEncoding(nn.Module):
    """Return [1,2,h,w] = [sin(lat), cos(lat)] stretched along height then repeated along width."""
    def __init__(self, h, w, lat_range=(-90.0, 90.0)):
        super().__init__()
        lat0, lat1 = lat_range
        lat = torch.linspace(lat0, lat1, steps=h).view(h, 1) * math.pi / 180.0
        pos = torch.stack([torch.sin(lat).repeat(1, w), torch.cos(lat).repeat(1, w)], dim=0).unsqueeze(0)
        self.register_buffer("pos_map", pos, persistent=False)
    def forward(self) -> torch.Tensor:
        return self.pos_map  # [1,2,h,w]


# =========================
# Differentiable Semi-Lagrangian Advection Block (DSL-Block)
# =========================
class DifferentiableSemiLagrangianAdvectionBlock(nn.Module):
    """
    AGB -> flow field (ux,uy) in pixels/step -> grid_sample backward tracing -> gated mix -> channel MLP.
    """
    def __init__(self, dim, H, W, mlp_ratio=4., dropout=0., path_drop=0., kernel_size=7, flow_max=3.0):
        super().__init__()
        self.H, self.W, self.flow_max = H, W, float(flow_max)

        # grid centers for align_corners=False
        xs = torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W)
        ys = torch.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer("base_grid", torch.stack([xx, yy], dim=-1).unsqueeze(0), persistent=False)  # [1,H,W,2]

        # AGB trunk
        self.ax = AxiallyGatedDepthwiseSeparableBlock(dim, mlp_ratio=mlp_ratio, dropout=dropout, path_drop=0., kernel_size=kernel_size)

        # flow/gate/mix
        self.flow = nn.Conv2d(dim, 2, kernel_size=3, padding=1, bias=True)
        self.mix  = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.gate = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        self.drop_path = DropPath(path_drop) if path_drop > 0. else nn.Identity()

        # channel MLP
        self.norm2 = nn.GroupNorm(1, dim)
        hid = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hid, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hid, dim, kernel_size=1, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        f = self.ax(x)

        # pixel flow -> normalized displacement (align_corners=False)
        uv = torch.tanh(self.flow(f)) * self.flow_max          # [B,2,H,W]
        disp = torch.cat([uv[:, 0:1] * (2.0 / self.W), uv[:, 1:2] * (2.0 / self.H)], dim=1)  # [B,2,H,W]
        grid = self.base_grid - disp.permute(0, 2, 3, 1)

        warped = F.grid_sample(f, grid, mode='bilinear', padding_mode='border', align_corners=False)

        g = torch.sigmoid(self.gate(f))
        y = self.mix(warped) * g
        x = res + self.drop_path(y)

        z = self.norm2(x)
        x = x + self.drop_path(self.mlp(z))
        return x


# =========================
# Model (single forward; renamed args) Differentiable Semi-Lagrangian Network for Multi-Horizon Ocean Forecasting
# =========================
class DSLCast(nn.Module):
    def __init__(self,
                 params,
                 img_size=(180, 360), patch_size=(2, 2),
                 in_chans=70, out_chans=69,
                 feat_dim=256,
                 enc_layers=16,
                 dec_feat_dim=512,
                 dec_layers=8,
                 mlp_ratio=4.,
                 path_drop=0.10,
                 dropout=0.0,
                 use_earth_posenc=True,
                 lat_range=(-90.0, 90.0),
                 adv_stride=4,
                 refine_hidden=128,
                 flow_max=3.0):
        super().__init__()

        self.img_size = tuple(img_size)
        self.patch_size = tuple(patch_size)
        self.N_in_channels = in_chans
        self.N_out_channels = out_chans
        self.refine_hidden = refine_hidden
        self.feat_dim = feat_dim
        self.dec_feat_dim = dec_feat_dim
        
        # 先构建 patch_embed，再从中读取 h,w,num_patches（与重叠 stride 一致）
        self.patch_embed = PatchEmbed2D(self.img_size, self.patch_size, self.N_in_channels, self.feat_dim)
        self.h = self.patch_embed.h
        self.w = self.patch_embed.w
        self.num_patches = self.patch_embed.num_patches


        # fixed token pos-embed (init once)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.feat_dim), requires_grad=False)
        pe = get_2d_sincos_pos_embed(self.feat_dim, (self.h, self.w), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))

        # init patch conv like linear
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # earth pos on feature grid
        self.use_earth_posenc = use_earth_posenc
        if self.use_earth_posenc:
            self.earth_pos = EarthAwareLatitudinalPositionalEncoding(self.h, self.w, lat_range=lat_range)
            self.lat_proj = nn.Conv2d(2, self.feat_dim, kernel_size=1, bias=True)

        # encoder backbone (AGB + DSL-Block)

        dpr = torch.linspace(0, path_drop, enc_layers).tolist()
        enc = []
        
        sh, sw = self.patch_embed.stride
        eff_down = max(sh, sw)
        scaled_flow_max = float(flow_max) * eff_down


        
        for i in range(enc_layers):
            if adv_stride > 0 and (i + 1) % adv_stride == 0:
                enc.append(DifferentiableSemiLagrangianAdvectionBlock(self.feat_dim, self.h, self.w,
                                                                      mlp_ratio=mlp_ratio, dropout=dropout,
                                                                      path_drop=dpr[i], kernel_size=7,
                                                                      flow_max=scaled_flow_max))   # ★ 用 scaled_flow_max
            else:
                enc.append(AxiallyGatedDepthwiseSeparableBlock(self.feat_dim, mlp_ratio=mlp_ratio,
                                                               dropout=dropout, path_drop=dpr[i], kernel_size=7))

        self.blocks = nn.ModuleList(enc)
        self.enc_norm2d = nn.GroupNorm(1, self.feat_dim)

        # decoder
        self.decoder_embed = nn.Conv2d(self.feat_dim, self.dec_feat_dim, kernel_size=1, bias=True)
        dpr_dec = torch.linspace(0, path_drop, dec_layers).tolist()

        self.decoder_blocks = nn.ModuleList([
            AxiallyGatedDepthwiseSeparableBlock(self.dec_feat_dim, mlp_ratio=mlp_ratio, dropout=dropout,
                                                path_drop=dpr_dec[i], kernel_size=7)
            for i in range(dec_layers)
        ])
        self.decoder_norm2d = nn.GroupNorm(1, self.dec_feat_dim)
        # ★ 新增：与编码端对偶的 kernel/stride/padding 和 output_padding
        kh, kw = self.patch_size
        sh, sw = self.patch_embed.stride
        ph, pw = self.patch_embed.padding
        H, W = self.img_size
        r_h = (H + 2*ph - kh) % sh
        r_w = (W + 2*pw - kw) % sw
        self.decoder_pred = nn.ConvTranspose2d(
            self.dec_feat_dim, self.N_out_channels,
            kernel_size=(kh, kw), stride=(sh, sw), padding=(ph, pw),
            output_padding=(r_h, r_w), bias=True
        )


        self.refine = nn.Sequential(
            nn.Conv2d(self.N_out_channels + self.N_in_channels, self.refine_hidden, kernel_size=1, bias=True),           # 1×1 降维（逐点）
            nn.GELU(),
            nn.Conv2d(self.refine_hidden, self.refine_hidden, kernel_size=3, padding=1, groups=self.refine_hidden, bias=False),                  # 3×3 深度卷积
            nn.Conv2d(self.refine_hidden, self.N_out_channels, kernel_size=1, bias=True)                                 # 1×1 逐点恢复到 Cout
        )
        self.refine_gamma = nn.Parameter(torch.tensor(0.0))  # 门控，初始=0，默认不影响原模型

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.0) -> torch.Tensor:
        """
        Single-pass forward: patch -> tokens (+pos) -> 2D feat (+earth pos) -> enc blocks -> dec blocks -> upsample.
        mask_ratio kept for API compat (unused).
        """
        # patch -> tokens
        t = self.patch_embed(x)                     # [B,L,D]
        t = t + self.pos_embed
        B, L, D = t.shape

        # tokens -> 2D feature
        f = t.view(B, self.h, self.w, D).permute(0, 3, 1, 2).contiguous()  # [B,D,h,w]

        # add earth pos
        if self.use_earth_posenc:
            f = f + self.lat_proj(self.earth_pos()) # [B,D,h,w]

        # encoder
        for blk in self.blocks:
            f = blk(f)
        f = self.enc_norm2d(f)

        # decoder
        y = self.decoder_embed(f)
        for blk in self.decoder_blocks:
            y = blk(y)
        y = self.decoder_norm2d(y)

        # depatchify
        out = self.decoder_pred(y)  # [B,out_ch,H,W]
        ref = self.refine(torch.cat([out, x], dim=1))   # 条件细化（深度可分）
        out = out + self.refine_gamma * ref             # 门控残差，γ 从 0 学起，安全不掉点
        return out

# =========================
# Quick self-test
# =========================
if __name__ == '__main__':
    from thop import profile

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = DSLCast().to(device)

    x = torch.randn(1, 70, 180, 360, device=device)
    y = net(x)
    print("Output:", tuple(y.shape))  # (1,93,120,240)

    macs, params = profile(net, inputs=(x,))
    print('macs: ', macs, 'params: ', params)
    print('macs: %.2f G, params: %.2f M' % (macs / 1e9, params / 1e6))
