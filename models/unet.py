import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_valid_group_num(num_channels: int, max_groups: int = 32) -> int:
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class TimeEmbedding(nn.Module):
    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.time_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.time_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, channels: int, max_h: int = 128, max_w: int = 128):
        super().__init__()
        self.row_embed = nn.Parameter(torch.zeros(1, channels, max_h, 1))
        self.col_embed = nn.Parameter(torch.zeros(1, channels, 1, max_w))
        nn.init.normal_(self.row_embed, std=0.02)
        nn.init.normal_(self.col_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        r = F.interpolate(self.row_embed, size=(H, 1), mode="bilinear", align_corners=False)
        c = F.interpolate(self.col_embed, size=(1, W), mode="bilinear", align_corners=False)
        return x + r + c


class PhysicsFlowAlignment(nn.Module):
    def __init__(self, flow_channels: int, cond_channels: int, inter_channels: int = 32):
        super().__init__()
        self.inter_channels = inter_channels

        # Length kept 15 for old checkpoint tensor compatibility.
        self.geo_indices = [0, 1, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        self.geo_dim = len(self.geo_indices)

        self.pos_enc = LearnablePositionalEncoding(inter_channels)
        self.to_q = nn.Conv2d(flow_channels, inter_channels, 1)
        self.to_k = nn.Conv2d(self.geo_dim, inter_channels, 1)
        self.to_v = nn.Conv2d(self.geo_dim, flow_channels, 1)

        self.gating = nn.Sequential(
            nn.Conv2d(inter_channels * 2, inter_channels, 1),
            nn.GroupNorm(4, inter_channels),
            nn.SiLU(),
            nn.Conv2d(inter_channels, 1, 1),
        )

        nn.init.xavier_normal_(self.to_v.weight)
        nn.init.zeros_(self.to_v.bias)
        nn.init.zeros_(self.gating[-1].weight)
        nn.init.zeros_(self.gating[-1].bias)
        nn.init.orthogonal_(self.to_q.weight, gain=1.0)
        nn.init.orthogonal_(self.to_k.weight, gain=1.0)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_flow: torch.Tensor, x_cond: torch.Tensor):
        x_geo = x_cond[:, self.geo_indices, :, :]
        q = self.to_q(x_flow)
        k = self.to_k(x_geo)
        v = self.to_v(x_geo)
        k = self.pos_enc(k)
        corr = (q * k) * (self.inter_channels ** -0.5)
        mask = torch.sigmoid(self.gating(torch.cat([corr, k], dim=1)) / self.temperature)
        return x_flow + mask * v, mask


class PhysicsSPADE(nn.Module):
    """
    Pipe26 response grouping.

    Group sizes intentionally match the old model:
      geometry=10, material=3, reflection=4, propagation=5, detail=4
    """

    def __init__(self, cond_channels: int, norm_nc: int, kernel_size: int = 3):
        super().__init__()

        self.channel_to_group = {
            "geometry": [0, 1, 12, 13, 14, 15, 16, 17, 18, 19],
            "material": [2, 3, 4],
            "reflection": [10, 11, 20, 21],
            "propagation": [5, 6, 7, 8, 9],
            "detail": [22, 23, 24, 25],
        }
        self.num_groups = len(self.channel_to_group)
        self.embed_dim = 32

        self.group_encoders = nn.ModuleDict()
        for name, indices in self.channel_to_group.items():
            self.group_encoders[name] = nn.Sequential(
                nn.Conv2d(len(indices), self.embed_dim, 1),
                nn.GroupNorm(min(4, self.embed_dim), self.embed_dim),
                nn.SiLU(),
                nn.Conv2d(self.embed_dim, self.embed_dim, 3, padding=1),
                nn.GroupNorm(min(4, self.embed_dim), self.embed_dim),
                nn.SiLU(),
            )

        self.physics_coupling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.num_groups * self.embed_dim, self.num_groups * self.embed_dim, 1, groups=self.num_groups),
            nn.SiLU(),
            nn.Conv2d(self.num_groups * self.embed_dim, self.num_groups * self.embed_dim, 1),
            nn.Sigmoid(),
        )

        self.fusion = nn.Conv2d(self.num_groups * self.embed_dim, norm_nc * 2, 3, padding=1)
        nn.init.zeros_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)
        self.coupling_strength = nn.Parameter(torch.zeros(1))

        self.geometry_enhance = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim // 4, 1),
            nn.SiLU(),
            nn.Conv2d(self.embed_dim // 4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        if cond.shape[-2:] != (H, W):
            cond = F.interpolate(cond, size=(H, W), mode="nearest")

        group_feats = {}
        for name, indices in self.channel_to_group.items():
            group_feats[name] = self.group_encoders[name](cond[:, indices, :, :])

        stack = torch.stack([group_feats[name] for name in self.channel_to_group.keys()], dim=1)
        flat = stack.flatten(1, 2)
        weights = self.physics_coupling(flat)
        interacted = flat * (1 + self.coupling_strength * weights)

        geo_mask = self.geometry_enhance(group_feats["geometry"])
        interacted_geo = interacted[:, :self.embed_dim] * (1 + geo_mask)
        interacted = torch.cat([interacted_geo, interacted[:, self.embed_dim:]], dim=1)

        gamma, beta = self.fusion(interacted).chunk(2, dim=1)
        return x * (1 + gamma) + beta


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_ch: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(get_valid_group_num(in_ch, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.spade = PhysicsSPADE(cond_ch, out_ch)
        self.pfa = PhysicsFlowAlignment(out_ch, cond_ch)
        self.norm2 = nn.GroupNorm(get_valid_group_num(out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = h + self.time_proj(F.silu(temb)).view(temb.shape[0], -1, 1, 1)
        h = self.norm2(h)
        h = self.spade(h, cond)

        # PFA remains off by default for stable checkpoint compatibility.
        # h, _ = self.pfa(h, cond)

        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(get_valid_group_num(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        q = q.view(B, self.num_heads, self.head_dim, H * W)
        k = k.view(B, self.num_heads, self.head_dim, H * W)
        v = v.view(B, self.num_heads, self.head_dim, H * W)
        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v).reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.op(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet_v(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        cond_channels: int = 26,
        time_dim: int = 256,
        base_channels: int = 128,
        dropout: float = 0.1,
        num_heads: int = 4,
        attn_resolutions=(32, 16),
    ):
        super().__init__()
        self.cond_channels = cond_channels
        self.time_embed = TimeEmbedding(time_dim)
        ch = base_channels
        ch_mults = [1, 2, 3, 4]
        resolutions = [128, 64, 32, 16]

        self.stem = nn.Conv2d(in_channels + cond_channels, ch, 3, padding=1)

        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.attn_enc = nn.ModuleList()
        in_ch = ch
        for i, mult in enumerate(ch_mults):
            out_ch = ch * mult
            self.enc_blocks.append(ResBlock(in_ch, out_ch, cond_channels, time_dim, dropout))
            self.enc_blocks.append(ResBlock(out_ch, out_ch, cond_channels, time_dim, dropout))
            in_ch = out_ch
            self.attn_enc.append(SelfAttention2d(in_ch, num_heads=num_heads) if resolutions[i] in attn_resolutions else nn.Identity())
            self.downs.append(Downsample(in_ch) if i != len(ch_mults) - 1 else nn.Identity())

        self.mid1 = ResBlock(in_ch, in_ch, cond_channels, time_dim, dropout)
        self.mid_attn = SelfAttention2d(in_ch, num_heads=num_heads) if 16 in attn_resolutions else nn.Identity()
        self.mid2 = ResBlock(in_ch, in_ch, cond_channels, time_dim, dropout)

        self.dec_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.attn_dec = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mults))):
            out_ch = ch * mult
            self.dec_blocks.append(ResBlock(in_ch + out_ch, out_ch, cond_channels, time_dim, dropout))
            self.dec_blocks.append(ResBlock(out_ch, out_ch, cond_channels, time_dim, dropout))
            in_ch = out_ch
            self.attn_dec.append(SelfAttention2d(in_ch, num_heads=num_heads) if resolutions[i] in attn_resolutions else nn.Identity())
            self.ups.append(Upsample(in_ch) if i != 0 else nn.Identity())

        self.out_norm = nn.GroupNorm(get_valid_group_num(in_ch), in_ch)
        self.out_conv = nn.Conv2d(in_ch, in_channels, 3, padding=1)

    def _cond_pyramid(self, C: torch.Tensor):
        C0 = C
        C1 = F.avg_pool2d(C0, 2)
        C2 = F.avg_pool2d(C1, 2)
        C3 = F.avg_pool2d(C2, 2)
        return [C0, C1, C2, C3]

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        temb = self.time_embed(t)
        C_pyr = self._cond_pyramid(C)
        x = self.stem(torch.cat([z_t, C_pyr[0]], dim=1))
        skips = []
        bi = 0
        for level in range(4):
            cond = C_pyr[level]
            x = self.enc_blocks[bi](x, temb, cond)
            x = self.enc_blocks[bi + 1](x, temb, cond)
            bi += 2
            x = self.attn_enc[level](x)
            skips.append(x)
            x = self.downs[level](x)

        x = self.mid1(x, temb, C_pyr[-1])
        x = self.mid_attn(x)
        x = self.mid2(x, temb, C_pyr[-1])

        bi = 0
        for level in range(4):
            rev_level = 3 - level
            cond = C_pyr[rev_level]
            x = torch.cat([x, skips[rev_level]], dim=1)
            x = self.dec_blocks[bi](x, temb, cond)
            x = self.dec_blocks[bi + 1](x, temb, cond)
            bi += 2
            x = self.attn_dec[level](x)
            x = self.ups[level](x)

        return self.out_conv(F.silu(self.out_norm(x)))
