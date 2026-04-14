"""
Brief:
提供 bert_clip 条件路径使用的文本 token refiner。
"""
import torch
import torch.nn as nn


def _modulate_norm(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TextRefinerBlock(nn.Module):
    """A small timestep-conditioned self-attention block for text tokens."""

    def __init__(self, dim, num_heads=4, ff_size=None, dropout=0.0):
        super().__init__()
        ff_size = ff_size or (dim * 4)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, dim),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, cond, key_padding_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=1)

        attn_inp = _modulate_norm(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(
            attn_inp,
            attn_inp,
            attn_inp,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + gate_msa.unsqueeze(1) * attn_out

        mlp_inp = _modulate_norm(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_inp)
        return x


class MotionRefinerBlock(nn.Module):
    """A standard self-attention block for motion prefix tokens."""

    def __init__(self, dim, num_heads=4, ff_size=None, dropout=0.0):
        super().__init__()
        ff_size = ff_size or (dim * 4)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, dim),
        )

    def forward(self, x, key_padding_mask=None):
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(
            norm_x,
            norm_x,
            norm_x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TextTokenRefiner(nn.Module):
    """Refine text tokens with timestep-aware conditioning before motion cross-attention."""

    def __init__(self, dim, num_layers=2, num_heads=4, ff_size=None, dropout=0.0):
        super().__init__()
        ff_size = ff_size or (dim * 4)
        self.context_encoder = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.blocks = nn.ModuleList(
            [TextRefinerBlock(dim, num_heads=num_heads, ff_size=ff_size, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, tokens, time_emb, key_padding_mask=None):
        x = tokens.permute(1, 0, 2)
        if key_padding_mask is not None:
            valid = (~key_padding_mask).float().unsqueeze(-1)
            denom = valid.sum(dim=1).clamp(min=1e-6)
            pooled = (x * valid).sum(dim=1) / denom
        else:
            pooled = x.mean(dim=1)
        cond = time_emb.squeeze(0) + self.context_encoder(pooled)
        for block in self.blocks:
            x = block(x, cond, key_padding_mask=key_padding_mask)
        return x.permute(1, 0, 2)

__all__ = ["TextRefinerBlock", "MotionRefinerBlock", "TextTokenRefiner"]
