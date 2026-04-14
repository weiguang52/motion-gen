"""
Brief:
定义纯单流 DiT 主干网络结构
"""
import torch
import torch.nn as nn


def modulate(x, shift, scale):
    # DiT-style adaptive layer-norm modulation
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MotionDiTBlock(nn.Module):
    """
    纯单流 DiT block：motion self-attn + text cross-attn + MLP。
    """

    def __init__(self, hidden_size, num_heads, ff_size, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size),
        )
        self.gate_chunks = 9
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, self.gate_chunks * hidden_size, bias=True),
        )

    def forward(self, x, cond, key_padding_mask=None, context=None, context_mask=None):
        gates = self.adaLN_modulation(cond).chunk(self.gate_chunks, dim=1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = gates[:6]

        attn_inp = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(
            attn_inp,
            attn_inp,
            attn_inp,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + gate_msa.unsqueeze(1) * attn_out

        if context is not None:
            shift_ca, scale_ca, gate_ca = gates[6], gates[7], gates[8]
            ca_inp = modulate(self.cross_norm(x), shift_ca, scale_ca)
            ca_out, _ = self.cross_attn(
                ca_inp,
                context,
                context,
                key_padding_mask=context_mask,
                need_weights=False,
            )
            x = x + gate_ca.unsqueeze(1) * ca_out

        mlp_inp = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(mlp_inp)
        return x


class MotionDiT(nn.Module):
    """
    纯单流 DiT，直接处理 motion 序列 (seqlen, batch, dim)。
    """

    def __init__(self, hidden_size, ff_size, depth, num_heads, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MotionDiTBlock(
                    hidden_size,
                    num_heads,
                    ff_size,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        self._init_weights()

    def forward(self, x, cond, key_padding_mask=None, context=None, context_mask=None):
        x = x.permute(1, 0, 2)
        for block in self.blocks:
            x = block(
                x,
                cond,
                key_padding_mask=key_padding_mask,
                context=context,
                context_mask=context_mask,
            )

        shift, scale = self.final_modulation(cond).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return x.permute(1, 0, 2)

    def _init_weights(self):
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_modulation[-1].weight, 0)
        nn.init.constant_(self.final_modulation[-1].bias, 0)


__all__ = ["MotionDiT", "MotionDiTBlock", "modulate"]
