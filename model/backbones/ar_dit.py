"""
Brief:
定义面向 prefix-completion / autoregressive 任务的 AR DiT 主干网络结构。
"""
import torch.nn as nn

from model.backbones.dit import modulate


class MotionARDiTBlock(nn.Module):
    """
    AR-specialized DiT block：motion self-attn + text cross-attn + MLP。
    """

    def __init__(self, hidden_size, num_heads, ff_size, dropout=0.0, use_dual_cross_attn=False):
        super().__init__()
        self.use_dual_cross_attn = use_dual_cross_attn
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        if self.use_dual_cross_attn:
            self.cross_norm_text = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.cross_attn_text = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
            self.cross_norm_prefix = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.cross_attn_prefix = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
            self.gate_chunks = 12
        else:
            self.cross_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
            self.gate_chunks = 9
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, ff_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, self.gate_chunks * hidden_size, bias=True),
        )

    def forward(
        self,
        x,
        cond,
        key_padding_mask=None,
        context=None,
        context_mask=None,
        text_context=None,
        text_context_mask=None,
        prefix_context=None,
        prefix_context_mask=None,
    ):
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

        if self.use_dual_cross_attn and (text_context is not None or prefix_context is not None):
            shift_ca_text, scale_ca_text, gate_ca_text = gates[6:9]
            shift_ca_prefix, scale_ca_prefix, gate_ca_prefix = gates[9:12]
            cross_residual = 0.0

            if text_context is not None:
                text_inp = modulate(self.cross_norm_text(x), shift_ca_text, scale_ca_text)
                text_out, _ = self.cross_attn_text(
                    text_inp,
                    text_context,
                    text_context,
                    key_padding_mask=text_context_mask,
                    need_weights=False,
                )
                cross_residual = cross_residual + gate_ca_text.unsqueeze(1) * text_out

            if prefix_context is not None:
                prefix_inp = modulate(self.cross_norm_prefix(x), shift_ca_prefix, scale_ca_prefix)
                prefix_out, _ = self.cross_attn_prefix(
                    prefix_inp,
                    prefix_context,
                    prefix_context,
                    key_padding_mask=prefix_context_mask,
                    need_weights=False,
                )
                cross_residual = cross_residual + gate_ca_prefix.unsqueeze(1) * prefix_out

            x = x + cross_residual
        elif (not self.use_dual_cross_attn) and context is not None:
            shift_ca, scale_ca, gate_ca = gates[6:9]
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


class MotionARDiT(nn.Module):
    """
    面向 prefix-conditioned generation 的 AR DiT，motion tokens keep standard
    bidirectional self-attention while cross-attending to external context.
    """

    def __init__(self, hidden_size, ff_size, depth, num_heads, dropout=0.0, use_dual_cross_attn=False):
        super().__init__()
        self.use_dual_cross_attn = use_dual_cross_attn
        self.blocks = nn.ModuleList(
            [
                MotionARDiTBlock(
                    hidden_size,
                    num_heads,
                    ff_size,
                    dropout=dropout,
                    use_dual_cross_attn=use_dual_cross_attn,
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

    def forward(
        self,
        x,
        cond,
        key_padding_mask=None,
        context=None,
        context_mask=None,
        text_context=None,
        text_context_mask=None,
        prefix_context=None,
        prefix_context_mask=None,
    ):
        x = x.permute(1, 0, 2)
        for block in self.blocks:
            x = block(
                x,
                cond,
                key_padding_mask=key_padding_mask,
                context=context,
                context_mask=context_mask,
                text_context=text_context,
                text_context_mask=text_context_mask,
                prefix_context=prefix_context,
                prefix_context_mask=prefix_context_mask,
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


__all__ = ["MotionARDiT", "MotionARDiTBlock"]
