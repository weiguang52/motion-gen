"""
Brief:
封装 prefix-completion 相关的上下文构建逻辑，主要服务于 AR 前缀条件编码与跨模态上下文拼接。
"""
import torch
import torch.nn as nn

from model.conditioning.text_refiner import MotionRefinerBlock


def should_use_prefix_cross_context(arch, is_prefix_comp, context_len):
    return arch == "ar_dit" and is_prefix_comp and context_len > 0


class PrefixConditioning(nn.Module):
    def __init__(self, input_feats, model_dim, context_len, pred_len, num_heads, ff_size, dropout):
        super().__init__()
        self.input_feats = input_feats
        self.context_len = context_len
        self.pred_len = pred_len
        self.prefix_proj = nn.Linear(input_feats, model_dim)
        self.prefix_pe = nn.Parameter(torch.randn(1, context_len, model_dim) * 0.02)
        self.motion_refiner = nn.ModuleList(
            [
                MotionRefinerBlock(
                    model_dim,
                    num_heads=num_heads,
                    ff_size=ff_size,
                    dropout=dropout,
                )
                for _ in range(2)
            ]
        )

    def apply_prefix_completion(self, x, y, batch_size, *, is_prefix_comp, use_cross_context):
        if is_prefix_comp and not use_cross_context:
            x = torch.cat([y["prefix"], x], dim=-1)
            y["mask"] = torch.cat(
                [
                    torch.ones(
                        [batch_size, 1, 1, self.context_len],
                        dtype=y["mask"].dtype,
                        device=y["mask"].device,
                    ),
                    y["mask"],
                ],
                dim=-1,
            )
        return x

    def build_prefix_context(self, y, *, use_cross_context):
        if not use_cross_context:
            return None, None

        prefix = y.get("prefix")
        if prefix is None:
            return None, None

        batch_size, _, _, prefix_len = prefix.shape
        prefix_feat = prefix.permute(0, 3, 1, 2).reshape(batch_size, prefix_len, self.input_feats)
        prefix_feat = self.prefix_proj(prefix_feat)
        prefix_feat = prefix_feat + self.prefix_pe[:, :prefix_len, :]

        prefix_mask = None
        orig_lengths = y.get("orig_lengths")
        if orig_lengths is not None:
            prefix_lengths = orig_lengths.to(device=prefix.device) - int(self.pred_len)
            prefix_lengths = prefix_lengths.clamp(min=0, max=prefix_len)
            prefix_positions = torch.arange(prefix_len, device=prefix.device).unsqueeze(0)
            prefix_mask = prefix_positions >= prefix_lengths.unsqueeze(1)

        for block in self.motion_refiner:
            prefix_feat = block(prefix_feat, key_padding_mask=prefix_mask)

        if prefix_mask is not None:
            prefix_feat = prefix_feat.masked_fill(prefix_mask.unsqueeze(-1), 0.0)
        return prefix_feat, prefix_mask


def compose_cross_context(text_emb, text_mask, prefix_context, prefix_mask):
    contexts = []
    masks = []

    if text_emb is not None:
        text_context = text_emb.permute(1, 0, 2)
        contexts.append(text_context)
        if text_mask is not None:
            masks.append(text_mask)
        else:
            masks.append(torch.zeros(text_context.shape[:2], dtype=torch.bool, device=text_context.device))

    if prefix_context is not None:
        contexts.append(prefix_context)
        if prefix_mask is not None:
            masks.append(prefix_mask)
        else:
            masks.append(torch.zeros(prefix_context.shape[:2], dtype=torch.bool, device=prefix_context.device))

    if not contexts:
        return None, None

    context = torch.cat(contexts, dim=1)
    context_mask = torch.cat(masks, dim=1) if masks else None
    return context, context_mask
