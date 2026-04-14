"""
Brief:
提供 MDM 使用的时间/位置编码模块。
"""
import math

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class _BaseContinuousTimestepEmbedder(nn.Module):
    def __init__(self, model_dim, sequence_pos_encoder):
        super().__init__()
        self.model_dim = model_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.model_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.model_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def _positional_embedding(self, positions):
        device = positions.device
        dtype = self.sequence_pos_encoder.pe.dtype
        positions_fp32 = positions.to(dtype=torch.float32)
        position = positions_fp32.unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.model_dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.model_dim)
        )
        emb = torch.zeros((positions.shape[0], self.model_dim), device=device, dtype=torch.float32)
        emb[:, 0::2] = torch.sin(position * div_term)
        emb[:, 1::2] = torch.cos(position * div_term)
        return emb.to(dtype=dtype).unsqueeze(1)

    def _positions_from_timesteps(self, timesteps):
        raise NotImplementedError

    def forward(self, timesteps):
        pos_emb = self._positional_embedding(self._positions_from_timesteps(timesteps))
        return self.time_embed(pos_emb).permute(1, 0, 2)


class DDPMContinuousTimestepEmbedder(_BaseContinuousTimestepEmbedder):
    def __init__(self, model_dim, sequence_pos_encoder, *, num_timesteps, rescale_timesteps=False):
        super().__init__(model_dim, sequence_pos_encoder)
        self.num_timesteps = int(num_timesteps)
        self.rescale_timesteps = bool(rescale_timesteps)

    def _positions_from_timesteps(self, timesteps):
        t_cont = timesteps.to(dtype=torch.float32).clamp(0.0, 1.0)
        positions = t_cont * float(max(self.num_timesteps - 1, 1))
        if self.rescale_timesteps:
            positions = positions * (1000.0 / float(max(self.num_timesteps, 1)))
        return positions


class FMContinuousTimestepEmbedder(_BaseContinuousTimestepEmbedder):
    def __init__(self, model_dim, sequence_pos_encoder, *, num_timesteps):
        super().__init__(model_dim, sequence_pos_encoder)
        del num_timesteps
        self.time_factor = 1.0

    def _positions_from_timesteps(self, timesteps):
        t_cont = timesteps.to(dtype=torch.float32).clamp(0.0, 1.0)
        return t_cont * self.time_factor


def build_timestep_embedder(
    model_dim,
    sequence_pos_encoder,
    *,
    mode="fm",
    num_timesteps=1000,
    rescale_timesteps=False,
):
    mode = str(mode).lower()
    if mode == "ddpm":
        return DDPMContinuousTimestepEmbedder(
            model_dim,
            sequence_pos_encoder,
            num_timesteps=num_timesteps,
            rescale_timesteps=rescale_timesteps,
        )
    if mode == "fm":
        return FMContinuousTimestepEmbedder(
            model_dim,
            sequence_pos_encoder,
            num_timesteps=num_timesteps,
        )
    raise ValueError(f"Unsupported time embedding mode: {mode}")


def _rotate_half(x):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head_dim [{dim}] must be even.")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(self, seq_len, device, dtype):
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1).to(dtype=dtype)
        sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1).to(dtype=dtype)
        return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)

    def apply_rotary(self, q, k):
        cos, sin = self._cos_sin(q.shape[-2], q.device, q.dtype)
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)
        return q, k


__all__ = [
    "PositionalEncoding",
    "DDPMContinuousTimestepEmbedder",
    "FMContinuousTimestepEmbedder",
    "RotaryEmbedding",
    "build_timestep_embedder",
]
