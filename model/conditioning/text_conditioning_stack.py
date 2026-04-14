"""
Brief:
封装 MDM 的文本条件初始化与前向构建逻辑，主要服务于 text-conditioning 相关模块装配与条件张量生成。
"""
import torch
import torch.nn as nn

from model.conditioning.text_conditioning import TextConditionEncoder, unpack_text_embed
from model.conditioning.text_refiner import TextTokenRefiner


class TextConditioningStack:
    def __init__(self, owner):
        self.owner = owner

    def init_conditioning(self, kargs, clip_version):
        if self.owner.cond_mode == "no_cond":
            return
        if "text" not in self.owner.cond_mode:
            return

        print("EMBED TEXT")
        self.owner.text_encoder_type = kargs.get("text_encoder_type", "bert_clip")
        self._init_text_conditioner(kargs, clip_version)
        self._init_text_projection_layers()
        self._init_text_refiner(kargs)

    def _init_text_conditioner(self, kargs, clip_version):
        assert self.owner.arch in ["trans_dec", "dit", "ar_dit"]
        if self.owner.text_encoder_type == "bert_clip":
            print("Loading BERT + CLIP (hybrid text conditioning)...")
        else:
            raise ValueError("We only support [BERT_CLIP] text encoders")

        self.owner.text_conditioner = TextConditionEncoder(
            encoder_type=self.owner.text_encoder_type,
            dataset=self.owner.dataset,
            clip_version=clip_version,
            cache_text_embeddings=kargs.get("cache_text_embeddings", False),
        )
        self.owner.encode_text = self.owner.text_conditioner.encode
        self.owner.clip_dim = self.owner.text_conditioner.token_dim
        self.owner.clip_global_dim = self.owner.text_conditioner.global_dim

    def _init_text_projection_layers(self):
        self.owner.embed_text = nn.Linear(self.owner.clip_dim, self.owner.model_dim)
        self.owner.embed_text_global = nn.Linear(self.owner.clip_global_dim, self.owner.model_dim)

    def _init_text_refiner(self, kargs):
        if self.owner.text_encoder_type != "bert_clip":
            return
        self.owner.text_refiner_layers = int(kargs.get("text_refiner_layers", 2))
        self.owner.text_refiner_heads = int(kargs.get("text_refiner_heads", self.owner.num_heads))
        if self.owner.text_refiner_layers <= 0:
            return
        self.owner.text_refiner = TextTokenRefiner(
            dim=self.owner.model_dim,
            num_layers=self.owner.text_refiner_layers,
            num_heads=self.owner.text_refiner_heads,
            ff_size=self.owner.ff_size,
            dropout=self.owner.dropout,
        )

    @staticmethod
    def _normalize_context_tokens(ctxt_tokens, text_mask, batch_size):
        if ctxt_tokens is not None:
            if ctxt_tokens.dim() == 3 and ctxt_tokens.shape[0] == batch_size and ctxt_tokens.shape[1] != batch_size:
                ctxt_tokens = ctxt_tokens.permute(1, 0, 2)
            if text_mask is not None and text_mask.dim() == 2 and text_mask.shape[0] != batch_size and text_mask.shape[1] == batch_size:
                text_mask = text_mask.permute(1, 0)
            if text_mask is not None and text_mask.shape[0] == 1 and batch_size > 1:
                text_mask = torch.repeat_interleave(text_mask, batch_size, dim=0)
        return ctxt_tokens, text_mask

    @staticmethod
    def _normalize_global_tokens(vtxt_tokens, batch_size):
        if vtxt_tokens is not None:
            if vtxt_tokens.dim() == 3 and vtxt_tokens.shape[0] == batch_size and vtxt_tokens.shape[1] == 1:
                vtxt_tokens = vtxt_tokens.permute(1, 0, 2)
        return vtxt_tokens

    def build_condition_embeddings(self, time_emb, y, batch_size, mask_cond_fn):
        text_mask = None
        text_emb = None
        text_global_emb = None
        cond_emb = None
        global_cond = time_emb

        if "text" in self.owner.cond_mode:
            enc_text = y["text_embed"] if "text_embed" in y else self.owner.encode_text(y["text"])
            ctxt_tokens, text_mask, vtxt_tokens = unpack_text_embed(enc_text)

            ctxt_tokens, text_mask = self._normalize_context_tokens(ctxt_tokens, text_mask, batch_size)
            if ctxt_tokens is not None:
                text_emb = self.owner.embed_text(
                    mask_cond_fn(ctxt_tokens, force_mask=y.get("uncond", False))
                )
                if hasattr(self.owner, "text_refiner"):
                    text_emb = self.owner.text_refiner(text_emb, time_emb, key_padding_mask=text_mask)

            vtxt_tokens = self._normalize_global_tokens(vtxt_tokens, batch_size)
            if vtxt_tokens is not None:
                embed_text_global = self.owner.embed_text_global if hasattr(self.owner, "embed_text_global") else self.owner.embed_text
                text_global_emb = embed_text_global(
                    mask_cond_fn(vtxt_tokens, force_mask=y.get("uncond", False))
                )

            memory_text_emb = text_emb if text_emb is not None else text_global_emb
            if memory_text_emb is None:
                raise ValueError("Text conditioning is enabled but no text embeddings were produced.")
            cond_emb = memory_text_emb + time_emb

        if self.owner.cond_mode == "no_cond":
            cond_emb = time_emb

        return cond_emb, text_emb, text_mask, global_cond, text_global_emb

    def clear_cache(self):
        if hasattr(self.owner, "text_conditioner"):
            self.owner.text_conditioner.clear_cache()
