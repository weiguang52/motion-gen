"""
Brief:
封装 MDM 的文本条件初始化与前向构建逻辑，主要服务于 text-conditioning 相关模块装配与条件张量生成。
"""
import torch.nn as nn

from model.conditioning.text_conditioning import TextConditionEncoder, unpack_text_embed
from model.conditioning.text_refiner import TextTokenRefiner


class TextConditioningStack:
    def __init__(self, owner):
        self.owner = owner

    def init_conditioning(self, kargs, clip_version):
        print("EMBED TEXT")
        self._init_text_conditioner(kargs, clip_version)
        self._init_text_projection_layers()
        self._init_text_refiner(kargs)

    def _init_text_conditioner(self, kargs, clip_version):
        assert self.owner.arch in ["dit", "ar_dit"]
        print("Loading BERT + CLIP (hybrid text conditioning)...")

        self.owner.text_conditioner = TextConditionEncoder(
            encoder_type="bert_clip",
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
        self.owner.text_refiner = None
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

    def build_condition_embeddings(self, time_emb, y, mask_cond_fn):
        text_mask = None
        text_emb = None
        text_global_emb = None

        enc_text = y["text_embed"] if "text_embed" in y else self.owner.encode_text(y["text"])
        ctxt_tokens, text_mask, vtxt_tokens = unpack_text_embed(enc_text)

        text_emb = self.owner.embed_text(
            mask_cond_fn(ctxt_tokens, force_mask=y.get("uncond", False))
        )
        if self.owner.text_refiner is not None:
            text_emb = self.owner.text_refiner(text_emb, time_emb, key_padding_mask=text_mask)

        text_global_emb = self.owner.embed_text_global(
            mask_cond_fn(vtxt_tokens, force_mask=y.get("uncond", False))
        )

        return text_emb, text_mask, text_global_emb

    def clear_cache(self):
        self.owner.text_conditioner.clear_cache()
