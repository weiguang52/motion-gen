"""
Brief:
统一封装 MDM 的文本条件模块，主要服务于冻结的 BERT+CLIP 编码、文本缓存与条件输出组织。
"""
import os
from pathlib import Path

import torch
import torch.nn as nn


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_language_model_root():
    env_root = os.environ.get("TEXT2MOTION_LM_ROOT")
    if env_root:
        return Path(env_root)
    return _REPO_ROOT / "language_models"


_LANGUAGE_MODEL_ROOT = _resolve_language_model_root()
_LOCAL_BERT_MODEL_PATH = _LANGUAGE_MODEL_ROOT / "distilbert-base-uncased"
_DEFAULT_BERT_MODEL_PATH = (
    str(_LOCAL_BERT_MODEL_PATH) if _LOCAL_BERT_MODEL_PATH.exists() else "distilbert/distilbert-base-uncased"
)
_LOCAL_CLIP_B32 = _LANGUAGE_MODEL_ROOT / "clip-vit-base-patch32"
_CLIP_B32_MODEL_ID = str(_LOCAL_CLIP_B32) if _LOCAL_CLIP_B32.exists() else "openai/clip-vit-base-patch32"


class FrozenBERTTextEncoder(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        from transformers import logging

        logging.set_verbosity_error()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.text_model = AutoModel.from_pretrained(model_path)
        self.eval()
        self.text_model.training = False
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, texts):
        encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        output = self.text_model(**encoded_inputs.to(self.text_model.device)).last_hidden_state
        mask = encoded_inputs.attention_mask.to(dtype=bool)
        return output, mask


def load_bert(model_path):
    return FrozenBERTTextEncoder(model_path)


def unpack_text_embed(enc_text):
    return enc_text["ctxt"], enc_text["ctxt_mask"], enc_text["vtxt"]


class TextConditionEncoder(nn.Module):
    def __init__(
        self,
        *,
        encoder_type: str,
        dataset: str,
        clip_version: str = "ViT-B/32",
        cache_text_embeddings: bool = False,
        bert_model_path: str = _DEFAULT_BERT_MODEL_PATH,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.dataset = dataset
        self.cache_text_embeddings = bool(cache_text_embeddings)
        self._text_cache_dtype = torch.float16
        self._bert_text_cache = {}
        self._clip_text_cache = {}

        if self.encoder_type != "bert_clip":
            raise ValueError(f"Unsupported text encoder type: {self.encoder_type!r}")

        self.bert_text_model = load_bert(bert_model_path)
        self.token_dim = self.bert_text_model.text_model.config.hidden_size
        self.clip_tokenizer, self.clip_text_model = self._load_frozen_clip(clip_version)
        self.global_dim = self._infer_clip_dim(self.clip_text_model)

    @staticmethod
    def _infer_clip_dim(clip_model):
        return clip_model.config.projection_dim

    @staticmethod
    def _load_frozen_clip(clip_version):
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

        if clip_version != "ViT-B/32":
            raise ValueError(f"Unsupported CLIP version: {clip_version!r}")
        model_id = _CLIP_B32_MODEL_ID
        tokenizer = CLIPTokenizer.from_pretrained(model_id)
        clip_model = CLIPTextModelWithProjection.from_pretrained(model_id)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return tokenizer, clip_model

    def _normalize_text_batch(self, raw_text):
        if isinstance(raw_text, str):
            return [raw_text]
        return list(raw_text)

    def clear_cache(self):
        self._bert_text_cache.clear()
        self._clip_text_cache.clear()

    def _cache_storage_dtype(self):
        return self._text_cache_dtype

    def _bert_target_device(self):
        return self.bert_text_model.text_model.device

    def _clip_target_device(self):
        return self.clip_text_model.device

    def _clip_encode_with_model(self, texts):
        device = self._clip_target_device()
        max_text_len = 20 if self.dataset in ("humanml", "hy263", "humanml201") else None
        if max_text_len is not None:
            default_context_length = self.clip_tokenizer.model_max_length
            context_length = max_text_len + 2
            assert context_length < default_context_length
            encoded = self.clip_tokenizer(
                texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=context_length,
            )
            if context_length < default_context_length:
                pad_len = default_context_length - context_length
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]
                pad_ids = torch.zeros((input_ids.shape[0], pad_len), dtype=input_ids.dtype)
                pad_mask = torch.zeros((attention_mask.shape[0], pad_len), dtype=attention_mask.dtype)
                encoded["input_ids"] = torch.cat([input_ids, pad_ids], dim=1)
                encoded["attention_mask"] = torch.cat([attention_mask, pad_mask], dim=1)
        else:
            encoded = self.clip_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        return self.clip_text_model(**encoded).text_embeds.float().unsqueeze(0)

    def _store_bert_cache_entries(self, texts):
        missing = [text for text in dict.fromkeys(texts) if text not in self._bert_text_cache]
        if not missing:
            return
        enc_text, valid_mask = self.bert_text_model(missing)
        enc_text = enc_text.detach().to(device="cpu", dtype=self._cache_storage_dtype())
        valid_mask = valid_mask.detach().to(device="cpu")
        for idx, text in enumerate(missing):
            valid_len = int(valid_mask[idx].sum().item())
            self._bert_text_cache[text] = enc_text[idx, :valid_len].contiguous()

    def _assemble_cached_bert_batch(self, texts):
        device = self._bert_target_device()
        target_dtype = next(self.bert_text_model.parameters()).dtype
        max_len = max(self._bert_text_cache[text].shape[0] for text in texts)
        feat_dim = self._bert_text_cache[texts[0]].shape[-1]
        enc_text = torch.zeros((max_len, len(texts), feat_dim), device=device, dtype=target_dtype)
        pad_mask = torch.ones((len(texts), max_len), device=device, dtype=torch.bool)
        for batch_idx, text in enumerate(texts):
            cached = self._bert_text_cache[text].to(device=device, dtype=target_dtype)
            valid_len = cached.shape[0]
            enc_text[:valid_len, batch_idx] = cached
            pad_mask[batch_idx, :valid_len] = False
        return enc_text, pad_mask

    def _store_clip_cache_entries(self, texts):
        missing = [text for text in dict.fromkeys(texts) if text not in self._clip_text_cache]
        if not missing:
            return
        global_text = self._clip_encode_with_model(missing)
        global_text = global_text.squeeze(0).detach().to(device="cpu", dtype=self._cache_storage_dtype())
        for idx, text in enumerate(missing):
            self._clip_text_cache[text] = global_text[idx].contiguous()

    def _assemble_cached_clip_batch(self, texts):
        device = self._clip_target_device()
        target_dtype = next(self.clip_text_model.parameters()).dtype
        global_text = torch.stack(
            [self._clip_text_cache[text].to(device=device, dtype=target_dtype) for text in texts],
            dim=0,
        )
        return global_text.unsqueeze(0)

    def encode_bert(self, raw_text):
        texts = self._normalize_text_batch(raw_text)
        if not self.cache_text_embeddings:
            enc_text, valid_mask = self.bert_text_model(texts)
            return enc_text.permute(1, 0, 2), ~valid_mask
        self._store_bert_cache_entries(texts)
        return self._assemble_cached_bert_batch(texts)

    def encode_clip(self, raw_text):
        texts = self._normalize_text_batch(raw_text)
        if not self.cache_text_embeddings:
            return self._clip_encode_with_model(texts)
        self._store_clip_cache_entries(texts)
        return self._assemble_cached_clip_batch(texts)

    def encode(self, raw_text):
        ctxt, ctxt_mask = self.encode_bert(raw_text)
        return {
            "ctxt": ctxt,
            "ctxt_mask": ctxt_mask,
            "vtxt": self.encode_clip(raw_text),
        }
