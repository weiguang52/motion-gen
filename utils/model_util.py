"""
Brief:
Create DiT / AR-DiT models, configure DDPM sampling, and load checkpoints.
"""
import inspect

import torch

from diffusion import gaussian_diffusion as gd
from model.mdm import MDM
from utils.parser_util import get_cond_mode


def _uses_ar_dit_dual_cross_attn_keys(state_dict):
    return any(".cross_attn_text." in key for key in state_dict.keys())


def checkpoint_uses_ar_dit_dual_cross_attn(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model_avg" in checkpoint:
            checkpoint = checkpoint["model_avg"]
        elif "model" in checkpoint:
            checkpoint = checkpoint["model"]
    return _uses_ar_dit_dual_cross_attn_keys(checkpoint)


def _upgrade_ar_dit_dual_cross_attn_state_dict(model, state_dict):
    expected_keys = set(model.state_dict().keys())
    if not any(".cross_attn_text." in key for key in expected_keys):
        return state_dict

    upgraded = dict(state_dict)
    for old_key in list(state_dict.keys()):
        new_keys = []
        if ".cross_attn." in old_key:
            new_keys = [
                old_key.replace(".cross_attn.", ".cross_attn_text."),
                old_key.replace(".cross_attn.", ".cross_attn_prefix."),
            ]
        elif ".cross_norm." in old_key:
            new_keys = [
                old_key.replace(".cross_norm.", ".cross_norm_text."),
                old_key.replace(".cross_norm.", ".cross_norm_prefix."),
            ]
        elif ".adaLN_modulation.1.weight" in old_key or ".adaLN_modulation.1.bias" in old_key:
            old_tensor = state_dict[old_key]
            if old_tensor.shape[0] % 9 != 0:
                continue
            chunk_size = old_tensor.shape[0] // 9
            chunks = old_tensor.split(chunk_size, dim=0)
            expanded = torch.cat([*chunks[:6], *chunks[6:9], *chunks[6:9]], dim=0)
            if expanded.shape != model.state_dict().get(old_key, expanded).shape:
                continue
            upgraded[old_key] = expanded
            continue
        if not new_keys:
            continue
        if not all(new_key in expected_keys for new_key in new_keys):
            continue
        for new_key in new_keys:
            upgraded.setdefault(new_key, state_dict[old_key].clone())
        del upgraded[old_key]
    return upgraded


def _remap_legacy_text_encoder_state_dict(state_dict):
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("bert_text_model."):
            new_key = "text_conditioner.bert_text_model." + key[len("bert_text_model."):]
        elif key.startswith("bert_model."):
            new_key = "text_conditioner.bert_text_model." + key[len("bert_model."):]
        elif key.startswith("clip_model."):
            new_key = "text_conditioner.bert_text_model." + key[len("clip_model."):]
        remapped[new_key] = value
    return remapped


def load_model_wo_clip(model, state_dict):
    model_expects_dual = _uses_ar_dit_dual_cross_attn_keys(model.state_dict())
    checkpoint_uses_dual = _uses_ar_dit_dual_cross_attn_keys(state_dict)
    state_dict = dict(state_dict)
    state_dict.pop("sequence_pos_encoder.pe", None)
    state_dict.pop("embed_timestep.sequence_pos_encoder.pe", None)
    state_dict = _remap_legacy_text_encoder_state_dict(state_dict)

    if model_expects_dual and not checkpoint_uses_dual:
        state_dict = _upgrade_ar_dit_dual_cross_attn_state_dict(model, state_dict)
    elif checkpoint_uses_dual and not model_expects_dual:
        raise ValueError(
            "Checkpoint uses AR-DiT dual cross-attention weights, but the current model was built "
            "without dual cross-attention. Enable --enable_ar_dit_dual_cross_attn for this checkpoint."
        )

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    allowed_missing_prefixes = (
        "text_conditioner.bert_text_model.",
        "text_conditioner.clip_text_model.",
        "clip_model.",
        "bert_text_model.",
        "bert_model.",
    )
    assert all(
        key.startswith(allowed_missing_prefixes)
        or "sequence_pos_encoder" in key
        or "q_zero" in key
        for key in missing_keys
    )


def create_model_and_diffusion(args, data):
    del data
    model = MDM(**get_model_args(args))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def get_model_args(args):
    clip_version = "ViT-B/32"
    cond_mode = get_cond_mode(args)

    if args.dataset != "humanml":
        raise ValueError(f"Unsupported dataset [{args.dataset}]")

    if not hasattr(args, "pred_len"):
        args.pred_len = 0
    if not hasattr(args, "context_len"):
        args.context_len = 0

    return {
        "njoints": 263,
        "nfeats": 1,
        "translation": True,
        "pose_rep": "rot6d",
        "glob": True,
        "model_dim": args.model_dim,
        "ff_size": args.ff_size,
        "num_layers": args.layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "activation": "gelu",
        "data_rep": "hml_vec",
        "cond_mode": cond_mode,
        "cond_mask_prob": args.cond_mask_prob,
        "arch": args.arch,
        "clip_version": clip_version,
        "dataset": args.dataset,
        "text_encoder_type": args.text_encoder_type,
        "text_refiner_layers": getattr(args, "text_refiner_layers", 2),
        "text_refiner_heads": getattr(args, "text_refiner_heads", 4),
        "cache_text_embeddings": getattr(args, "cache_text_embeddings", False),
        "time_embed_mode": "ddpm",
        "diffusion_steps": args.diffusion_steps,
        "rescale_timesteps": False,
        "pos_embed_max_len": args.pos_embed_max_len,
        "pred_len": args.pred_len,
        "context_len": args.context_len,
        "ar_dit_dual_cross_attn": getattr(args, "ar_dit_dual_cross_attn", False),
    }


def create_gaussian_diffusion(args):
    predict_xstart = True
    steps = args.diffusion_steps
    scale_beta = 1.0
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    return gd.GaussianDiffusion(
        betas=betas,
        model_mean_type=gd.ModelMeanType.START_X if predict_xstart else gd.ModelMeanType.EPSILON,
        model_var_type=(
            gd.ModelVarType.FIXED_SMALL if args.sigma_small else gd.ModelVarType.FIXED_LARGE
        )
        if not learn_sigma
        else gd.ModelVarType.LEARNED_RANGE,
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def load_saved_model(model, model_path, use_avg: bool = False):
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    state_dict = torch.load(model_path, **load_kwargs)
    if use_avg and "model_avg" in state_dict:
        print("loading avg model")
        state_dict = state_dict["model_avg"]
    elif "model" in state_dict:
        print("loading model without avg")
        state_dict = state_dict["model"]
    else:
        print("checkpoint has no avg model, loading as usual.")
    load_model_wo_clip(model, state_dict)
    return model
