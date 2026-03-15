"""
Brief:
Create the trans_dec model, configure DDPM sampling, and load checkpoints.
"""
import inspect

import torch

from diffusion import gaussian_diffusion as gd
from model.mdm import MDM
from utils.parser_util import get_cond_mode


def _remap_text_encoder_state_dict(state_dict):
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("bert_text_model."):
            new_key = "clip_model." + key[len("bert_text_model."):]
        elif key.startswith("bert_model."):
            new_key = "clip_model." + key[len("bert_model."):]
        remapped[new_key] = value
    return remapped


def load_model_wo_clip(model, state_dict):
    state_dict = dict(state_dict)
    state_dict.pop("sequence_pos_encoder.pe", None)
    state_dict.pop("embed_timestep.sequence_pos_encoder.pe", None)
    state_dict = _remap_text_encoder_state_dict(state_dict)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    allowed_missing_prefixes = ("clip_model.", "bert_text_model.", "bert_model.")
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

    emb_policy = args.__dict__.get("emb_policy", "add")
    return {
        "modeltype": "",
        "njoints": 263,
        "nfeats": 1,
        "translation": True,
        "pose_rep": "rot6d",
        "glob": True,
        "glob_rot": True,
        "model_dim": args.model_dim,
        "ff_size": 1024,
        "num_layers": args.layers,
        "num_heads": 4,
        "dropout": 0.1,
        "activation": "gelu",
        "data_rep": "hml_vec",
        "cond_mode": cond_mode,
        "cond_mask_prob": args.cond_mask_prob,
        "arch": args.arch,
        "emb_trans_dec": args.emb_trans_dec,
        "clip_version": clip_version,
        "dataset": args.dataset,
        "text_encoder_type": args.text_encoder_type,
        "pos_embed_max_len": args.pos_embed_max_len,
        "mask_frames": args.mask_frames,
        "pred_len": args.pred_len,
        "context_len": args.context_len,
        "emb_policy": emb_policy,
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
