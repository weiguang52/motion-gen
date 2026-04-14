"""
Brief:
Create DiT / AR-DiT models, configure DDPM sampling, and load checkpoints.
"""
import torch

from diffusion import gaussian_diffusion as gd
from model.mdm import MDM


def load_model_wo_clip(model, state_dict):
    state_dict = dict(state_dict)
    state_dict.pop("sequence_pos_encoder.pe", None)
    state_dict.pop("embed_timestep.sequence_pos_encoder.pe", None)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    allowed_missing_prefixes = (
        "text_conditioner.bert_text_model.",
        "text_conditioner.clip_text_model.",
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

    if args.dataset != "humanml":
        raise ValueError(f"Unsupported dataset [{args.dataset}]")

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
        "cond_mode": "text",
        "cond_mask_prob": args.cond_mask_prob,
        "arch": args.arch,
        "clip_version": clip_version,
        "dataset": args.dataset,
        "text_refiner_layers": getattr(args, "text_refiner_layers", 2),
        "text_refiner_heads": getattr(args, "text_refiner_heads", 4),
        "cache_text_embeddings": getattr(args, "cache_text_embeddings", False),
        "time_embed_mode": "ddpm",
        "diffusion_steps": args.diffusion_steps,
        "rescale_timesteps": False,
        "pos_embed_max_len": args.pos_embed_max_len,
        "pred_len": args.pred_len,
        "context_len": args.context_len,
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
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
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
