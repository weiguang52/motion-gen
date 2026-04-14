"""
Brief:
Define reusable argument-parsing helpers for DDPM DiT / AR-DiT inference.
"""
import argparse
import json
import os


DEFAULT_SAMPLING_STEPS = 50


def default_guidance_for_arch(arch: str) -> float:
    return 4.0 if arch == "ar_dit" else 2.5


def load_args_from_model(args, args_to_overwrite, skip_overwrite_dests=None):
    model_path = args.model_path
    args_path = os.path.join(os.path.dirname(model_path), "args.json")
    assert os.path.exists(args_path), "Arguments json file was not found!"
    with open(args_path, "r") as fr:
        model_args = json.load(fr)

    skip_overwrite_dests = set(skip_overwrite_dests or ())
    for arg_name in args_to_overwrite:
        if arg_name in skip_overwrite_dests:
            continue
        if arg_name in model_args:
            setattr(args, arg_name, model_args[arg_name])
    return args


def apply_rules(args):
    if getattr(args, "layers", None) is None:
        args.layers = 8
    if getattr(args, "ff_size", None) is None:
        args.ff_size = 1024
    if getattr(args, "num_heads", None) is None:
        args.num_heads = 4
    if getattr(args, "dropout", None) is None:
        args.dropout = 0.1

    if getattr(args, "pred_len", 0) == 0:
        args.pred_len = getattr(args, "context_len", 0)
    if getattr(args, "context_len", 0) > 0 and getattr(args, "pred_len", 0) > 0:
        args.autoregressive = True
    if getattr(args, "autoregressive_init", "isaac") == "data":
        args.autoregressive_init = "isaac"
    if getattr(args, "guidance_param", None) is None:
        args.guidance_param = default_guidance_for_arch(getattr(args, "arch", "dit"))
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    raise ValueError("group_name was not found.")


def add_base_options(parser):
    group = parser.add_argument_group("base")
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--local_rank", default=-1, type=int, help="DDP local rank.")
    group.add_argument("--seed", default=10, type=int, help="Random seed.")
    group.add_argument("--batch_size", default=64, type=int, help="Batch size.")


def add_diffusion_options(parser):
    group = parser.add_argument_group("diffusion")
    group.add_argument("--noise_schedule", default="cosine", choices=["linear", "cosine"], type=str)
    group.add_argument("--diffusion_steps", default=1000, type=int)
    group.add_argument("--sigma_small", default=True, type=bool)
    group.add_argument("--sampling_steps", default=DEFAULT_SAMPLING_STEPS, type=int)


def add_model_options(parser):
    group = parser.add_argument_group("model")
    group.add_argument("--arch", default="dit", choices=["dit", "ar_dit"], type=str)
    group.add_argument("--layers", default=None, type=int)
    group.add_argument("--model_dim", default=512, type=int)
    group.add_argument("--ff_size", default=None, type=int)
    group.add_argument("--num_heads", default=None, type=int)
    group.add_argument("--dropout", default=None, type=float)
    group.add_argument("--text_refiner_layers", default=2, type=int)
    group.add_argument("--text_refiner_heads", default=4, type=int)
    group.add_argument("--cache_text_embeddings", dest="cache_text_embeddings", action="store_true", default=False)
    group.add_argument("--disable_text_embedding_cache", dest="cache_text_embeddings", action="store_false")
    group.add_argument("--cond_mask_prob", default=0.1, type=float)
    group.add_argument("--lambda_rcxyz", default=0.0, type=float)
    group.add_argument("--lambda_vel", default=0.0, type=float)
    group.add_argument("--lambda_fc", default=0.0, type=float)
    group.add_argument("--lambda_target_loc", default=0.0, type=float)
    group.add_argument("--pos_embed_max_len", default=5000, type=int)
    group.add_argument("--use_ema", action="store_true")
    group.add_argument("--context_len", default=0, type=int)
    group.add_argument("--pred_len", default=0, type=int)


def add_data_options(parser):
    group = parser.add_argument_group("dataset")
    group.add_argument("--dataset", default="humanml", choices=["humanml"], type=str)
    group.add_argument("--data_dir", default="", type=str)


def add_sampling_options(parser):
    group = parser.add_argument_group("sampling")
    group.add_argument("--model_path", required=True, type=str)
    group.add_argument("--num_samples", default=1, type=int)
    group.add_argument("--num_repetitions", default=1, type=int)
    group.add_argument("--guidance_param", default=None, type=float)
    group.add_argument("--autoregressive", action="store_true")
    group.add_argument("--autoregressive_include_prefix", action="store_true")
    group.add_argument("--autoregressive_init", default="isaac", choices=["data", "isaac"], type=str)
    group.add_argument("--autoregressive_motion_blend_frames", default=12, type=int)


def add_generate_options(parser):
    group = parser.add_argument_group("generate")
    group.add_argument("--motion_length", default=9.8, type=float)
    group.add_argument("--input_text", default="", type=str)
    group.add_argument("--dynamic_text_path", default="", type=str)
    group.add_argument("--dynamic_text_blend", default=0.35, type=float)
    group.add_argument("--text_prompt", default="", type=str)
    group.add_argument("--init_prefix_npy", default="", type=str)
