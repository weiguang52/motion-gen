#!/usr/bin/env python3
"""
Brief:
Run local DDPM DiT / AR-DiT inference and save decoded HumanML motions.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders.humanml.utils import paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
from data_loaders.tensors import collate
from utils import dist_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.parser_util import (
    add_base_options,
    add_data_options,
    add_diffusion_options,
    add_generate_options,
    add_model_options,
    add_sampling_options,
    apply_rules,
    default_guidance_for_arch,
    get_args_per_group_name,
    get_cond_mode,
    load_args_from_model,
)
from utils.sampler_util import (
    AutoRegressiveSampler,
    ClassifierFreeSampleModel,
    repeat_text_embed_for_dynamic,
)


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_AR_DIT_INIT_PREFIX_NPY = os.path.join(_REPO_ROOT, "assets", "ar_dit_default_stand_prefix.npy")


class _DummyData:
    dataset = type("Dataset", (), {})()


def _coerce_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes")
    return bool(value)


def _read_model_args(model_path: str) -> dict:
    args_path = os.path.join(os.path.dirname(model_path), "args.json")
    if not os.path.exists(args_path):
        return {}
    with open(args_path, "r") as fr:
        return json.load(fr)


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDPM HumanML text inference for DiT / AR-DiT.")
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    parser.add_argument("--out", default="outputs/sample.npy", type=str, help="Output .npy path.")
    parser.add_argument(
        "--force_use_ema",
        action="store_true",
        help="Force EMA weights if present in the checkpoint.",
    )
    parser.add_argument(
        "--mp4_dir",
        default="",
        type=str,
        help="Directory to save MP4 files (set to enable rendering).",
    )
    args = parser.parse_args()

    args_to_overwrite = []
    for group_name in ["dataset", "model", "diffusion"]:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)
    args = load_args_from_model(args, args_to_overwrite)

    model_args = _read_model_args(args.model_path)
    if "autoregressive" in model_args:
        args.autoregressive = _coerce_bool(model_args["autoregressive"])
    explicit_guidance = any(
        token == "--guidance_param" or token.startswith("--guidance_param=")
        for token in sys.argv[1:]
    )
    if not explicit_guidance and "gen_guidance_param" in model_args:
        args.guidance_param = float(model_args["gen_guidance_param"])

    if args.force_use_ema:
        args.use_ema = True
    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    args = apply_rules(args)
    if not explicit_guidance and "gen_guidance_param" not in model_args:
        args.guidance_param = default_guidance_for_arch(args.arch)
    return args


def _resolve_data_root(args: argparse.Namespace) -> str:
    if getattr(args, "data_dir", ""):
        return args.data_dir
    return "./dataset/HumanML3D"


def _resolve_init_prefix_npy(args: argparse.Namespace) -> str:
    if getattr(args, "init_prefix_npy", ""):
        return args.init_prefix_npy
    if (
        getattr(args, "arch", "") == "ar_dit"
        and getattr(args, "autoregressive", False)
        and int(getattr(args, "context_len", 0)) > 0
        and getattr(args, "autoregressive_init", "isaac") == "isaac"
    ):
        return _DEFAULT_AR_DIT_INIT_PREFIX_NPY
    return ""


def _load_init_prefix_from_npy(path: str, mean: np.ndarray, std: np.ndarray, context_len: int, expected_dim: int):
    if not path:
        return None
    if context_len <= 0:
        raise SystemExit("init_prefix_npy requires context_len > 0.")
    if not os.path.exists(path):
        raise SystemExit(f"init_prefix_npy not found: {path}")

    arr = np.load(path)
    if arr.ndim != 2:
        raise SystemExit(f"Expected init_prefix_npy shape [T, D], got {arr.shape}")
    if arr.shape[1] != expected_dim:
        raise SystemExit(f"Expected init_prefix_npy feature dim {expected_dim}, got {arr.shape[1]}")

    prefix = torch.from_numpy(arr[:context_len]).float()
    if prefix.shape[0] < context_len:
        if prefix.shape[0] > 0:
            pad = prefix[-1:].repeat(context_len - prefix.shape[0], 1)
        else:
            pad = torch.zeros(context_len, expected_dim)
        prefix = torch.cat([prefix, pad], dim=0)

    mean_t = torch.from_numpy(mean).float()
    std_t = torch.from_numpy(std).float()
    prefix = (prefix - mean_t) / std_t
    return prefix.T.unsqueeze(1).contiguous()


def _build_model_kwargs(model, args, texts, n_frames, mean, std):
    if args.context_len > 0:
        inp = torch.zeros(model.njoints, model.nfeats, args.pred_len)
        prefix_path = _resolve_init_prefix_npy(args)
        prefix = _load_init_prefix_from_npy(prefix_path, mean, std, int(args.context_len), int(model.njoints))
        if prefix is None:
            prefix = torch.zeros(model.njoints, model.nfeats, args.context_len)
        collate_args = [{"inp": inp, "prefix": prefix, "lengths": args.pred_len}] * args.num_samples
    else:
        collate_args = [{"inp": torch.zeros(n_frames), "lengths": n_frames}] * args.num_samples

    collate_args = [dict(arg, text=txt) for arg, txt in zip(collate_args, texts)]
    _, model_kwargs = collate(collate_args)
    model_kwargs["y"] = {
        key: val.to(dist_util.dev()) if torch.is_tensor(val) else val
        for key, val in model_kwargs["y"].items()
    }
    return model_kwargs


def _read_text_lines(path: str):
    if not os.path.exists(path):
        raise SystemExit(f"text file not found: {path}")
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    return [line for line in lines if line]


def _prepare_text_condition(model, texts, dynamic_texts, num_samples):
    if dynamic_texts is None:
        text_embed = model.encode_text(texts)
        return texts, text_embed

    text_embed = model.encode_text(dynamic_texts)
    return [list(dynamic_texts) for _ in range(num_samples)], repeat_text_embed_for_dynamic(text_embed, num_samples)


def _expand_dynamic_caption(caption, length, pred_len):
    if not isinstance(caption, list):
        return caption
    if pred_len <= 0:
        return [caption[min(i, len(caption) - 1)] for i in range(length)]
    title = []
    for segment_text in caption:
        title.extend([segment_text] * pred_len)
        if len(title) >= length:
            break
    if not title:
        return ""
    if len(title) < length:
        title.extend([title[-1]] * (length - len(title)))
    return title[:length]


def _resolve_target_frames(args, fps, max_frames, dynamic_texts):
    if dynamic_texts is not None:
        return len(dynamic_texts) * args.pred_len
    target_frames = int(args.motion_length * fps)
    if args.autoregressive:
        return target_frames
    return min(max_frames, target_frames)


def main():
    args = _build_args()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    if not 0.0 <= args.dynamic_text_blend <= 1.0:
        raise SystemExit("dynamic_text_blend must be between 0 and 1.")
    if args.autoregressive_motion_blend_frames < 0:
        raise SystemExit("autoregressive_motion_blend_frames must be >= 0.")

    if get_cond_mode(args) != "text":
        raise SystemExit("Only text-conditioned models are supported.")
    if args.dataset != "humanml":
        raise SystemExit("Only HumanML checkpoints are supported in this script.")
    if args.arch not in {"dit", "ar_dit"}:
        raise SystemExit(f"Only dit/ar_dit checkpoints are supported here, got arch={args.arch!r}.")
    if args.text_encoder_type != "bert_clip":
        raise SystemExit(f"Only bert_clip text encoder is supported, got {args.text_encoder_type!r}.")
    if args.autoregressive and args.arch != "ar_dit":
        raise SystemExit("Autoregressive sampling requires an ar_dit checkpoint.")
    if args.autoregressive and (args.context_len <= 0 or args.pred_len <= 0):
        raise SystemExit("Autoregressive sampling requires context_len > 0 and pred_len > 0.")
    if not args.text_prompt and not args.input_text and not args.dynamic_text_path:
        raise SystemExit("Provide --text_prompt, --input_text, or --dynamic_text_path.")

    max_frames = 196
    fps = 20
    dynamic_texts = None
    if args.text_prompt:
        texts = [args.text_prompt] * args.num_samples
    else:
        if args.dynamic_text_path:
            if not args.autoregressive:
                raise SystemExit("Dynamic text sampling requires --autoregressive.")
            dynamic_texts = _read_text_lines(args.dynamic_text_path)
            if not dynamic_texts:
                raise SystemExit("dynamic_text_path is empty.")
            texts = [dynamic_texts[0]] * args.num_samples
        else:
            texts = _read_text_lines(args.input_text)
            args.num_samples = len(texts)
    n_frames = _resolve_target_frames(args, fps, max_frames, dynamic_texts)

    if n_frames <= 0:
        raise SystemExit("motion_length too small for current FPS.")

    args.batch_size = args.num_samples

    data_root = _resolve_data_root(args)
    mean = np.load(os.path.join(data_root, "Mean.npy"))
    std = np.load(os.path.join(data_root, "Std.npy"))

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, _DummyData())

    print(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path, use_avg=args.use_ema)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()

    model_kwargs = _build_model_kwargs(model, args, texts, n_frames, mean, std)
    if args.guidance_param != 1:
        model_kwargs["y"]["scale"] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
    model_kwargs["y"]["text"], model_kwargs["y"]["text_embed"] = _prepare_text_condition(
        model,
        model_kwargs["y"]["text"],
        dynamic_texts,
        args.num_samples,
    )

    def base_sample_fn(model_obj, shape, **sample_kwargs):
        return diffusion.logsnr_sample_loop(
            model_obj,
            shape,
            steps=int(getattr(args, "sampling_steps", 50)),
            **sample_kwargs,
        )

    sample_fn = base_sample_fn
    if args.autoregressive:
        sample_fn = AutoRegressiveSampler(args, base_sample_fn, n_frames).sample

    sample_frames = n_frames if args.autoregressive or args.context_len == 0 else args.pred_len
    motion_shape = (args.batch_size, model.njoints, model.nfeats, sample_frames)

    all_motions = []
    all_lengths = []
    all_text = []

    with torch.inference_mode():
        fixseed(args.seed)
        for rep_i in range(args.num_repetitions):
            print(f"Sampling repetition {rep_i + 1}/{args.num_repetitions}...")
            sample = sample_fn(
                model,
                motion_shape,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
            )

            sample = sample.cpu().permute(0, 2, 3, 1)
            sample = sample * std + mean
            sample = recover_from_ric(sample.float(), 22)
            sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

            all_motions.append(sample.cpu().numpy())
            if args.autoregressive:
                lengths = np.full((args.batch_size,), n_frames, dtype=np.int64)
            elif args.context_len > 0:
                lengths = np.full((args.batch_size,), sample.shape[-1], dtype=np.int64)
            else:
                lengths = model_kwargs["y"]["lengths"].cpu().numpy()
            all_lengths.append(lengths)
            all_text += model_kwargs["y"]["text"]

    total_num_samples = args.num_samples * args.num_repetitions
    all_motions = np.concatenate(all_motions, axis=0)[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
    all_text = all_text[:total_num_samples]

    results = {
        "motion": all_motions,
        "text": all_text,
        "lengths": all_lengths,
        "fps": fps,
        "num_samples": args.num_samples,
        "num_repetitions": args.num_repetitions,
        "model_path": args.model_path,
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(args.out, results)
    print(f"Saved results to {args.out}")

    mp4_dir = args.mp4_dir.strip()
    if mp4_dir:
        os.makedirs(mp4_dir, exist_ok=True)
        print(f"Rendering {total_num_samples} MP4 files to {mp4_dir}...")
        skeleton = paramUtil.t2m_kinematic_chain

        for rep_i in range(args.num_repetitions):
            for sample_i in range(args.num_samples):
                idx = rep_i * args.num_samples + sample_i
                motion = all_motions[idx].transpose(2, 0, 1)
                length = int(all_lengths[idx])
                motion = motion[:length]

                caption = _expand_dynamic_caption(all_text[idx], length, args.pred_len)
                save_file = f"sample{sample_i:02d}_rep{rep_i:02d}.mp4"
                save_path = os.path.join(mp4_dir, save_file)
                clip = plot_3d_motion(
                    save_path,
                    skeleton,
                    motion,
                    title=caption,
                    dataset=args.dataset,
                    fps=fps,
                )
                if clip is not None:
                    clip.write_videofile(save_path, fps=fps, threads=4, logger=None)
                    clip.close()

        print(f"Saved MP4 files to {mp4_dir}")


if __name__ == "__main__":
    main()
