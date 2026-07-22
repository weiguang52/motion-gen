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

from data_loaders.humanml.scripts.motion_process import recover_from_ric_stateful
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
    load_args_from_model,
)
from utils.sampler_util import (
    AutoRegressiveSampler,
    ClassifierFreeSampleModel,
    repeat_text_embed_for_dynamic,
    smooth_continuation_boundary,
)


_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_AR_DIT_INIT_PREFIX_NPY = os.path.join(_REPO_ROOT, "assets", "ar_dit_zero_vel_stand_prefix.npy")


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


def _load_continuation_prefix(
    path: str,
    mean: np.ndarray,
    std: np.ndarray,
    prefix_frames: int,
    expected_dim: int,
):
    """Load the final HumanML frames and decode state from a previous result."""
    if not path:
        return None, None
    if prefix_frames <= 0:
        raise SystemExit("continuation_prefix_frames must be greater than zero.")
    if not os.path.exists(path):
        raise SystemExit(f"continuation_npy not found: {path}")

    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.shape == ():
        loaded = loaded.item()
    decode_state = None
    if isinstance(loaded, dict):
        if "hml_vec" not in loaded:
            raise SystemExit(
                "continuation_npy must contain hml_vec. Re-run infer.py with the updated code "
                "or provide a raw [T, D] HumanML array."
            )
        state_data = loaded.get("decode_state")
        if isinstance(state_data, dict) and {"root_rot_ang", "root_pos"} <= state_data.keys():
            decode_state = {
                "root_rot_ang": torch.as_tensor(state_data["root_rot_ang"]).float()[:1],
                "root_pos": torch.as_tensor(state_data["root_pos"]).float()[:1],
            }
        loaded = loaded["hml_vec"]

    hml_vec = np.asarray(loaded)
    if hml_vec.ndim == 3:
        hml_vec = hml_vec[0]
    if hml_vec.ndim != 2:
        raise SystemExit(f"Expected continuation hml_vec shape [T, D] or [B, T, D], got {hml_vec.shape}")
    if hml_vec.shape[1] != expected_dim:
        raise SystemExit(
            f"Expected continuation feature dim {expected_dim}, got {hml_vec.shape[1]}"
        )
    if hml_vec.shape[0] == 0:
        raise SystemExit("continuation_npy contains no motion frames.")

    full_hml_vec = torch.from_numpy(hml_vec).float()
    if decode_state is None:
        _, decode_state = recover_from_ric_stateful(
            full_hml_vec.unsqueeze(0).unsqueeze(0),
            22,
        )
    hml_vec = full_hml_vec[-prefix_frames:]
    mean_t = torch.from_numpy(mean).float()
    std_t = torch.from_numpy(std).float()
    prefix = ((hml_vec - mean_t) / std_t).T.unsqueeze(0).unsqueeze(2).contiguous()
    return prefix, decode_state


def _expand_decode_state(state, batch_size):
    if state is None:
        return None
    expanded = {}
    for key in ("root_rot_ang", "root_pos"):
        value = state[key]
        if value.shape[0] == 1 and batch_size > 1:
            value = value.expand(batch_size, *value.shape[1:])
        elif value.shape[0] != batch_size:
            raise SystemExit(
                f"continuation decode state batch size {value.shape[0]} does not match {batch_size}."
            )
        expanded[key] = value
    return expanded


def _build_known_prefix(prefix: torch.Tensor, total_frames: int, device):
    prefix = prefix.to(device)
    prefix_len = prefix.shape[-1]
    known_motion = torch.zeros(
        prefix.shape[0], prefix.shape[1], prefix.shape[2], total_frames,
        dtype=prefix.dtype,
        device=device,
    )
    known_motion[..., :prefix_len] = prefix
    known_mask = torch.zeros_like(known_motion, dtype=torch.bool)
    known_mask[..., :prefix_len] = True
    return known_motion, known_mask


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
    if args.num_frames > 0:
        return args.num_frames
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
    if args.num_frames < 0:
        raise SystemExit("num_frames must be >= 0.")
    if args.continuation_prefix_frames <= 0:
        raise SystemExit("continuation_prefix_frames must be greater than zero.")
    if args.continuation_blend_frames < 0:
        raise SystemExit("continuation_blend_frames must be >= 0.")
    if args.continuation_npy and args.arch != "dit":
        raise SystemExit("continuation_npy is only supported by a plain dit checkpoint.")

    if args.dataset != "humanml":
        raise SystemExit("Only HumanML checkpoints are supported in this script.")
    if args.arch not in {"dit", "ar_dit"}:
        raise SystemExit(f"Only dit/ar_dit checkpoints are supported here, got arch={args.arch!r}.")
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

    continuation_prefix, continuation_decode_state = _load_continuation_prefix(
        args.continuation_npy,
        mean,
        std,
        args.continuation_prefix_frames,
        int(mean.shape[0]),
    )
    continuation_decode_state = _expand_decode_state(
        continuation_decode_state,
        args.batch_size,
    )
    prefix_frames = 0 if continuation_prefix is None else continuation_prefix.shape[-1]
    sample_frames = n_frames + prefix_frames
    if not args.autoregressive and sample_frames > max_frames:
        raise SystemExit(
            f"Requested {n_frames} new frames plus {prefix_frames} prefix frames, "
            f"but this DiT supports at most {max_frames} total frames."
        )

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, _DummyData())

    print(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path, use_avg=args.use_ema)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)
    model.to(dist_util.dev())
    model.eval()

    model_kwargs = _build_model_kwargs(model, args, texts, sample_frames, mean, std)
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

    model_sample_frames = (
        sample_frames if args.autoregressive or args.context_len == 0 else args.pred_len
    )
    motion_shape = (args.batch_size, model.njoints, model.nfeats, model_sample_frames)
    inpaint_kwargs = {}
    if continuation_prefix is not None:
        continuation_prefix = continuation_prefix.expand(args.batch_size, -1, -1, -1)
        known_motion, known_mask = _build_known_prefix(
            continuation_prefix,
            model_sample_frames,
            dist_util.dev(),
        )
        inpaint_kwargs = {"known_motion": known_motion, "known_mask": known_mask}

    all_motions = []
    all_hml_vec = []
    all_root_rot_ang = []
    all_root_pos = []
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
                **inpaint_kwargs,
            )

            if prefix_frames:
                suffix = sample[..., prefix_frames:]
                sample = smooth_continuation_boundary(
                    continuation_prefix,
                    suffix,
                    args.continuation_blend_frames,
                )
            hml_vec = sample.cpu().permute(0, 2, 3, 1)
            hml_vec = (hml_vec * std) + mean
            decoded, next_decode_state = recover_from_ric_stateful(
                hml_vec.float(),
                22,
                state=continuation_decode_state,
            )
            decoded = decoded.view(-1, *decoded.shape[2:]).permute(0, 2, 3, 1)

            all_motions.append(decoded.cpu().numpy())
            all_hml_vec.append(hml_vec.squeeze(1).cpu().numpy())
            all_root_rot_ang.append(next_decode_state["root_rot_ang"].cpu().numpy())
            all_root_pos.append(next_decode_state["root_pos"].cpu().numpy())
            if args.autoregressive:
                lengths = np.full((args.batch_size,), n_frames, dtype=np.int64)
            elif args.context_len > 0:
                lengths = np.full((args.batch_size,), sample.shape[-1], dtype=np.int64)
            else:
                lengths = np.full((args.batch_size,), n_frames, dtype=np.int64)
            all_lengths.append(lengths)
            all_text += model_kwargs["y"]["text"]

    total_num_samples = args.num_samples * args.num_repetitions
    all_motions = np.concatenate(all_motions, axis=0)[:total_num_samples]
    all_hml_vec = np.concatenate(all_hml_vec, axis=0)[:total_num_samples]
    all_root_rot_ang = np.concatenate(all_root_rot_ang, axis=0)[:total_num_samples]
    all_root_pos = np.concatenate(all_root_pos, axis=0)[:total_num_samples]
    all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
    all_text = all_text[:total_num_samples]

    results = {
        "motion": all_motions,
        "hml_vec": all_hml_vec,
        "text": all_text,
        "lengths": all_lengths,
        "fps": fps,
        "num_samples": args.num_samples,
        "num_repetitions": args.num_repetitions,
        "model_path": args.model_path,
        "continuation_prefix_frames": prefix_frames,
        "continuation_blend_frames": min(args.continuation_blend_frames, n_frames) if prefix_frames else 0,
        "decode_state": {
            "root_rot_ang": all_root_rot_ang,
            "root_pos": all_root_pos,
        },
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
