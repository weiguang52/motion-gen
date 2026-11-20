"""
Brief:
Serve DDPM DiT / AR-DiT text-to-motion generation over FastAPI.
"""
from typing import Literal
import argparse
import copy
import io
import json
import os
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover
    ConfigDict = None

from data_loaders.humanml.scripts.motion_process import recover_from_ric_stateful
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


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_AR_DIT_INIT_PREFIX_NPY = os.path.join(_REPO_ROOT, "assets", "ar_dit_default_stand_prefix.npy")


class GenerateRequest(BaseModel):
    text: Optional[str] = Field(
        None,
        min_length=1,
        max_length=200,
        description="Single text prompt to generate a motion (max 200 characters).",
    )
    texts: Optional[List[str]] = Field(
        None,
        description="Optional dynamic prompts, one sentence per autoregressive segment.",
    )
    motion_length: Optional[float] = Field(
        None,
        gt=0,
        description="Optional motion length in seconds. If omitted, uses the model default.",
    )
    num_frames: Optional[int] = Field(
        None,
        gt=0,
        description="Exact number of new frames to return; overrides motion_length.",
    )
    init_prefix: Optional[Literal["stand_mean", "stand_data"]] = None
    blend_mode: Literal["position", "c1"] = "position"
    lookahead_frames: Optional[int] = Field(None, ge=0)
    is_final: bool = False
    session_id: str = Field(
        "default",
        min_length=1,
        max_length=128,
        description="Continuation state is isolated by this identifier.",
    )
    continue_from_previous: bool = Field(
        True,
        description="For plain DiT, use the previous result in this session as a fixed prefix.",
    )
    reset_session: bool = Field(
        False,
        description="Discard the previous result before generating this request.",
    )
    prefix_frames: Optional[int] = Field(
        None,
        gt=0,
        description="Override the server's fixed-prefix frame count for this request.",
    )
    blend_frames: Optional[int] = Field(
        None,
        ge=0,
        description="Override the number of new frames used for boundary smoothing.",
    )

    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")
    else:
        class Config:
            extra = "forbid"


@dataclass
class ServerState:
    args: argparse.Namespace
    model: torch.nn.Module
    guided_model: Optional[torch.nn.Module]
    diffusion: object
    mean: np.ndarray
    std: np.ndarray
    max_frames: int
    fps: float
    lock: threading.Lock
    continuation_prefix_frames: int
    continuation_blend_frames: int
    session_motions: Dict[str, torch.Tensor]
    session_decode_states: Dict[str, Dict[str, torch.Tensor]]


class _DummyData:
    dataset = type("Dataset", (), {})()


def _build_args(model_path: str, device: int, use_ema_override: Optional[bool]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args(["--model_path", model_path, "--device", str(device)])

    args_to_overwrite = []
    for group_name in ["dataset", "model", "diffusion"]:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)
    args = load_args_from_model(args, args_to_overwrite)

    if use_ema_override is not None:
        args.use_ema = use_ema_override
    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return apply_rules(args)


def _read_model_args(model_path: str) -> dict:
    args_path = os.path.join(os.path.dirname(model_path), "args.json")
    if not os.path.exists(args_path):
        return {}
    with open(args_path, "r") as fr:
        return json.load(fr)


def _coerce_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes")
    return bool(value)


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
        raise RuntimeError("init_prefix_npy requires context_len > 0.")
    if not os.path.exists(path):
        raise RuntimeError(f"init_prefix_npy not found: [{path}]")

    arr = np.load(path)
    if arr.ndim != 2:
        raise RuntimeError(f"Expected init_prefix_npy shape [T, D], got {arr.shape}")
    if arr.shape[1] != expected_dim:
        raise RuntimeError(f"Expected init_prefix_npy feature dim {expected_dim}, got {arr.shape[1]}")

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


def _build_model_kwargs(state: ServerState, args: argparse.Namespace, texts, n_frames):
    use_prefix = args.context_len > 0
    if use_prefix:
        inp = torch.zeros(state.model.njoints, state.model.nfeats, args.pred_len)
        prefix_path = _resolve_init_prefix_npy(args)
        prefix = _load_init_prefix_from_npy(
            prefix_path,
            state.mean,
            state.std,
            int(args.context_len),
            int(state.model.njoints),
        )
        if prefix is None:
            prefix = torch.zeros(state.model.njoints, state.model.nfeats, args.context_len)
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


def _resolve_target_frames(args: argparse.Namespace, fps: float, max_frames: int, motion_length: float) -> int:
    target_frames = int(motion_length * fps)
    if args.autoregressive:
        return target_frames
    return min(max_frames, target_frames)


def _normalize_request_texts(req: GenerateRequest):
    prompt = (req.text or "").strip()
    dynamic_texts = None
    if req.texts is not None:
        dynamic_texts = [text.strip() for text in req.texts if text and text.strip()]

    if prompt and dynamic_texts:
        raise HTTPException(status_code=422, detail="Provide either text or texts, not both.")
    if dynamic_texts is not None:
        if not dynamic_texts:
            raise HTTPException(status_code=422, detail="texts must contain at least one non-empty sentence.")
        if any(len(text) > 200 for text in dynamic_texts):
            raise HTTPException(status_code=422, detail="each item in texts must be at most 200 characters.")
        return [dynamic_texts[0]], dynamic_texts
    if not prompt:
        raise HTTPException(status_code=422, detail="text or texts is required.")
    if len(prompt) > 200:
        raise HTTPException(status_code=422, detail="text must be at most 200 characters.")
    return [prompt], None


def _prepare_text_condition(model, texts, dynamic_texts, num_samples):
    if dynamic_texts is None:
        return texts, model.encode_text(texts)

    text_embed = model.encode_text(dynamic_texts)
    return [list(dynamic_texts) for _ in range(num_samples)], repeat_text_embed_for_dynamic(text_embed, num_samples)


def _init_state() -> ServerState:
    model_path = os.environ.get("MDM_MODEL_PATH", "")
    if not model_path:
        raise RuntimeError("MDM_MODEL_PATH is required.")
    device = int(os.environ.get("MDM_DEVICE", "0"))
    use_ema_env = os.environ.get("MDM_USE_EMA")
    use_ema = None
    if use_ema_env is not None:
        use_ema = use_ema_env.lower() in ("1", "true", "yes")

    args = _build_args(model_path, device, use_ema)
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    model_args = _read_model_args(model_path)

    guidance_env = os.environ.get("MDM_GUIDANCE_PARAM")
    if guidance_env is not None:
        args.guidance_param = float(guidance_env)
    elif "gen_guidance_param" in model_args:
        args.guidance_param = float(model_args["gen_guidance_param"])
    else:
        args.guidance_param = default_guidance_for_arch(args.arch)

    if "autoregressive" in model_args:
        args.autoregressive = _coerce_bool(model_args["autoregressive"])

    autoregressive_env = os.environ.get("MDM_AUTOREGRESSIVE")
    if autoregressive_env is not None:
        args.autoregressive = autoregressive_env.lower() in ("1", "true", "yes")
    ar_include_env = os.environ.get("MDM_AUTOREGRESSIVE_INCLUDE_PREFIX")
    if ar_include_env is not None:
        args.autoregressive_include_prefix = ar_include_env.lower() in ("1", "true", "yes")

    if args.dataset != "humanml":
        raise RuntimeError(f"Only HumanML is supported, got dataset={args.dataset!r}.")
    if args.arch not in {"dit", "ar_dit"}:
        raise RuntimeError(f"Only dit/ar_dit are supported here, got arch={args.arch!r}.")
    if args.autoregressive and args.arch != "ar_dit":
        raise RuntimeError("Autoregressive serving requires an ar_dit checkpoint.")
    if args.autoregressive_motion_blend_frames < 0:
        raise RuntimeError("autoregressive_motion_blend_frames must be >= 0.")
    if args.autoregressive and (args.context_len <= 0 or args.pred_len <= 0):
        raise RuntimeError("Autoregressive sampling requires context_len > 0 and pred_len > 0.")

    continuation_prefix_frames = int(os.environ.get("MDM_CONTINUATION_PREFIX_FRAMES", "20"))
    if continuation_prefix_frames <= 0:
        raise RuntimeError("MDM_CONTINUATION_PREFIX_FRAMES must be greater than zero.")
    continuation_blend_frames = int(os.environ.get("MDM_CONTINUATION_BLEND_FRAMES", "12"))
    if continuation_blend_frames < 0:
        raise RuntimeError("MDM_CONTINUATION_BLEND_FRAMES must be >= 0.")

    max_frames = 196
    fps = 20

    model, diffusion = create_model_and_diffusion(args, _DummyData())
    load_saved_model(model, args.model_path, use_avg=args.use_ema)
    model.to(dist_util.dev())
    model.eval()

    guided_model = None
    if model.cond_mask_prob > 0:
        guided_model = ClassifierFreeSampleModel(model)
        guided_model.to(dist_util.dev())
        guided_model.eval()

    data_root = _resolve_data_root(args)
    mean = np.load(os.path.join(data_root, "Mean.npy"))
    std = np.load(os.path.join(data_root, "Std.npy"))
    return ServerState(
        args=args,
        model=model,
        guided_model=guided_model,
        diffusion=diffusion,
        mean=mean,
        std=std,
        max_frames=max_frames,
        fps=fps,
        lock=threading.Lock(),
        continuation_prefix_frames=continuation_prefix_frames,
        continuation_blend_frames=continuation_blend_frames,
        session_motions={},
        session_decode_states={},
    )


def _sample(state: ServerState, req: GenerateRequest) -> bytes:
    args = copy.deepcopy(state.args)
    texts, dynamic_texts = _normalize_request_texts(req)

    args.num_samples = 1
    args.num_repetitions = 1

    if dynamic_texts is not None:
        if not args.autoregressive:
            raise HTTPException(status_code=400, detail="texts requires an autoregressive checkpoint.")
        n_frames = len(dynamic_texts) * args.pred_len
    elif req.num_frames is not None:
        n_frames = req.num_frames
    else:
        motion_length = req.motion_length if req.motion_length is not None else args.motion_length
        n_frames = _resolve_target_frames(args, state.fps, state.max_frames, motion_length)
    if n_frames <= 0:
        raise HTTPException(status_code=400, detail="motion_length too small for current FPS.")

    lookahead_frames = 0 if req.is_final else (req.lookahead_frames if req.lookahead_frames is not None else int(os.environ.get("MDM_CONTINUATION_LOOKAHEAD_FRAMES", "0")))
    if lookahead_frames < 0 or (lookahead_frames and args.arch != "dit"):
        raise HTTPException(status_code=400, detail="lookahead requires plain dit and nonnegative frames")
    if req.init_prefix is not None and args.arch != "dit":
        raise HTTPException(status_code=400, detail="init_prefix requires plain dit")
    continuation_prefix = None
    continuation_decode_state = None
    prefix_frames = 0
    blend_frames = 0
    if args.arch == "dit":
        if req.reset_session:
            state.session_motions.pop(req.session_id, None)
            state.session_decode_states.pop(req.session_id, None)
        if req.init_prefix is not None and req.session_id not in state.session_motions:
            if not req.continue_from_previous:
                raise HTTPException(status_code=400, detail="init_prefix requires continue_from_previous")
            from infer import _load_continuation_prefix
            init_path = (os.path.join(_REPO_ROOT, "assets", "ar_dit_zero_vel_stand_prefix.npy") if req.init_prefix == "stand_mean" else os.environ.get("MDM_STAND_DATA_PREFIX_NPY", os.path.join(_REPO_ROOT, "assets", "dit_stand_data_prefix.npy")))
            try:
                init_motion, init_state = _load_continuation_prefix(init_path, state.mean, state.std, 20, int(state.mean.shape[0]))
            except (SystemExit, ValueError, OSError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            state.session_motions[req.session_id] = init_motion.expand(args.num_samples, -1, -1, -1).clone().detach().cpu()
            state.session_decode_states[req.session_id] = {key: value.expand(args.num_samples, *value.shape[1:]).clone().detach().cpu() for key, value in init_state.items()}
        if req.continue_from_previous:
            previous_motion = state.session_motions.get(req.session_id)
            if previous_motion is not None:
                requested_prefix_frames = req.prefix_frames or state.continuation_prefix_frames
                prefix_frames = min(requested_prefix_frames, previous_motion.shape[-1])
                continuation_prefix = previous_motion[..., -prefix_frames:]
                continuation_decode_state = state.session_decode_states.get(req.session_id)
                requested_blend_frames = (
                    state.continuation_blend_frames
                    if req.blend_frames is None
                    else req.blend_frames
                )
                blend_frames = min(requested_blend_frames, n_frames)

        if n_frames + prefix_frames + lookahead_frames > state.max_frames:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Requested {n_frames} new frames plus {prefix_frames} prefix frames, "
                    f"but this DiT supports at most {state.max_frames} total frames."
                ),
            )

    args.batch_size = args.num_samples
    total_frames = n_frames + prefix_frames + lookahead_frames

    def base_sample_fn(model, shape, **sample_kwargs):
        return state.diffusion.logsnr_sample_loop(
            model,
            shape,
            steps=int(getattr(args, "sampling_steps", 50)),
            **sample_kwargs,
        )

    sample_fn = base_sample_fn
    if args.autoregressive:
        sample_fn = AutoRegressiveSampler(args, base_sample_fn, n_frames).sample

    if args.guidance_param != 1:
        if state.guided_model is None:
            raise HTTPException(status_code=400, detail="Guidance is not available for this checkpoint.")
        model_to_use = state.guided_model
    else:
        model_to_use = state.model

    model_kwargs = _build_model_kwargs(state, args, texts, total_frames)
    if args.guidance_param != 1:
        model_kwargs["y"]["scale"] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
    model_kwargs["y"]["text"], model_kwargs["y"]["text_embed"] = _prepare_text_condition(
        state.model,
        model_kwargs["y"]["text"],
        dynamic_texts,
        args.num_samples,
    )

    sample_frames = total_frames if args.autoregressive or args.context_len == 0 else args.pred_len
    motion_shape = (args.batch_size, state.model.njoints, state.model.nfeats, sample_frames)
    inpaint_kwargs = {}
    if continuation_prefix is not None:
        known_motion, known_mask = _build_known_prefix(
            continuation_prefix,
            sample_frames,
            dist_util.dev(),
        )
        inpaint_kwargs = {"known_motion": known_motion, "known_mask": known_mask}

    with torch.inference_mode():
        fixseed(args.seed)
        sample = sample_fn(
            model_to_use,
            motion_shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            **inpaint_kwargs,
        )
        if args.arch == "dit":
            sample = sample[..., :prefix_frames + n_frames]
        if prefix_frames:
            sample = smooth_continuation_boundary(
                continuation_prefix,
                sample[..., prefix_frames:],
                blend_frames,
                blend_mode=req.blend_mode,
            )
        if args.arch == "dit":
            state.session_motions[req.session_id] = sample.detach().cpu()
        motions, hml_vec, next_decode_state = _decode_motion_batch_stateful(
            sample,
            state.mean,
            state.std,
            decode_state=continuation_decode_state,
        )
        if args.arch == "dit":
            state.session_decode_states[req.session_id] = {
                key: value.detach().cpu()
                for key, value in next_decode_state.items()
            }
        if args.autoregressive:
            lengths = np.full((args.batch_size,), n_frames, dtype=np.int64)
        elif args.context_len > 0:
            lengths = np.full((args.batch_size,), motions.shape[-1], dtype=np.int64)
        else:
            lengths = np.full((args.batch_size,), n_frames, dtype=np.int64)

    results = {
        "motion": motions[: args.num_samples],
        "hml_vec": hml_vec[: args.num_samples],
        "text": model_kwargs["y"]["text"][: args.num_samples],
        "lengths": lengths[: args.num_samples],
        "fps": state.fps,
        "num_samples": args.num_samples,
        "num_repetitions": args.num_repetitions,
        "session_id": req.session_id,
        "continuation_blend_mode": req.blend_mode,
        "continuation_lookahead_frames": lookahead_frames,
        "continuation_prefix_frames": prefix_frames,
        "continuation_blend_frames": blend_frames,
        "decode_state": {
            key: value.cpu().numpy()
            for key, value in next_decode_state.items()
        },
    }

    buf = io.BytesIO()
    np.save(buf, results)
    return buf.getvalue()


def _denormalize_hml_vec(sample: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    sample = sample.cpu().permute(0, 2, 3, 1)
    mean_t = torch.as_tensor(mean, dtype=sample.dtype)
    std_t = torch.as_tensor(std, dtype=sample.dtype)
    return (sample * std_t) + mean_t


def _decode_motion_batch_stateful(
    sample: torch.Tensor,
    mean: np.ndarray,
    std: np.ndarray,
    decode_state=None,
):
    hml_vec = _denormalize_hml_vec(sample, mean, std)
    motion, next_decode_state = recover_from_ric_stateful(
        hml_vec.float(),
        22,
        state=decode_state,
    )
    motion = motion.view(-1, *motion.shape[2:]).permute(0, 2, 3, 1)
    hml_vec = hml_vec.squeeze(1).permute(0, 2, 1)
    return motion.cpu().numpy(), hml_vec.cpu().numpy(), next_decode_state


def _decode_hml_vec_chunk_stateful(hml_vec_chunk: torch.Tensor, state=None):
    init_rot_ang = None if state is None else state["root_rot_ang"].to(hml_vec_chunk.device)
    rot_vel = hml_vec_chunk[..., 0]
    root_rot_ang = torch.zeros_like(rot_vel)
    if init_rot_ang is None:
        root_rot_ang[..., 0] = 0
    else:
        root_rot_ang[..., 0] = init_rot_ang
    if rot_vel.shape[-1] > 1:
        root_rot_ang[..., 1:] = rot_vel[..., :-1]
    root_rot_ang = torch.cumsum(root_rot_ang, dim=-1)

    motion, next_state = recover_from_ric_stateful(hml_vec_chunk.float(), 22, state=state)
    motion = motion.view(-1, *motion.shape[2:]).permute(0, 2, 3, 1)
    robot = {
        "root_pos": motion[:, 0].permute(0, 2, 1).cpu().numpy(),
        "root_rot_ang": root_rot_ang.squeeze(1).cpu().numpy(),
        "next_root_rot_ang": next_state["root_rot_ang"].detach().cpu().numpy(),
    }
    return motion.cpu().numpy(), next_state, robot


def _stream_autoregressive(state: ServerState, req: GenerateRequest):
    args = copy.deepcopy(state.args)
    texts, dynamic_texts = _normalize_request_texts(req)
    if not args.autoregressive:
        raise HTTPException(status_code=400, detail="Streaming is only available for autoregressive checkpoints.")

    args.num_samples = 1
    args.num_repetitions = 1

    if dynamic_texts is not None:
        n_frames = len(dynamic_texts) * args.pred_len
    else:
        motion_length = req.motion_length if req.motion_length is not None else args.motion_length
        n_frames = _resolve_target_frames(args, state.fps, state.max_frames, motion_length)
    if n_frames <= 0:
        raise HTTPException(status_code=400, detail="motion_length too small for current FPS.")

    args.batch_size = args.num_samples

    def base_sample_fn(model, shape, **sample_kwargs):
        return state.diffusion.logsnr_sample_loop(
            model,
            shape,
            steps=int(getattr(args, "sampling_steps", 50)),
            **sample_kwargs,
        )

    if args.guidance_param != 1:
        if state.guided_model is None:
            raise HTTPException(status_code=400, detail="Guidance is not available for this checkpoint.")
        model_to_use = state.guided_model
    else:
        model_to_use = state.model

    model_kwargs = _build_model_kwargs(state, args, texts, n_frames)
    if args.guidance_param != 1:
        model_kwargs["y"]["scale"] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param
    model_kwargs["y"]["text"], model_kwargs["y"]["text_embed"] = _prepare_text_condition(
        state.model,
        model_kwargs["y"]["text"],
        dynamic_texts,
        args.num_samples,
    )

    sampler = AutoRegressiveSampler(args, base_sample_fn, n_frames)
    motion_shape = (args.batch_size, state.model.njoints, state.model.nfeats, n_frames)

    def event_iter():
        with state.lock:
            with torch.inference_mode():
                fixseed(args.seed)
                total_segments = (n_frames // args.pred_len) + int(n_frames % args.pred_len > 0)
                decode_state = None
                for segment_index, (chunk, seg_text) in enumerate(
                    sampler.iter_chunks(
                        model_to_use,
                        motion_shape,
                        clip_denoised=False,
                        model_kwargs=model_kwargs,
                        progress=False,
                    )
                ):
                    hml_vec_chunk = _denormalize_hml_vec(chunk, state.mean, state.std)
                    decoded, decode_state, robot = _decode_hml_vec_chunk_stateful(
                        hml_vec_chunk,
                        state=decode_state,
                    )
                    segment_frames = min(args.pred_len, n_frames - (segment_index * args.pred_len))
                    current_frames = min((segment_index + 1) * args.pred_len, n_frames)
                    payload = {
                        "segment_index": segment_index,
                        "num_segments": total_segments,
                        "segment_frames": segment_frames,
                        "segment_seconds": segment_frames / state.fps,
                        "frames": current_frames,
                        "seconds": current_frames / state.fps,
                        "fps": state.fps,
                        "text": seg_text[0],
                        "motion": decoded[0][..., :segment_frames].tolist(),
                        "hml_vec": hml_vec_chunk[0, 0, :segment_frames].cpu().numpy().tolist(),
                        "robot": {
                            "root_pos": robot["root_pos"][0, :segment_frames].tolist(),
                            "root_rot_ang": robot["root_rot_ang"][0, :segment_frames].tolist(),
                            "next_root_rot_ang": float(robot["next_root_rot_ang"][0, 0]),
                        },
                        "done": current_frames >= n_frames,
                    }
                    yield json.dumps(payload, ensure_ascii=False) + "\n"

    return StreamingResponse(event_iter(), media_type="application/x-ndjson")


app = FastAPI()


@app.on_event("startup")
def _startup() -> None:
    app.state.mdm = _init_state()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: GenerateRequest):
    state = app.state.mdm
    with state.lock:
        payload = _sample(state, req)
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=results.npy"},
    )


@app.post("/generate_stream")
def generate_stream(req: GenerateRequest):
    state = app.state.mdm
    return _stream_autoregressive(state, req)
