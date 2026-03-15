#!/usr/bin/env python3
"""
Brief:
Consume the streaming generation endpoint, save each segment as .npy, and
optionally merge all segments into a final results.npy compatible payload.
"""
import argparse
import json
import os
import sys
from urllib import error, request

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from data_loaders.humanml.scripts.motion_process import recover_from_ric_stateful


def _build_args():
    parser = argparse.ArgumentParser(description="Test the streaming motion generation endpoint.")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/generate_stream",
        help="Streaming endpoint URL.",
    )
    parser.add_argument("--text", default="", help="Single prompt.")
    parser.add_argument(
        "--texts_path",
        default="",
        help="Path to a text file with one prompt per line for dynamic-text streaming.",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/stream_client",
        help="Directory to save segment .npy files and the merged result.",
    )
    parser.add_argument(
        "--motion_length",
        default=0.0,
        type=float,
        help="Optional motion length for single-prompt streaming.",
    )
    return parser.parse_args()


def _read_text_lines(path: str):
    if not os.path.exists(path):
        raise SystemExit(f"text file not found: {path}")
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    return [line for line in lines if line]


def _build_payload(args):
    if args.text and args.texts_path:
        raise SystemExit("Provide either --text or --texts_path, not both.")
    if args.texts_path:
        texts = _read_text_lines(args.texts_path)
        if not texts:
            raise SystemExit("texts_path is empty.")
        return {"texts": texts}, texts
    if not args.text:
        raise SystemExit("Provide --text or --texts_path.")

    payload = {"text": args.text}
    if args.motion_length > 0:
        payload["motion_length"] = args.motion_length
    return payload, [args.text]


def _decode_hml_vec_chunk_stateful(hml_vec_chunk, state=None):
    chunk = torch.as_tensor(hml_vec_chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    init_rot_ang = None if state is None else state["root_rot_ang"]
    rot_vel = chunk[..., 0]
    root_rot_ang = torch.zeros_like(rot_vel)
    if init_rot_ang is None:
        root_rot_ang[..., 0] = 0
    else:
        root_rot_ang[..., 0] = init_rot_ang
    if rot_vel.shape[-1] > 1:
        root_rot_ang[..., 1:] = rot_vel[..., :-1]
    root_rot_ang = torch.cumsum(root_rot_ang, dim=-1)

    motion, next_state = recover_from_ric_stateful(chunk, 22, state=state)
    motion = motion.view(-1, *motion.shape[2:]).permute(0, 2, 3, 1)[0].cpu().numpy()
    robot = {
        "root_pos": motion[0].T.copy(),
        "root_rot_ang": root_rot_ang.squeeze(0).squeeze(0).cpu().numpy(),
    }
    return motion, next_state, robot


def _save_segment(out_dir, payload, motion_xyz, robot):
    segment_index = int(payload["segment_index"])
    fps = int(payload["fps"])
    segment_frames = int(payload["segment_frames"])
    motion = motion_xyz[..., :segment_frames]
    hml_vec = np.asarray(payload["hml_vec"], dtype=np.float32)[:segment_frames]

    segment_result = {
        "motion": motion[np.newaxis, ...],
        "hml_vec": hml_vec[np.newaxis, ...],
        "text": [payload["text"]],
        "lengths": np.asarray([segment_frames], dtype=np.int64),
        "fps": fps,
        "robot": {
            "root_pos": robot["root_pos"][:segment_frames][np.newaxis, ...],
            "root_rot_ang": robot["root_rot_ang"][:segment_frames][np.newaxis, ...],
        },
        "segment_index": segment_index,
        "num_segments": int(payload["num_segments"]),
        "segment_seconds": float(payload["segment_seconds"]),
        "done": bool(payload["done"]),
    }
    segment_path = os.path.join(out_dir, f"segment_{segment_index:03d}.npy")
    np.save(segment_path, segment_result)
    return motion, hml_vec, robot, segment_path


def _save_merged(out_dir, texts, fps, segment_motions, segment_hml_vecs, robot_roots, robot_angles):
    full_motion = np.concatenate(segment_motions, axis=-1)
    full_hml_vec = np.concatenate(segment_hml_vecs, axis=0)
    full_root_pos = np.concatenate(robot_roots, axis=0)
    full_root_rot_ang = np.concatenate(robot_angles, axis=0)
    result = {
        "motion": full_motion[np.newaxis, ...],
        "hml_vec": full_hml_vec[np.newaxis, ...],
        "text": [texts if len(texts) > 1 else texts[0]],
        "lengths": np.asarray([full_motion.shape[-1]], dtype=np.int64),
        "fps": fps,
        "robot": {
            "joints_xyz": full_motion[np.newaxis, ...],
            "root_pos": full_root_pos[np.newaxis, ...],
            "root_rot_ang": full_root_rot_ang[np.newaxis, ...],
        },
        "num_samples": 1,
        "num_repetitions": 1,
    }
    result_path = os.path.join(out_dir, "results.npy")
    np.save(result_path, result)
    return result_path


def main():
    args = _build_args()
    payload, texts = _build_payload(args)
    os.makedirs(args.out_dir, exist_ok=True)

    req = request.Request(
        args.url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    segment_motions = []
    segment_hml_vecs = []
    robot_roots = []
    robot_angles = []
    fps = None
    decode_state = None
    try:
        with request.urlopen(req) as resp:
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                payload = json.loads(line)
                motion, decode_state, robot = _decode_hml_vec_chunk_stateful(
                    payload["hml_vec"],
                    state=decode_state,
                )
                motion, hml_vec, robot, segment_path = _save_segment(
                    args.out_dir,
                    payload,
                    motion,
                    robot,
                )
                segment_motions.append(motion)
                segment_hml_vecs.append(hml_vec)
                robot_roots.append(robot["root_pos"])
                robot_angles.append(robot["root_rot_ang"])
                fps = int(payload["fps"])
                print(
                    f"saved segment {payload['segment_index'] + 1}/{payload['num_segments']} "
                    f"to {segment_path}"
                )
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"request failed: HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise SystemExit(f"request failed: {exc}") from exc

    if not segment_motions:
        raise SystemExit("no segments were received from the streaming endpoint.")

    result_path = _save_merged(
        args.out_dir,
        texts,
        fps,
        segment_motions,
        segment_hml_vecs,
        robot_roots,
        robot_angles,
    )
    print(f"saved merged result to {result_path}")


if __name__ == "__main__":
    main()
