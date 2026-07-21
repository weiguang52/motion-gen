import argparse
import json
import os
from pathlib import Path

import numpy as np


def score_window(w):
    """
    HumanML3D 263-dim rough layout:
    0: root angular velocity
    1:3 root xz velocity
    3: root y
    4:67 relative joint positions
    67:193 local rotations
    193:259 local joint velocities
    259:263 foot contacts
    """
    root_vel = w[:, :3]
    ric = w[:, 4:67]
    joint_vel = w[:, 193:259] if w.shape[1] >= 259 else np.zeros((len(w), 1), dtype=w.dtype)
    foot = w[:, 259:263] if w.shape[1] >= 263 else np.zeros((len(w), 1), dtype=w.dtype)

    dric = np.diff(ric, axis=0) if len(w) > 1 else np.zeros_like(ric)

    # 越小越静止；foot contact 多一点略微加分
    s = 5.0 * np.mean(root_vel ** 2)
    s += 1.0 * np.mean(joint_vel ** 2)
    s += 1.0 * np.mean(dric ** 2)
    s -= 0.02 * np.mean(foot)

    return float(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--data_dir", default="dataset/HumanML3D")
    parser.add_argument("--out", default="assets/ar_dit_default_stand_prefix.npy")
    parser.add_argument("--scan_limit", type=int, default=5000)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()

    args_json = os.path.join(args.ckpt_dir, "args.json")
    with open(args_json, "r") as f:
        model_args = json.load(f)

    context_len = int(model_args["context_len"])
    expected_dim = 263

    data_dir = Path(args.data_dir)
    mean_path = data_dir / "Mean.npy"

    if not mean_path.exists():
        raise FileNotFoundError(f"Missing {mean_path}")

    mean = np.load(mean_path)
    if mean.shape[0] != expected_dim:
        raise ValueError(f"Mean.npy dim should be {expected_dim}, got {mean.shape}")

    vec_dir = data_dir / "new_joint_vecs"
    best = None

    if vec_dir.exists():
        files = sorted(vec_dir.rglob("*.npy"))[: args.scan_limit]
        print(f"[prefix] scanning {len(files)} files from {vec_dir}")

        for fp in files:
            try:
                x = np.load(fp)
            except Exception:
                continue

            if x.ndim != 2 or x.shape[1] != expected_dim or x.shape[0] < context_len:
                continue
            if not np.isfinite(x).all():
                continue

            T = x.shape[0]
            for st in range(0, T - context_len + 1, args.stride):
                w = x[st: st + context_len]
                s = score_window(w)
                if best is None or s < best["score"]:
                    best = {
                        "score": s,
                        "path": str(fp),
                        "start": st,
                        "prefix": w.astype("float32"),
                    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if best is not None:
        prefix = best["prefix"]
        np.save(args.out, prefix)

        info = (
            f"mode: dataset_static_window\n"
            f"source: {best['path']}\n"
            f"start: {best['start']}\n"
            f"context_len: {context_len}\n"
            f"score: {best['score']}\n"
            f"shape: {prefix.shape}\n"
        )
        print(info)

        with open(args.out + ".info.txt", "w") as f:
            f.write(info)

    else:
        print("[prefix] WARNING: no valid HumanML3D motion window found.")
        print("[prefix] fallback to tiled Mean.npy. This only tests code flow; quality may be poor.")

        prefix = np.tile(mean[None, :], (context_len, 1)).astype("float32")
        np.save(args.out, prefix)

        info = (
            f"mode: mean_fallback\n"
            f"context_len: {context_len}\n"
            f"shape: {prefix.shape}\n"
        )
        print(info)

        with open(args.out + ".info.txt", "w") as f:
            f.write(info)

    print(f"[prefix] saved to {args.out}")


if __name__ == "__main__":
    main()
