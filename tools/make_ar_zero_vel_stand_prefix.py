import argparse
import json
import os
from pathlib import Path

import numpy as np


def score_pose_window(w):
    root_vel = w[:, :3]
    ric = w[:, 4:67]
    joint_vel = w[:, 193:259] if w.shape[1] >= 259 else np.zeros((len(w), 1), dtype=w.dtype)
    dric = np.diff(ric, axis=0) if len(w) > 1 else np.zeros_like(ric)

    score = 10.0 * np.mean(root_vel ** 2)
    score += 2.0 * np.mean(joint_vel ** 2)
    score += 2.0 * np.mean(dric ** 2)
    return float(score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--data_dir", default="dataset/HumanML3D")
    parser.add_argument("--out", default="assets/ar_dit_zero_vel_stand_prefix.npy")
    parser.add_argument("--scan_limit", type=int, default=5000)
    args = parser.parse_args()

    with open(os.path.join(args.ckpt_dir, "args.json"), "r") as f:
        model_args = json.load(f)

    context_len = int(model_args["context_len"])
    dim = 263

    data_dir = Path(args.data_dir)
    mean_path = data_dir / "Mean.npy"
    if not mean_path.exists():
        raise FileNotFoundError(mean_path)

    mean = np.load(mean_path).astype("float32")
    if mean.shape[0] != dim:
        raise ValueError(f"Mean.npy should have dim 263, got {mean.shape}")

    vec_dir = data_dir / "new_joint_vecs"

    best = None
    if vec_dir.exists():
        files = sorted(vec_dir.rglob("*.npy"))[: args.scan_limit]
        print(f"[prefix] scanning {len(files)} files from {vec_dir}")

        for fp in files:
            try:
                x = np.load(fp).astype("float32")
            except Exception:
                continue

            if x.ndim != 2 or x.shape[1] != dim or x.shape[0] < context_len:
                continue
            if not np.isfinite(x).all():
                continue

            for st in range(0, x.shape[0] - context_len + 1, 2):
                w = x[st: st + context_len]
                s = score_pose_window(w)
                if best is None or s < best["score"]:
                    best = {
                        "score": s,
                        "path": str(fp),
                        "start": st,
                        "window": w.copy(),
                    }

    if best is not None:
        # 取最静止窗口中间一帧的姿态，然后复制成 20 帧静止 prefix
        pose = best["window"][context_len // 2].copy()
        mode = "dataset_pose_zero_velocity"
        source = best["path"]
        start = best["start"]
        score = best["score"]
    else:
        # 回退到均值姿态
        pose = mean.copy()
        mode = "mean_pose_zero_velocity"
        source = "Mean.npy"
        start = -1
        score = -1.0

    prefix = np.tile(pose[None, :], (context_len, 1)).astype("float32")

    # HumanML3D 263-dim 常用布局：
    # 0: root angular velocity
    # 1:3 root xz linear velocity
    # 3: root height
    # 4:67 RIC joint positions
    # 67:193 local rotations
    # 193:259 local joint velocities
    # 259:263 foot contact
    prefix[:, 0:3] = 0.0
    prefix[:, 193:259] = 0.0
    prefix[:, 259:263] = 1.0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, prefix)

    info = (
        f"mode: {mode}\n"
        f"source: {source}\n"
        f"start: {start}\n"
        f"score: {score}\n"
        f"context_len: {context_len}\n"
        f"shape: {prefix.shape}\n"
        f"root_vel_mean_abs: {np.mean(np.abs(prefix[:, 0:3]))}\n"
        f"joint_vel_mean_abs: {np.mean(np.abs(prefix[:, 193:259]))}\n"
        f"foot_contact_mean: {np.mean(prefix[:, 259:263])}\n"
    )

    with open(args.out + ".info.txt", "w") as f:
        f.write(info)

    print(info)
    print(f"[prefix] saved to {args.out}")


if __name__ == "__main__":
    main()
