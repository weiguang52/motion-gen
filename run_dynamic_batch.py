#!/usr/bin/env python3
"""
批量调用项目 infer.py 的 --dynamic_text_path 模式，只保留最终 MP4。

设计目标：
- 不改项目代码；
- 不修改原模型目录及原 args.json；
- 按《动作批测结果.md》的记录顺序逐条生成；
- 单段记录写 1 行 prompt，多段记录按原顺序写多行；
- 每条记录调用一次 infer.py 的 --dynamic_text_path；
- 默认将每一段动作控制为 6 秒；
- 项目 infer.py 固定使用 20 FPS，因此 6 秒对应 pred_len=120 帧；
- 最终只保留 MP4 文件；
- MP4 文件名与非流式批量生成保持一致：
    sample00_rep00.mp4
    sample01_rep00.mp4
    ...
- 即使使用 --start、--end 或 --only-multi，文件编号仍按原始 516 条记录排序：
    记录 1   -> sample00_rep00.mp4
    记录 251 -> sample250_rep00.mp4
    记录 516 -> sample515_rep00.mp4
- 如果某条记录生成多次重复，例如额外传入 --num_repetitions 3，则得到：
    sample00_rep00.mp4
    sample00_rep01.mp4
    sample00_rep02.mp4

工作原理：
- dynamic_text_path 模式的每段时长不是由 --motion_length 控制；
- infer.py 使用 checkpoint 同目录 args.json 中的 pred_len；
- 本脚本会创建一个临时模型目录：
  1. 链接原 checkpoint；
  2. 复制原 args.json；
  3. 只把临时 args.json 的 pred_len 改为 segment_seconds * 20；
- 批量生成结束后，临时模型目录会自动删除。

示例：
    python run_dynamic_batch_6s_mp4_only.py \
      --repo-dir /home/cxm2/data/motion-gen \
      --source-md /home/cxm2/data/motion-gen/动作批测结果.md \
      --model-path save/ar_ardit_cfg40_cosine600k_ctx_strongreg_genprefix_mid_steps6/model000550000.pt \
      --output-dir outputs/dynamic_batch_mp4

显式指定每段 6 秒：
    python run_dynamic_batch_6s_mp4_only.py \
      --repo-dir /home/cxm2/data/motion-gen \
      --source-md /home/cxm2/data/motion-gen/动作批测结果.md \
      --model-path save/ar_ardit_cfg40_cosine600k_ctx_strongreg_genprefix_mid_steps6/model000550000.pt \
      --output-dir outputs/dynamic_batch_mp4 \
      --segment-seconds 6

只跑多段记录：
    python run_dynamic_batch_6s_mp4_only.py \
      --repo-dir /home/cxm2/data/motion-gen \
      --source-md /home/cxm2/data/motion-gen/动作批测结果.md \
      --model-path save/ar_ardit_cfg40_cosine600k_ctx_strongreg_genprefix_mid_steps6/model000550000.pt \
      --output-dir outputs/dynamic_multi_mp4 \
      --only-multi

断点续跑：
    python run_dynamic_batch_6s_mp4_only.py \
      --repo-dir /home/cxm2/data/motion-gen \
      --source-md /home/cxm2/data/motion-gen/动作批测结果.md \
      --model-path save/ar_ardit_cfg40_cosine600k_ctx_strongreg_genprefix_mid_steps6/model000550000.pt \
      --output-dir outputs/dynamic_batch_mp4 \
      --skip-existing

传递额外 infer.py 参数：
    python run_dynamic_batch_6s_mp4_only.py \
      --repo-dir /home/cxm2/data/motion-gen \
      --source-md /home/cxm2/data/motion-gen/动作批测结果.md \
      --model-path save/ar_ardit_cfg40_cosine600k_ctx_strongreg_genprefix_mid_steps6/model000550000.pt \
      --output-dir outputs/dynamic_batch_mp4 \
      --segment-seconds 6 \
      -- --num_repetitions 3 --guidance_param 4.0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple


# infer.py 中固定使用 fps = 20。
PROJECT_FPS = 20


def parse_markdown(md_path: Path) -> List[Dict[str, Any]]:
    """提取每条记录的编号、原始中文文本和 motion-gen prompts。"""
    lines = md_path.read_text(encoding="utf-8").splitlines()

    records: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None
    in_prompts = False

    for line in lines:
        record_match = re.match(r"^###\s+(\d+)\.\s+(.*)$", line)
        if record_match:
            if current is not None:
                records.append(current)

            current = {
                "index": int(record_match.group(1)),
                "source_text": record_match.group(2).strip(),
                "prompts": [],
            }
            in_prompts = False
            continue

        if current is None:
            continue

        if re.match(r"^-\s+\*\*motion-gen prompts\*\*", line):
            in_prompts = True
            continue

        if in_prompts:
            prompt_match = re.match(r"^\s*\d+\.\s+(.*\S)\s*$", line)
            if prompt_match:
                current["prompts"].append(prompt_match.group(1))
                continue

            if re.match(r"^-\s+\*\*error\*\*", line):
                in_prompts = False

    if current is not None:
        records.append(current)

    return records


def seconds_to_frames(segment_seconds: float) -> int:
    """按照项目固定 20 FPS，把每段秒数转换成 pred_len 帧数。"""
    if segment_seconds <= 0:
        raise ValueError("segment_seconds 必须大于 0。")

    exact_frames = segment_seconds * PROJECT_FPS
    rounded_frames = int(round(exact_frames))

    if rounded_frames <= 0:
        raise ValueError("segment_seconds 太小，换算后的帧数小于 1。")

    # 避免用户输入无法在 20 FPS 下精确表达的时长。
    if abs(exact_frames - rounded_frames) > 1e-6:
        raise ValueError(
            f"segment_seconds={segment_seconds} 无法在 {PROJECT_FPS} FPS 下精确表示；"
            f"请输入 0.05 秒的整数倍。"
        )

    return rounded_frames


def link_or_copy_file(source: Path, destination: Path) -> str:
    """
    优先创建硬链接，其次创建符号链接，最后才复制 checkpoint。

    返回实际使用的方式：hardlink / symlink / copy。
    """
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        pass

    try:
        destination.symlink_to(source)
        return "symlink"
    except OSError:
        pass

    shutil.copy2(source, destination)
    return "copy"


def prepare_temporary_model(
    original_model_path: Path,
    temporary_root: Path,
    target_pred_len: int,
) -> Tuple[Path, int | None, str]:
    """
    创建临时模型目录并覆盖临时 args.json 中的 pred_len。

    不会修改原 checkpoint，也不会修改原 args.json。

    返回：
      临时 checkpoint 路径、原 pred_len、checkpoint 放置方式。
    """
    original_args_path = original_model_path.parent / "args.json"
    if not original_args_path.is_file():
        raise FileNotFoundError(f"未找到模型配置：{original_args_path}")

    try:
        model_args = json.loads(original_args_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"模型 args.json 不是有效 JSON：{original_args_path}: {exc}") from exc

    original_pred_len_raw = model_args.get("pred_len")
    original_pred_len: int | None
    try:
        original_pred_len = (
            int(original_pred_len_raw)
            if original_pred_len_raw is not None
            else None
        )
    except (TypeError, ValueError):
        original_pred_len = None

    model_args["pred_len"] = target_pred_len

    # 保留或显式启用 AR 标记，避免配置中布尔值缺失。
    model_args["autoregressive"] = True

    temporary_model_dir = temporary_root / "model"
    temporary_model_dir.mkdir(parents=True, exist_ok=True)

    temporary_model_path = temporary_model_dir / original_model_path.name
    link_mode = link_or_copy_file(original_model_path, temporary_model_path)

    temporary_args_path = temporary_model_dir / "args.json"
    temporary_args_path.write_text(
        json.dumps(model_args, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return temporary_model_path, original_pred_len, link_mode


def build_command(
    python_exe: str,
    infer_path: Path,
    model_path: Path,
    prompt_file: Path,
    npy_path: Path,
    mp4_dir: Path,
    extra_args: List[str],
) -> List[str]:
    return [
        python_exe,
        str(infer_path),
        "--model_path",
        str(model_path),
        "--dynamic_text_path",
        str(prompt_file),
        "--out",
        str(npy_path),
        "--mp4_dir",
        str(mp4_dir),
        *extra_args,
    ]


def move_generated_mp4s(
    temp_mp4_dir: Path,
    final_output_dir: Path,
    sample_index: int,
) -> List[Path]:
    """
    将单次 infer.py 生成的 MP4 移动到最终目录，并按非流式规则改名：
      sample{sample_index:02d}_rep{rep_index:02d}.mp4
    """
    generated = sorted(temp_mp4_dir.glob("sample*_rep*.mp4"))
    moved: List[Path] = []

    for src in generated:
        match = re.match(r"^sample\d+_rep(\d+)\.mp4$", src.name)
        if not match:
            continue

        rep_index = int(match.group(1))
        dst = final_output_dir / f"sample{sample_index:02d}_rep{rep_index:02d}.mp4"

        if dst.exists():
            dst.unlink()

        shutil.move(str(src), str(dst))
        moved.append(dst)

    return moved


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "逐条调用 infer.py，批量执行 dynamic_text_path 生成，"
            "默认每段 6 秒并且只保留 MP4。"
        )
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        required=True,
        help="motion-gen 项目根目录，目录中应包含 infer.py。",
    )
    parser.add_argument(
        "--source-md",
        type=Path,
        required=True,
        help="动作批测结果.md 的路径。",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="AR 模型路径。相对路径按 repo-dir 解析。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dynamic_batch_mp4"),
        help="最终 MP4 输出目录。相对路径按 repo-dir 解析。",
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=6.0,
        help=(
            "每个 dynamic prompt 段落的时长，默认 6 秒。"
            "项目固定为 20 FPS，因此 6 秒对应 pred_len=120。"
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="用于运行 infer.py 的 Python 解释器，默认使用当前解释器。",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="起始记录编号，包含。",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=516,
        help="结束记录编号，包含。",
    )
    parser.add_argument(
        "--only-multi",
        action="store_true",
        help="只运行包含两个或更多 prompts 的记录。",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果目标 sampleXX_rep00.mp4 已存在，则跳过该记录。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印命令和时长配置，不实际运行 infer.py。",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="任意记录失败后立即停止；默认记录错误后继续。",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help=(
            "传给 infer.py 的额外参数。使用 -- 分隔，例如："
            "-- --num_repetitions 2 --guidance_param 4.0"
        ),
    )
    args = parser.parse_args()

    try:
        target_pred_len = seconds_to_frames(args.segment_seconds)
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    repo_dir = args.repo_dir.resolve()
    source_md = args.source_md.resolve()

    infer_path = repo_dir / "infer.py"
    if not infer_path.is_file():
        print(f"错误：未找到 infer.py：{infer_path}", file=sys.stderr)
        return 2

    if not source_md.is_file():
        print(f"错误：未找到源文件：{source_md}", file=sys.stderr)
        return 2

    model_path = args.model_path
    if not model_path.is_absolute():
        model_path = repo_dir / model_path
    model_path = model_path.resolve()

    if not model_path.is_file():
        print(f"错误：未找到模型文件：{model_path}", file=sys.stderr)
        return 2

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_dir / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = parse_markdown(source_md)
    selected = [
        record
        for record in records
        if args.start <= record["index"] <= args.end
        and record["prompts"]
    ]

    if args.only_multi:
        selected = [record for record in selected if len(record["prompts"]) > 1]

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    print(f"共解析 {len(records)} 条记录，本次选中 {len(selected)} 条。")
    print(f"每段动作时长：{args.segment_seconds:g} 秒")
    print(f"项目 FPS：{PROJECT_FPS}")
    print(f"临时 pred_len：{target_pred_len} 帧")
    print(f"最终 MP4 输出目录：{output_dir}")

    success_count = 0
    failed_count = 0
    skipped_count = 0

    # 临时模型目录只创建一次，供所有子进程复用。
    with tempfile.TemporaryDirectory(prefix="dynamic_model_config_") as model_temp_str:
        model_temp_root = Path(model_temp_str)

        try:
            effective_model_path, original_pred_len, link_mode = prepare_temporary_model(
                original_model_path=model_path,
                temporary_root=model_temp_root,
                target_pred_len=target_pred_len,
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            print(f"错误：准备临时模型配置失败：{exc}", file=sys.stderr)
            return 2

        print(
            "模型 pred_len："
            f"原值={original_pred_len if original_pred_len is not None else '未知'}，"
            f"本次临时值={target_pred_len}"
        )
        print(f"临时 checkpoint 方式：{link_mode}")

        for position, record in enumerate(selected, start=1):
            # 固定按照原始 516 条记录编号映射到非流式文件名。
            # 记录 1 -> sample00，记录 516 -> sample515。
            sample_index = record["index"] - 1
            target_rep0 = output_dir / f"sample{sample_index:02d}_rep00.mp4"

            if args.skip_existing and target_rep0.exists():
                print(
                    f"[SKIP {position}/{len(selected)}] "
                    f"record #{record['index']:03d} -> {target_rep0.name}"
                )
                skipped_count += 1
                continue

            with tempfile.TemporaryDirectory(prefix="dynamic_case_") as case_temp_str:
                case_temp_dir = Path(case_temp_str)
                prompt_file = case_temp_dir / "prompts.txt"
                npy_path = case_temp_dir / "temp.npy"
                temp_mp4_dir = case_temp_dir / "mp4"
                temp_mp4_dir.mkdir(parents=True, exist_ok=True)

                prompt_file.write_text(
                    "\n".join(record["prompts"]) + "\n",
                    encoding="utf-8",
                )

                command = build_command(
                    python_exe=args.python,
                    infer_path=infer_path,
                    model_path=effective_model_path,
                    prompt_file=prompt_file,
                    npy_path=npy_path,
                    mp4_dir=temp_mp4_dir,
                    extra_args=extra_args,
                )

                expected_seconds = len(record["prompts"]) * args.segment_seconds
                printable = subprocess.list2cmdline(command)

                print(
                    f"[RUN  {position}/{len(selected)}] "
                    f"record #{record['index']:03d} "
                    f"segments={len(record['prompts'])} "
                    f"expected={expected_seconds:g}s"
                )
                print(
                    f"  -> 输出名: sample{sample_index:02d}_rep00.mp4"
                )
                print(f"  {printable}")

                if args.dry_run:
                    continue

                process = subprocess.run(
                    command,
                    cwd=repo_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )

                if process.returncode != 0:
                    failed_count += 1
                    print(
                        f"[FAIL {position}/{len(selected)}] "
                        f"record #{record['index']:03d}",
                        file=sys.stderr,
                    )
                    print(process.stdout, file=sys.stderr)

                    if args.stop_on_error:
                        break
                    continue

                moved = move_generated_mp4s(
                    temp_mp4_dir=temp_mp4_dir,
                    final_output_dir=output_dir,
                    sample_index=sample_index,
                )

                if not moved:
                    failed_count += 1
                    print(
                        f"[FAIL {position}/{len(selected)}] "
                        f"record #{record['index']:03d} 未找到生成的 MP4",
                        file=sys.stderr,
                    )

                    if process.stdout:
                        print(process.stdout, file=sys.stderr)

                    if args.stop_on_error:
                        break
                    continue

                success_count += 1
                moved_names = ", ".join(path.name for path in moved)
                print(
                    f"[DONE {position}/{len(selected)}] "
                    f"record #{record['index']:03d} -> {moved_names}"
                )

    print(
        f"完成：成功 {success_count}，失败 {failed_count}，跳过 {skipped_count}。"
    )
    print(f"MP4 目录：{output_dir}")

    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
