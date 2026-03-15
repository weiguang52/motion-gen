#!/usr/bin/env python3
"""
Brief:
Benchmark the local generation server over repeated requests and report
average, P50, and P95 latencies.
"""
import argparse
import json
import os
import time
from urllib import error, request

import numpy as np


def _build_args():
    parser = argparse.ArgumentParser(description="Benchmark the motion generation server.")
    parser.add_argument(
        "--base_url",
        default="http://127.0.0.1:8000",
        help="Server base URL, for example http://127.0.0.1:8000.",
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["generate", "stream", "both"],
        help="Which endpoint(s) to benchmark.",
    )
    parser.add_argument("--runs", default=20, type=int, help="Number of repeated requests.")
    parser.add_argument("--text", default="", help="Single prompt payload.")
    parser.add_argument(
        "--texts_path",
        default="",
        help="Path to a text file with one prompt per line for dynamic-text benchmarking.",
    )
    parser.add_argument(
        "--motion_length",
        default=8.0,
        type=float,
        help="Motion length for single-prompt benchmarking.",
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
        return {"texts": texts}
    if args.text:
        return {"text": args.text, "motion_length": args.motion_length}
    return {
        "texts": [
            "a person stands still",
            "a person raises both arms",
            "a person waves both hands",
            "a person claps hands",
        ]
    }


def _request_json_lines(url, payload):
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req) as resp:
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                yield json.loads(line)
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"request failed: HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise SystemExit(f"request failed: {exc}") from exc


def _request_bytes(url, payload):
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req) as resp:
            return resp.read()
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"request failed: HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise SystemExit(f"request failed: {exc}") from exc


def _summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "avg": round(float(arr.mean()), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }


def _benchmark_generate(base_url, payload, runs):
    url = f"{base_url.rstrip('/')}/generate"
    total_times = []
    byte_sizes = []

    for run_idx in range(runs):
        start = time.perf_counter()
        data = _request_bytes(url, payload)
        elapsed = time.perf_counter() - start
        total_times.append(elapsed)
        byte_sizes.append(len(data))
        print(
            f"[generate] run {run_idx + 1:02d}/{runs}: "
            f"{elapsed:.3f}s, bytes={len(data)}"
        )

    return {
        "endpoint": "/generate",
        "runs": runs,
        "latency_s": _summarize(total_times),
        "bytes": {
            "avg": round(float(np.mean(byte_sizes)), 1),
            "min": int(np.min(byte_sizes)),
            "max": int(np.max(byte_sizes)),
        },
    }


def _benchmark_stream(base_url, payload, runs):
    url = f"{base_url.rstrip('/')}/generate_stream"
    first_chunk_times = []
    total_times = []
    segment_counts = []

    for run_idx in range(runs):
        start = time.perf_counter()
        first_chunk = None
        segment_count = 0
        for obj in _request_json_lines(url, payload):
            now = time.perf_counter()
            if first_chunk is None:
                first_chunk = now - start
            segment_count += 1
            if obj.get("done"):
                break
        total = time.perf_counter() - start
        first_chunk_times.append(first_chunk if first_chunk is not None else total)
        total_times.append(total)
        segment_counts.append(segment_count)
        print(
            f"[stream] run {run_idx + 1:02d}/{runs}: "
            f"first={first_chunk_times[-1]:.3f}s, total={total:.3f}s, segments={segment_count}"
        )

    return {
        "endpoint": "/generate_stream",
        "runs": runs,
        "first_chunk_latency_s": _summarize(first_chunk_times),
        "total_latency_s": _summarize(total_times),
        "segments": {
            "avg": round(float(np.mean(segment_counts)), 2),
            "min": int(np.min(segment_counts)),
            "max": int(np.max(segment_counts)),
        },
    }


def main():
    args = _build_args()
    if args.runs <= 0:
        raise SystemExit("--runs must be > 0.")

    payload = _build_payload(args)
    summaries = []

    if args.mode in ("generate", "both"):
        summaries.append(_benchmark_generate(args.base_url, payload, args.runs))
    if args.mode in ("stream", "both"):
        summaries.append(_benchmark_stream(args.base_url, payload, args.runs))

    print("\nSummary")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
