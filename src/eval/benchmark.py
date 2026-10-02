"""Measure warmed local decision latency, including tokenization and decoding."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path

from src.eval.jev import _canonical_public, read_rows, sha256
from src.inference.engine import DecisionEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backend", choices=("hf", "xtuner", "npu"), default="npu")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=96)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.batch_size < 1:
        parser.error("warmup must be nonnegative and repeats must be positive")
    if args.output.exists():
        parser.error("Output exists; choose a fresh destination")
    rows = [_canonical_public(row) for row in read_rows(args.data)]
    if not rows:
        parser.error("The dataset is empty")
    engine = DecisionEngine(args.checkpoint, backend=args.backend)
    for i in range(args.warmup):
        engine.predict_batch([rows[(i * args.batch_size + j) % len(rows)] for j in range(args.batch_size)])
    samples, decisions = [], 0
    for i in range(args.repeats):
        start = time.perf_counter()
        results = engine.predict_batch([rows[(i * args.batch_size + j) % len(rows)] for j in range(args.batch_size)])
        samples.append((time.perf_counter() - start) * 1000)
        decisions += sum(len(result["answers"]) for result in results)
    report = {
        "backend": args.backend,
        "scope": "template, tokenization, device forward, probability decoding; no HTTP",
        "batch_size": args.batch_size,
        "dataset_sha256": sha256(args.data),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[math.ceil(len(samples) * 0.95) - 1],
        "requests_per_second": args.repeats * args.batch_size * 1000 / sum(samples),
        "decisions_per_second": decisions * 1000 / sum(samples),
        "samples_ms": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({key: value for key, value in report.items() if key != "samples_ms"}))


if __name__ == "__main__":
    main()
