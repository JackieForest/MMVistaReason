#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from common import load_config


def count_jsonl(path: Path) -> tuple[int, int, int]:
    if not path.exists():
        return 0, 0, 0
    lines = single = scored = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            lines += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            best = row.get("best_rollout", {})
            if best.get("selection_reason") == "single_correct_rollout":
                single += 1
            else:
                scored += 1
    return lines, single, scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    input_splits_dir = Path(cfg["input_splits_dir"])
    output_dir = Path(cfg["output_dir"])
    shard_dir = output_dir / "shards"
    stop_dir = output_dir / "stop_files"
    num_splits = int(cfg.get("num_splits", 400))
    num_shards = int(cfg.get("num_shards", 20))

    expected = [0 for _ in range(num_shards)]
    for idx in range(num_splits):
        path = input_splits_dir / f"{idx}.parquet"
        if path.exists():
            expected[idx % num_shards] += pq.ParquetFile(path).metadata.num_rows

    total_expected = total_done = total_single = total_scored = 0
    for shard_id in range(num_shards):
        path = shard_dir / f"shard_{shard_id}.jsonl"
        done, single, scored = count_jsonl(path)
        stop = (stop_dir / f"shard_{shard_id}.flag").exists()
        total_expected += expected[shard_id]
        total_done += done
        total_single += single
        total_scored += scored
        pct = done / expected[shard_id] * 100 if expected[shard_id] else 0
        print(
            f"shard {shard_id:02d}: expected={expected[shard_id]} done={done} "
            f"single={single} scored={scored} pct={pct:.2f}% stop={stop}"
        )
    pct_total = total_done / total_expected * 100 if total_expected else 0
    print(
        f"TOTAL: expected={total_expected} done={total_done} single={total_single} "
        f"scored={total_scored} pct={pct_total:.2f}%"
    )


if __name__ == "__main__":
    main()
