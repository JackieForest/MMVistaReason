#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from common import load_config


def count_valid_jsonl(path: Path, valid_key: str, expected_rollouts: int) -> tuple[int, int, int]:
    if not path.exists():
        return 0, 0, 0
    valid = 0
    tag_complete = 0
    lines = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            lines += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rollouts = obj.get("rollouts", [])
            if obj.get(valid_key) and isinstance(rollouts, list) and len(rollouts) == expected_rollouts:
                valid += 1
                if all(
                    isinstance(r, dict) and not r.get("error") and bool(r.get("has_answer_tag"))
                    for r in rollouts
                ):
                    tag_complete += 1
    return valid, tag_complete, lines


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show rollout split progress.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(cfg["data_dir"])
    output_dir = Path(cfg["output_dir"])
    splits_dir = data_dir / "splits"
    success_name = str(cfg.get("success_dir_name", "rollouts"))
    failed_name = str(cfg.get("failed_dir_name", "failed_rollouts"))
    valid_key = str(cfg.get("valid_key", "rollout_valid"))
    expected_rollouts = int(cfg.get("rollouts_per_sample", 4))
    labels_dir = output_dir / success_name
    failed_dir = output_dir / failed_name
    stop_dir = output_dir / "stop_files"

    total_rows = total_done = total_tag_complete = total_fail = total_lines = 0
    split_paths = sorted(
        [p for p in splits_dir.glob("*.parquet") if p.stem.isdigit()],
        key=lambda p: int(p.stem),
    )
    for split_path in split_paths:
        idx = int(split_path.stem)
        rows = pq.ParquetFile(split_path).metadata.num_rows
        done, tag_complete, success_lines = count_valid_jsonl(labels_dir / f"{idx}.jsonl", valid_key, expected_rollouts)
        fail = count_lines(failed_dir / f"{idx}.jsonl")
        stop = (stop_dir / f"rollout_{idx}.flag").exists()
        total_rows += rows
        total_done += done
        total_tag_complete += tag_complete
        total_fail += fail
        total_lines += success_lines
        pct = done / rows * 100 if rows else 0
        print(
            f"split {idx:02d}: rows={rows} done={done} tag_complete={tag_complete} "
            f"success_lines={success_lines} fail_lines={fail} pct={pct:.2f}% stop={stop}"
        )
    pct_total = total_done / total_rows * 100 if total_rows else 0
    print(
        f"TOTAL: rows={total_rows} done={total_done} tag_complete={total_tag_complete} "
        f"success_lines={total_lines} fail_lines={total_fail} pct={pct_total:.2f}%"
    )


if __name__ == "__main__":
    main()
