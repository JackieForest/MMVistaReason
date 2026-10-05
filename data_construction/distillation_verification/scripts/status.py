#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import load_config


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def count_verify(path: Path) -> tuple[int, int, int, int, int]:
    if not path.exists():
        return 0, 0, 0, 0, 0
    latest: dict[str, dict] = {}
    json_errors = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                json_errors += 1
                continue
            key = str(obj.get("id", obj.get("index", "")))
            latest[key] = obj

    rows = correct = wrong = invalid = parse_failed = 0
    for obj in latest.values():
        if not obj.get("verify_valid"):
            parse_failed += int(obj.get("parse_failed_count", 0) or 0)
            continue
        rows += 1
        correct += int(obj.get("correct_count", 0) or 0)
        wrong += int(obj.get("wrong_count", 0) or 0)
        invalid += int(obj.get("invalid_count", 0) or 0)
        parse_failed += int(obj.get("parse_failed_count", 0) or 0)
    return rows, correct, wrong, invalid, parse_failed + json_errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Show distillation verification progress.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    input_dir = Path(cfg["input_rollout_dir"])
    output_root = Path(cfg["output_dir"])
    verify_dir = output_root / str(cfg.get("success_dir_name", "verify"))
    failed_dir = output_root / str(cfg.get("failed_dir_name", "failed_verify"))
    stop_dir = output_root / str(cfg.get("stop_dir_name", "verify_stop_files"))
    num_splits = int(cfg.get("num_splits", 400))
    expected_rollouts = int(cfg.get("expected_rollouts", 4))

    total_rows = total_done = total_failed = 0
    total_correct = total_wrong = total_invalid = total_parse_failed = 0
    stopped = 0
    completed_splits = 0

    for idx in range(num_splits):
        input_rows = count_lines(input_dir / f"{idx}.jsonl")
        done, correct, wrong, invalid, parse_failed = count_verify(verify_dir / f"{idx}.jsonl")
        failed = count_lines(failed_dir / f"{idx}.jsonl")
        stop = (stop_dir / f"verify_{idx}.flag").exists()
        if stop:
            stopped += 1
        if input_rows and done >= input_rows:
            completed_splits += 1
        total_rows += input_rows
        total_done += done
        total_failed += failed
        total_correct += correct
        total_wrong += wrong
        total_invalid += invalid
        total_parse_failed += parse_failed
        pct = done / input_rows * 100 if input_rows else 0
        print(
            f"split {idx:03d}: rows={input_rows} done={done} "
            f"rollout_judged={done * expected_rollouts} pct={pct:.2f}% "
            f"correct={correct} wrong={wrong} invalid={invalid} parse_failed={parse_failed} "
            f"failed_lines={failed} stop={stop}"
        )

    total_rollouts = total_rows * expected_rollouts
    judged = total_done * expected_rollouts
    pct_total = total_done / total_rows * 100 if total_rows else 0
    acc = total_correct / total_rollouts * 100 if total_rollouts else 0
    print(
        f"TOTAL: splits={completed_splits}/{num_splits} stop_flags={stopped}/{num_splits} "
        f"rows={total_rows} done={total_done} pct={pct_total:.2f}% "
        f"rollouts={total_rollouts} judged={judged} correct={total_correct} "
        f"wrong={total_wrong} invalid={total_invalid} parse_failed={total_parse_failed} "
        f"rollout_acc={acc:.2f}% failed_lines={total_failed}"
    )


if __name__ == "__main__":
    main()
