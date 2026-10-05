#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from common import load_config


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def load_best_rows(shard_dir: Path, num_shards: int) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for shard_id in tqdm(range(num_shards), desc="load best shards", dynamic_ncols=True):
        path = shard_dir / f"shard_{shard_id}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        count = 0
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sid = str(row.get("id", row.get("index", "")))
                if not sid:
                    raise RuntimeError(f"missing id in {path}:{line_no}")
                if sid in rows:
                    raise RuntimeError(f"duplicate id={sid} in {path}:{line_no}")
                if not isinstance(row.get("best_rollout"), dict):
                    raise RuntimeError(f"missing best_rollout for id={sid} in {path}:{line_no}")
                rows[sid] = row
                count += 1
        logging.info("loaded shard=%s rows=%s path=%s", shard_id, count, path)
    return rows


def final_rollout_row(best_row: dict[str, Any]) -> dict[str, Any]:
    best_rollout = dict(best_row["best_rollout"])
    return {
        "index": best_row.get("index"),
        "id": best_row.get("id"),
        "question": best_row.get("question"),
        "answer": best_row.get("answer"),
        "original_answer": best_row.get("original_answer"),
        "source": best_row.get("source"),
        "original_id": best_row.get("original_id"),
        "domain": best_row.get("domain"),
        "type": best_row.get("type"),
        "subtype": best_row.get("subtype"),
        "label_valid": best_row.get("label_valid"),
        "distill_model": best_row.get("distill_model"),
        "difficulty": best_row.get("difficulty"),
        "rollout_id": best_row.get("selected_rollout_id"),
        "response": best_rollout.get("response"),
        "final_answer": best_rollout.get("final_answer"),
        "has_answer_tag": best_rollout.get("has_answer_tag"),
        "finish_reason": best_rollout.get("finish_reason"),
        "response_chars": best_rollout.get("response_chars"),
        "selection_method": best_row.get("selection_method"),
        "selection_reason": best_rollout.get("selection_reason"),
        "num_correct_rollouts": best_row.get("num_correct_rollouts"),
        "correct_rollout_ids": best_row.get("correct_rollout_ids"),
        "answer_logprob_mean": best_rollout.get("answer_logprob_mean"),
        "answer_logprob_sum": best_rollout.get("answer_logprob_sum"),
        "answer_token_count": best_rollout.get("answer_token_count"),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build final distillation rollouts with exactly one selected rollout per sample."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <config output_dir>/final_rollouts.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Defaults to <config output_dir>/build_final_rollouts.log.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_splits_dir = Path(cfg["input_splits_dir"])
    best_output_dir = Path(cfg["output_dir"])
    shard_dir = best_output_dir / "shards"
    output_dir = Path(args.output_dir) if args.output_dir else best_output_dir / "final_rollouts"
    log_file = Path(args.log_file) if args.log_file else best_output_dir / "build_final_rollouts.log"
    setup_logging(log_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_splits = int(cfg.get("num_splits", 400))
    num_shards = int(cfg.get("num_shards", 20))

    logging.info("input_splits_dir=%s", input_splits_dir)
    logging.info("best_shard_dir=%s", shard_dir)
    logging.info("output_dir=%s", output_dir)
    logging.info("num_splits=%s num_shards=%s overwrite=%s", num_splits, num_shards, args.overwrite)

    rows_by_id = load_best_rows(shard_dir, num_shards)
    logging.info("loaded total best rows=%s", len(rows_by_id))

    seen_ids: set[str] = set()
    total_written = 0
    for split in tqdm(range(num_splits), desc="write final rollouts", dynamic_ncols=True):
        split_path = input_splits_dir / f"{split}.parquet"
        out_path = output_dir / f"{split}.jsonl"
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"{out_path} exists; pass --overwrite")

        df = pd.read_parquet(split_path, columns=["id"])
        ids = df["id"].astype(str).tolist()
        repeated = [sid for sid in ids if sid in seen_ids]
        if repeated:
            raise RuntimeError(f"split {split}: ids repeated from earlier splits, preview={repeated[:10]}")
        missing = [sid for sid in ids if sid not in rows_by_id]
        if missing:
            raise RuntimeError(f"split {split}: missing {len(missing)} ids, preview={missing[:10]}")

        rows = [final_rollout_row(rows_by_id[sid]) for sid in ids]
        write_jsonl(out_path, rows)
        seen_ids.update(ids)
        total_written += len(rows)
        logging.info("wrote split=%s rows=%s path=%s", split, len(rows), out_path)

    extra_ids = set(rows_by_id) - seen_ids
    if extra_ids:
        preview = sorted(extra_ids)[:10]
        raise RuntimeError(f"best shards contain {len(extra_ids)} ids not present in final_splits, preview={preview}")
    if total_written != len(rows_by_id):
        raise RuntimeError(f"total_written={total_written} loaded_best_rows={len(rows_by_id)}")

    logging.info("done total_written=%s output_dir=%s", total_written, output_dir)


if __name__ == "__main__":
    main()
