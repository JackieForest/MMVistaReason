#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from common import load_config


def load_shards(shard_dir: Path, num_shards: int) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for shard_id in range(num_shards):
        path = shard_dir / f"shard_{shard_id}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                sid = str(row.get("id", row.get("index", "")))
                if sid in rows:
                    raise RuntimeError(f"duplicate id={sid} in {path}:{line_no}")
                rows[sid] = row
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge best-of-n shard outputs into split-aligned jsonl files.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_splits_dir = Path(cfg["input_splits_dir"])
    output_dir = Path(cfg["output_dir"])
    shard_dir = output_dir / "shards"
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    num_splits = int(cfg.get("num_splits", 400))
    num_shards = int(cfg.get("num_shards", 20))

    rows_by_id = load_shards(shard_dir, num_shards)
    total_written = 0
    for split in tqdm(range(num_splits), desc="merge best rollouts", dynamic_ncols=True):
        out_path = split_dir / f"{split}.jsonl"
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"{out_path} exists; pass --overwrite")
        df = pd.read_parquet(input_splits_dir / f"{split}.parquet", columns=["id"])
        ids = df["id"].astype(str).tolist()
        missing = [sid for sid in ids if sid not in rows_by_id]
        if missing:
            raise RuntimeError(f"split {split}: missing {len(missing)} ids, preview={missing[:10]}")
        rows = [rows_by_id[sid] for sid in ids]
        write_jsonl(out_path, rows)
        total_written += len(rows)
        print(f"split={split} rows={len(rows)} path={out_path}", flush=True)

    expected = sum(
        pd.read_parquet(input_splits_dir / f"{split}.parquet", columns=["id"]).shape[0]
        for split in range(num_splits)
    )
    if total_written != expected:
        raise RuntimeError(f"total_written={total_written} expected={expected}")
    print(f"done total_written={total_written} output={split_dir}", flush=True)


if __name__ == "__main__":
    main()
