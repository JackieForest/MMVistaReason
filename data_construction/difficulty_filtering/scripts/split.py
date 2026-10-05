#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from common import load_config


def split_counts(total_rows: int, num_splits: int) -> list[int]:
    base = total_rows // num_splits
    rem = total_rows % num_splits
    return [base + (1 if i < rem else 0) for i in range(num_splits)]


def add_index_column(table: pa.Table, start_index: int) -> pa.Table:
    indexes = pa.array(range(start_index, start_index + table.num_rows), type=pa.int64())
    if "index" in table.column_names:
        col_idx = table.column_names.index("index")
        return table.set_column(col_idx, "index", indexes)
    return table.append_column("index", indexes)


def row_group_bounds(pf: pq.ParquetFile) -> tuple[list[int], list[int]]:
    starts: list[int] = []
    ends: list[int] = []
    cursor = 0
    for rg_idx in range(pf.num_row_groups):
        starts.append(cursor)
        cursor += pf.metadata.row_group(rg_idx).num_rows
        ends.append(cursor)
    return starts, ends


def compute_rg_window(
    row_starts: list[int],
    row_ends: list[int],
    split_start: int,
    split_end: int,
) -> tuple[int, int]:
    first = bisect.bisect_right(row_ends, split_start)
    last = bisect.bisect_left(row_starts, split_end)
    return first, last


def build_split_worker(
    source: str,
    out_path: str,
    split_idx: int,
    split_start: int,
    split_end: int,
    row_starts: list[int],
    row_ends: list[int],
    compression: str | None,
    arrow_threads: int,
) -> tuple[int, int]:
    arrow_threads = max(1, int(arrow_threads))
    pa.set_cpu_count(arrow_threads)
    pa.set_io_thread_count(arrow_threads)

    pf = pq.ParquetFile(source)
    first_rg, last_rg = compute_rg_window(row_starts, row_ends, split_start, split_end)
    writer: pq.ParquetWriter | None = None
    total_written = 0
    current_index = split_start
    final_path = Path(out_path)
    tmp_path = final_path.with_name(f"{final_path.name}.tmp.{os.getpid()}")
    if tmp_path.exists():
        tmp_path.unlink()

    try:
        for rg_idx in range(first_rg, last_rg):
            rg_start = row_starts[rg_idx]
            table = pf.read_row_group(rg_idx, use_threads=True)

            local_start = max(0, split_start - rg_start)
            local_end = min(table.num_rows, split_end - rg_start)
            if local_start or local_end != table.num_rows:
                table = table.slice(local_start, local_end - local_start)

            if table.num_rows == 0:
                continue

            table = add_index_column(table, current_index)
            current_index += table.num_rows
            total_written += table.num_rows

            if writer is None:
                writer = pq.ParquetWriter(
                    tmp_path,
                    table.schema,
                    compression=compression,
                    use_dictionary=False,
                )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    expected = split_end - split_start
    if total_written != expected:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Split {split_idx} wrote {total_written} rows, expected {expected}")

    tmp_path.replace(final_path)
    return split_idx, total_written


def valid_split(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        pf = pq.ParquetFile(path)
        return pf.metadata.num_rows == expected_rows and "index" in pf.schema_arrow.names
    except Exception:
        return False


def numeric_split_files(splits_dir: Path) -> list[Path]:
    return sorted(
        [p for p in splits_dir.glob("*.parquet") if p.stem.isdigit()],
        key=lambda p: int(p.stem),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split Pool.parquet or copy/link an existing split directory into numeric shards."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", default=None, help="Override source parquet path or split directory.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip complete numeric split files.")
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--arrow-threads", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    source = Path(args.source or cfg["source_data"])
    num_splits = int(cfg.get("num_splits", 50))
    workers = max(1, min(int(args.workers or cfg.get("split_workers", 8)), num_splits))
    arrow_threads = max(1, int(args.arrow_threads or cfg.get("split_arrow_threads", 2)))
    data_dir = Path(cfg["data_dir"])
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    existing_numeric = numeric_split_files(splits_dir)
    if existing_numeric and not args.overwrite and not args.resume:
        raise FileExistsError(f"{splits_dir} already has numeric split files; use --overwrite or --resume.")
    if args.overwrite:
        for path in existing_numeric:
            path.unlink()
        for path in splits_dir.glob("*.tmp.*"):
            path.unlink()

    if source.is_dir():
        source_files = numeric_split_files(source)
        if len(source_files) != num_splits:
            raise RuntimeError(f"Expected {num_splits} numeric parquet files in {source}, found {len(source_files)}")
        copied = 0
        for src in source_files:
            dst = splits_dir / src.name
            if args.resume and dst.exists():
                copied += 1
                continue
            if dst.exists():
                dst.unlink()
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            copied += 1
            print(f"split {src.stem}: {src} -> {dst}", flush=True)
        total_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in source_files)
        (splits_dir / "SPLIT_SUMMARY.txt").write_text(
            "\n".join(
                [
                    "4B Rollout Split Summary",
                    f"source_dir: {source}",
                    f"splits_dir: {splits_dir}",
                    f"total_rows: {total_rows}",
                    f"num_splits: {num_splits}",
                    "mode: directory copy/link",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Done. Prepared {copied} splits in {splits_dir}", flush=True)
        return

    pf = pq.ParquetFile(source)
    total_rows = pf.metadata.num_rows
    counts = split_counts(total_rows, num_splits)
    row_starts, row_ends = row_group_bounds(pf)
    compression = None if args.compression in {"", "none", "None"} else args.compression

    split_specs = []
    cursor = 0
    for split_idx, count in enumerate(counts):
        split_specs.append((split_idx, cursor, cursor + count))
        cursor += count

    print(f"[{cfg.get('name', 'qwen35_4b_pr')}] Streaming source: {source}", flush=True)
    print(f"Total rows: {total_rows}", flush=True)
    print(f"Num splits: {num_splits}", flush=True)
    print(f"Row groups: {pf.num_row_groups}", flush=True)
    print(f"Workers: {workers}", flush=True)
    print(f"Arrow threads per worker: {arrow_threads}", flush=True)

    results: dict[int, int] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {}
        for split_idx, split_start, split_end in split_specs:
            out_path = splits_dir / f"{split_idx}.parquet"
            expected_rows = split_end - split_start
            if args.resume and valid_split(out_path, expected_rows):
                results[split_idx] = expected_rows
                print(f"split {split_idx:02d}: already complete -> {out_path}", flush=True)
                continue
            if args.resume and out_path.exists():
                print(f"split {split_idx:02d}: removing incomplete/bad file -> {out_path}", flush=True)
                out_path.unlink()
            fut = ex.submit(
                build_split_worker,
                str(source),
                str(out_path),
                split_idx,
                split_start,
                split_end,
                row_starts,
                row_ends,
                compression,
                arrow_threads,
            )
            futures[fut] = (split_idx, expected_rows, out_path)

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Writing splits", unit="split"):
            split_idx, expected_rows, out_path = futures[fut]
            got_idx, written = fut.result()
            if got_idx != split_idx or written != expected_rows:
                raise RuntimeError(f"Split {split_idx} wrote {written}, expected {expected_rows}")
            results[split_idx] = written
            print(f"split {split_idx:02d}: wrote {written} rows -> {out_path}", flush=True)

    summary_lines = [
        "4B Rollout Split Summary",
        f"source: {source}",
        f"splits_dir: {splits_dir}",
        f"total_rows: {total_rows}",
        f"num_splits: {num_splits}",
        f"compression: {args.compression}",
        "",
        "Per split:",
    ]
    for split_idx in range(num_splits):
        summary_lines.append(f"{split_idx}\t{results[split_idx]}")
    (splits_dir / "SPLIT_SUMMARY.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Done. Split summary: {splits_dir / 'SPLIT_SUMMARY.txt'}", flush=True)


if __name__ == "__main__":
    main()
