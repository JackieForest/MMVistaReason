#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


def split_counts(total_rows: int, num_splits: int) -> list[int]:
    base = total_rows // num_splits
    rem = total_rows % num_splits
    return [base + (1 if i < rem else 0) for i in range(num_splits)]


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


def valid_split(path: Path, expected_rows: int) -> bool:
    if not path.exists():
        return False
    try:
        return pq.ParquetFile(path).metadata.num_rows == expected_rows
    except Exception:
        return False


def write_one_split(
    source: str,
    output_path: str,
    split_idx: int,
    split_start: int,
    split_end: int,
    row_starts: list[int],
    row_ends: list[int],
    compression: str | None,
) -> tuple[int, int, str]:
    pf = pq.ParquetFile(source)
    first_rg, last_rg = compute_rg_window(row_starts, row_ends, split_start, split_end)
    final_path = Path(output_path)
    tmp_path = final_path.with_name(f"{final_path.name}.tmp.{os.getpid()}")
    if tmp_path.exists():
        tmp_path.unlink()

    writer: pq.ParquetWriter | None = None
    written = 0
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

            if writer is None:
                writer = pq.ParquetWriter(
                    tmp_path,
                    table.schema,
                    compression=compression,
                    use_dictionary=False,
                )
            writer.write_table(table)
            written += table.num_rows
    finally:
        if writer is not None:
            writer.close()

    expected = split_end - split_start
    if written != expected:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"split {split_idx} wrote {written} rows, expected {expected}")

    if writer is None:
        schema = pf.schema_arrow
        empty = pa.Table.from_arrays([pa.array([], type=field.type) for field in schema], schema=schema)
        pq.write_table(empty, tmp_path, compression=compression, use_dictionary=False)

    tmp_path.replace(final_path)
    return split_idx, written, str(final_path)


def resolve_source(path: Path) -> Path:
    if path.exists():
        return path
    raise FileNotFoundError(f"Source parquet not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split qwen35_4b_pr Pool parquet into ordered parquet shards."
    )
    parser.add_argument("--source", type=Path, required=True, help="Input Pool Parquet file.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for output shards.")
    parser.add_argument("--num-splits", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--compression", default="zstd", help="zstd/snappy/none")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip complete existing split files.")
    args = parser.parse_args()

    source = resolve_source(args.source)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    num_splits = int(args.num_splits)
    workers = max(1, min(int(args.workers), num_splits))
    compression = None if args.compression in {"", "none", "None"} else args.compression

    existing = sorted(output_dir.glob("*.parquet"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    numeric_existing = [p for p in existing if p.stem.isdigit()]
    if numeric_existing and not args.overwrite and not args.resume:
        raise FileExistsError(f"{output_dir} already has numeric parquet splits; use --overwrite or --resume.")
    if args.overwrite:
        for path in numeric_existing:
            path.unlink()
        for path in output_dir.glob("*.tmp.*"):
            path.unlink()

    pf = pq.ParquetFile(source)
    total_rows = pf.metadata.num_rows
    counts = split_counts(total_rows, num_splits)
    row_starts, row_ends = row_group_bounds(pf)

    print(
        f"Splitting source parquet in original row order\n"
        f"source={source}\n"
        f"output_dir={output_dir}\n"
        f"total_rows={total_rows}\n"
        f"num_splits={num_splits}\n"
        f"row_groups={pf.num_row_groups}\n"
        f"workers={workers}\n"
        f"compression={args.compression}",
        flush=True,
    )

    specs = []
    cursor = 0
    for split_idx, count in enumerate(counts):
        specs.append((split_idx, cursor, cursor + count))
        cursor += count

    results: dict[int, int] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for split_idx, split_start, split_end in specs:
            output_path = output_dir / f"{split_idx}.parquet"
            expected = split_end - split_start
            if args.resume and valid_split(output_path, expected):
                results[split_idx] = expected
                print(f"split {split_idx:02d}: already complete rows={expected}", flush=True)
                continue
            if args.resume and output_path.exists():
                output_path.unlink()

            fut = executor.submit(
                write_one_split,
                str(source),
                str(output_path),
                split_idx,
                split_start,
                split_end,
                row_starts,
                row_ends,
                compression,
            )
            futures[fut] = split_idx

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Writing splits", unit="split"):
            split_idx, written, output_path = fut.result()
            results[split_idx] = written
            print(f"split {split_idx:02d}: wrote rows={written} -> {output_path}", flush=True)

    if len(results) != num_splits:
        missing = sorted(set(range(num_splits)) - set(results))
        raise RuntimeError(f"Missing split results: {missing}")

    summary_path = output_dir / f"SPLIT_{num_splits}_SUMMARY.txt"
    lines = [
        f"Qwen35 4B PR {num_splits}-way Split Summary",
        f"source: {source}",
        f"output_dir: {output_dir}",
        f"total_rows: {total_rows}",
        f"num_splits: {num_splits}",
        f"compression: {args.compression}",
        "",
        "split\trows\tpath",
    ]
    for split_idx in range(num_splits):
        lines.append(f"{split_idx}\t{results[split_idx]}\t{output_dir / f'{split_idx}.parquet'}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\nDone.")
    print(f"total_rows: {total_rows}")
    print(f"written_rows: {sum(results.values())}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
