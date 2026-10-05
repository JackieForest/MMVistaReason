#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

from common import load_config, load_prompt, render_prompt


SHORT_MIN = 32
LONG_MAX = 2048


def compute_scale(w: int, h: int) -> float:
    short_side, long_side = min(w, h), max(w, h)
    need_up = short_side < SHORT_MIN
    need_down = long_side > LONG_MAX
    if not need_up and not need_down:
        return 1.0
    up_min = (SHORT_MIN / short_side) if need_up else 1.0
    down_max = (LONG_MAX / long_side) if need_down else 1.0
    if need_up and not need_down:
        return float(up_min)
    if need_down and not need_up:
        return float(down_max)
    return float(up_min) if up_min <= down_max else float(down_max)


def to_pil_image(image_obj: Any) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj
    if isinstance(image_obj, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(image_obj)))
    if isinstance(image_obj, dict):
        if image_obj.get("bytes"):
            return Image.open(io.BytesIO(image_obj["bytes"]))
        if image_obj.get("path"):
            return Image.open(image_obj["path"])
    raise TypeError(f"Unsupported image object: {type(image_obj)!r}")


def encode_image(image_obj: Any) -> str:
    image = ImageOps.exif_transpose(to_pil_image(image_obj))
    w, h = image.size
    scale = compute_scale(w, h)
    if scale != 1.0:
        new_w = max(1, int(math.ceil(w * scale)))
        new_h = max(1, int(math.ceil(h * scale)))
        image = image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def numeric_split_paths(input_splits_dir: Path, num_splits: int, shard_id: int, num_shards: int) -> list[Path]:
    return [input_splits_dir / f"{idx}.parquet" for idx in range(num_splits) if idx % num_shards == shard_id]


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad json {path}:{line_no}: {exc}") from exc
            rows[str(row.get("id", row.get("index", "")))] = row
    return rows


def load_existing_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("best_rollout") and "id" in row:
                ids.add(str(row["id"]))
    return ids


def split_response_for_answer_score(response: str, final_answer: str) -> tuple[str, str]:
    stripped_response = response or ""
    if not stripped_response.strip():
        raise RuntimeError("empty response")
    final_answer_match = re.search(r"(?is)<answer>\s*(.*?)\s*</answer>\.?\s*$", stripped_response)
    if not final_answer_match or not final_answer_match.group(1).strip():
        raise RuntimeError(f"response does not end with final answer tag: {stripped_response[-500:]}")

    target_start = final_answer_match.start(1)
    prefix = stripped_response[:target_start]
    return prefix, final_answer_match.group(1).strip()


def response_with_target(prefix: str, target: str) -> str:
    return prefix + target


def token_logprob(item: Any) -> float | None:
    if item is None:
        return None
    if isinstance(item, dict):
        value = item.get("logprob")
        return float(value) if value is not None else None
    value = getattr(item, "logprob", None)
    return float(value) if value is not None else None


def result_prompt_token_ids(result: dict[str, Any]) -> list[int]:
    for key in ("prompt_token_ids", "prompt_tokens"):
        value = result.get(key)
        if isinstance(value, list):
            return [int(x) for x in value]
    if result.get("choices"):
        choice = result["choices"][0]
        for key in ("prompt_token_ids", "prompt_tokens"):
            value = choice.get(key)
            if isinstance(value, list):
                return [int(x) for x in value]
    return []


def get_logprob_entry(entry: dict[str, Any], token_id: int) -> Any | None:
    if token_id in entry:
        return entry[token_id]
    key = str(token_id)
    if key in entry:
        return entry[key]
    return None


def actual_prompt_token_logprobs(result: dict[str, Any]) -> list[float | None]:
    prompt_logprobs = result.get("prompt_logprobs")
    if prompt_logprobs is None and result.get("choices"):
        prompt_logprobs = result["choices"][0].get("prompt_logprobs")
    if prompt_logprobs is None:
        raise RuntimeError(f"response has no prompt_logprobs keys={list(result.keys())}")

    token_ids = result_prompt_token_ids(result)
    if not token_ids:
        raise RuntimeError(
            "response has prompt_logprobs but no prompt_token_ids; "
            "vLLM may not support return_token_ids for this endpoint"
        )
    if len(token_ids) != len(prompt_logprobs):
        raise RuntimeError(f"prompt_token_ids length {len(token_ids)} != prompt_logprobs length {len(prompt_logprobs)}")

    values: list[float | None] = []
    for token_id, entry in zip(token_ids, prompt_logprobs):
        if entry is None:
            values.append(None)
            continue
        if isinstance(entry, dict):
            if "logprob" in entry and int(entry.get("token_id", token_id)) == token_id:
                values.append(token_logprob(entry))
                continue
            actual = get_logprob_entry(entry, token_id)
            if actual is None:
                raise RuntimeError(f"prompt token id {token_id} missing from prompt_logprobs entry")
            values.append(token_logprob(actual))
            continue
        values.append(token_logprob(entry))
    return values


def common_prefix_len(left: list[int], right: list[int]) -> int:
    count = 0
    for left_id, right_id in zip(left, right):
        if left_id != right_id:
            break
        count += 1
    return count


async def score_one_candidate(
    session: aiohttp.ClientSession,
    api_url: str,
    cfg: dict[str, Any],
    prompt_text: str,
    image_url: str | None,
    response: str,
    final_answer: str,
) -> dict[str, Any]:
    prefix, target = split_response_for_answer_score(response, final_answer)
    if not target:
        raise RuntimeError("empty answer target")

    prefix_text = response_with_target(prefix, "")
    full_text = response_with_target(prefix, target)
    prefix_messages: list[dict[str, Any]] = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    full_messages: list[dict[str, Any]] = [{"role": "user", "content": [{"type": "text", "text": prompt_text}]}]
    if image_url:
        prefix_messages[0]["content"].append({"type": "image_url", "image_url": {"url": image_url}})
        full_messages[0]["content"].append({"type": "image_url", "image_url": {"url": image_url}})
    prefix_messages.append({"role": "assistant", "content": prefix_text})
    full_messages.append({"role": "assistant", "content": full_text})

    common_payload = {
        "model": cfg["model"],
        "temperature": 0.0,
        "max_tokens": 1,
        "prompt_logprobs": int(cfg.get("score_top_logprobs", 1)),
        "return_token_ids": True,
    }
    max_retries = int(cfg.get("max_retries", 3))
    last_error = ""
    for attempt in range(max_retries):
        try:
            prefix_payload = {**common_payload, "messages": prefix_messages}
            full_payload = {**common_payload, "messages": full_messages}
            async with session.post(f"{api_url}/chat/completions", json=prefix_payload) as resp:
                prefix_text_resp = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"prefix HTTP {resp.status}: {prefix_text_resp[:1000]}")
                prefix_result = json.loads(prefix_text_resp)
            async with session.post(f"{api_url}/chat/completions", json=full_payload) as resp:
                full_text_resp = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"full HTTP {resp.status}: {full_text_resp[:1000]}")
                full_result = json.loads(full_text_resp)

            prefix_token_ids = result_prompt_token_ids(prefix_result)
            full_token_ids = result_prompt_token_ids(full_result)
            full_lps = actual_prompt_token_logprobs(full_result)
            target_start = common_prefix_len(prefix_token_ids, full_token_ids)
            target_lps = [lp for lp in full_lps[target_start:] if lp is not None]
            if not target_lps:
                raise RuntimeError(
                    f"no target logprobs prefix_tokens={len(prefix_token_ids)} full_tokens={len(full_token_ids)} "
                    f"target_chars={len(target)}"
                )
            total = float(sum(target_lps))
            count = len(target_lps)
            return {
                "answer_logprob_sum": total,
                "answer_token_count": count,
                "answer_logprob_mean": total / count,
                "answer_target": target,
                "prefix_token_count": len(prefix_token_ids),
                "full_token_count": len(full_token_ids),
                "target_start_token": target_start,
            }
        except Exception as exc:
            last_error = repr(exc)
            if attempt < max_retries - 1:
                await asyncio.sleep(2 * (attempt + 1))
    raise RuntimeError(last_error)


async def process_sample(
    sample: dict[str, Any],
    rollout_record: dict[str, Any],
    session: aiohttp.ClientSession,
    api_url: str,
    cfg: dict[str, Any],
    prompt_template: str,
) -> dict[str, Any]:
    rollouts = rollout_record.get("rollouts", [])
    if not rollouts:
        raise RuntimeError(f"id={sample.get('id')} has no rollouts")

    if len(rollouts) == 1:
        selected = dict(rollouts[0])
        selected["answer_logprob_mean"] = None
        selected["answer_logprob_sum"] = None
        selected["answer_token_count"] = 0
        selected["selection_reason"] = "single_correct_rollout"
        return build_output(sample, rollout_record, selected, rollouts, [], cfg)

    prompt_text = render_prompt(prompt_template, sample)
    image_url = encode_image(sample["image"]) if bool(cfg.get("use_image", True)) else None
    scored: list[dict[str, Any]] = []
    for rollout in rollouts:
        score = await score_one_candidate(
            session=session,
            api_url=api_url,
            cfg=cfg,
            prompt_text=prompt_text,
            image_url=image_url,
            response=str(rollout.get("response", "")),
            final_answer=str(rollout.get("final_answer", "")),
        )
        item = dict(rollout)
        item.update(score)
        scored.append(item)
    best = max(scored, key=lambda r: (r["answer_logprob_mean"], -int(r.get("rollout_id", 999999))))
    best["selection_reason"] = "max_normalized_answer_likelihood"
    return build_output(sample, rollout_record, best, rollouts, scored, cfg)


def build_output(
    sample: dict[str, Any],
    rollout_record: dict[str, Any],
    selected: dict[str, Any],
    original_rollouts: list[dict[str, Any]],
    scored_rollouts: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    meta_keys = [
        "index",
        "id",
        "question",
        "answer",
        "original_answer",
        "source",
        "original_id",
        "domain",
        "type",
        "subtype",
        "label_valid",
        "distill_model",
        "difficulty",
    ]
    out = {key: sample.get(key) for key in meta_keys if key in sample}
    out.update(
        {
            "selection_method": cfg.get("selection_method", "normalized_answer_likelihood"),
            "num_correct_rollouts": len(original_rollouts),
            "selected_rollout_id": selected.get("rollout_id"),
            "best_rollout": selected,
            "candidate_scores": [
                {
                    "rollout_id": item.get("rollout_id"),
                    "final_answer": item.get("final_answer"),
                    "answer_logprob_mean": item.get("answer_logprob_mean"),
                    "answer_logprob_sum": item.get("answer_logprob_sum"),
                    "answer_token_count": item.get("answer_token_count"),
                }
                for item in (scored_rollouts or [selected])
            ],
            "correct_rollout_ids": rollout_record.get("correct_rollout_ids"),
        }
    )
    return out


async def process_shard(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    input_splits_dir = Path(cfg["input_splits_dir"])
    input_rollouts_dir = Path(cfg["input_rollouts_dir"])
    output_dir = Path(cfg["output_dir"])
    shard_dir = output_dir / "shards"
    stop_dir = output_dir / "stop_files"
    shard_dir.mkdir(parents=True, exist_ok=True)
    stop_dir.mkdir(parents=True, exist_ok=True)

    output_file = shard_dir / f"shard_{args.shard_id}.jsonl"
    stop_file = stop_dir / f"shard_{args.shard_id}.flag"
    split_paths = numeric_split_paths(
        input_splits_dir,
        int(cfg.get("num_splits", 400)),
        args.shard_id,
        int(cfg.get("num_shards", 20)),
    )
    existing_ids = load_existing_ids(output_file)
    prompt_template = load_prompt(cfg)
    api_url = f"http://{args.url}:{args.port}/v1"
    timeout = aiohttp.ClientTimeout(total=int(cfg.get("request_timeout", 1800)))
    max_concurrent = int(cfg.get("max_concurrent", 8))

    work: list[tuple[dict[str, Any], dict[str, Any]]] = []
    total_rows = 0
    for split_path in split_paths:
        split_idx = int(split_path.stem)
        rollout_path = input_rollouts_dir / f"{split_idx}.jsonl"
        df = pd.read_parquet(split_path)
        rollout_by_id = load_jsonl_by_id(rollout_path)
        total_rows += len(df)
        for sample in df.to_dict(orient="records"):
            sid = str(sample.get("id", sample.get("index", "")))
            if sid in existing_ids:
                continue
            if sid not in rollout_by_id:
                raise RuntimeError(f"split={split_idx} id={sid} missing rollout record")
            work.append((sample, rollout_by_id[sid]))

    print(
        f"shard={args.shard_id} split_count={len(split_paths)} total_rows={total_rows} "
        f"existing={len(existing_ids)} remaining={len(work)}",
        flush=True,
    )
    if not work:
        stop_file.write_text("stop\n", encoding="utf-8")
        return

    sem = asyncio.Semaphore(max_concurrent)
    ok = 0
    failed = 0
    started = time.time()

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def run_one(item: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
            sample, rollout_record = item
            async with sem:
                return await process_sample(sample, rollout_record, session, api_url, cfg, prompt_template)

        tasks = [asyncio.create_task(run_one(item)) for item in work]
        with output_file.open("a", encoding="utf-8") as out_f:
            pbar = tqdm(total=len(tasks), desc=f"shard {args.shard_id}", unit="sample")
            for fut in asyncio.as_completed(tasks):
                try:
                    result = await fut
                    out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out_f.flush()
                    ok += 1
                except Exception as exc:
                    failed += 1
                    print(f"ERROR shard={args.shard_id}: {repr(exc)}", flush=True)
                pbar.update(1)
                pbar.set_postfix(ok=ok, failed=failed)
            pbar.close()

    elapsed = time.time() - started
    print(f"done shard={args.shard_id} ok={ok} failed={failed} elapsed={elapsed:.1f}s", flush=True)
    if failed == 0 and len(load_existing_ids(output_file)) >= total_rows:
        stop_file.write_text("stop\n", encoding="utf-8")
        print(f"wrote stop flag: {stop_file}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the best distillation rollout by normalized answer likelihood.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ[key] = ""

    cfg = load_config(args.config)
    asyncio.run(process_shard(args, cfg))


if __name__ == "__main__":
    main()
