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
from datasets import load_dataset
from PIL import Image, ImageOps
from tqdm import tqdm

from common import extract_final_answer, load_config, load_prompt, render_prompt


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


def load_existing_indexes(path: Path, valid_key: str, expected_rollouts: int) -> set[int]:
    if not path.exists():
        return set()
    existing: set[int] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rollouts = obj.get("rollouts", [])
            if obj.get(valid_key) and "index" in obj and isinstance(rollouts, list) and len(rollouts) == expected_rollouts:
                existing.add(int(obj["index"]))
    return existing


def metadata_record(example: dict[str, Any], local_idx: int) -> dict[str, Any]:
    keys = [
        "id",
        "source",
        "original_id",
        "domain",
        "type",
        "subtype",
    ]
    record = {
        "index": int(example.get("index", local_idx)),
        "id": example.get("id", int(example.get("index", local_idx))),
        "question": str(example.get("question", "")),
        "answer": str(example.get("answer", "")),
    }
    for key in keys:
        if key in example:
            record[key] = example.get(key)
    return record


def normalize_reasoning_tags(response: str) -> str:
    if not response or "<think>" in response.lower():
        return response

    text = response.strip()
    if not text:
        return response

    answer_line_match = re.search(
        r"(?is)(?:^|\n)\s*(therefore,\s*the\s*final\s*answer\s*is\s*<answer>.*?</answer>\.?)\s*$",
        text,
    )
    if answer_line_match:
        answer_line = answer_line_match.group(1).strip()
        reasoning = text[: answer_line_match.start()].strip()
        if reasoning:
            return f"<think>\n{reasoning}\n</think>\n{answer_line}"
        return f"<think>\n\n</think>\n{answer_line}"

    answer_tag_match = re.search(r"(?is)<answer>.*?</answer>\.?\s*$", text)
    if answer_tag_match:
        answer_part = text[answer_tag_match.start() :].strip()
        reasoning = text[: answer_tag_match.start()].strip()
        if reasoning:
            return f"<think>\n{reasoning}\n</think>\nTherefore, the final answer is {answer_part}"
        return f"<think>\n\n</think>\nTherefore, the final answer is {answer_part}"

    return f"<think>\n{text}\n</think>"


async def request_one_rollout(
    rollout_id: int,
    session: aiohttp.ClientSession,
    api_url: str,
    cfg: dict[str, Any],
    content: list[dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": content}],
        "temperature": cfg.get("temperature", 0.7),
        "top_p": cfg.get("top_p", 0.8),
        "max_tokens": cfg.get("max_tokens", 4096),
    }
    for key in ("top_k", "min_p", "presence_penalty", "repetition_penalty"):
        if key in cfg:
            payload[key] = cfg[key]
    if "enable_thinking" in cfg:
        payload["chat_template_kwargs"] = {"enable_thinking": bool(cfg.get("enable_thinking"))}

    max_retries = int(cfg.get("max_retries", 3))
    last_error = ""
    for attempt in range(max_retries):
        try:
            async with session.post(f"{api_url}/chat/completions", json=payload) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    raise RuntimeError(f"HTTP {resp.status}: {text[:1000]}")
                result = json.loads(text)
            choice = result["choices"][0]
            response = choice["message"]["content"]
            if bool(cfg.get("enable_thinking")) and response and not response.lstrip().startswith("<think>"):
                response = "<think>" + response
            response = normalize_reasoning_tags(response)
            final_answer, has_answer_tag = extract_final_answer(response)
            return {
                "rollout_id": rollout_id,
                "response": response,
                "final_answer": final_answer,
                "has_answer_tag": has_answer_tag,
                "finish_reason": choice.get("finish_reason", ""),
                "response_chars": len(response or ""),
            }
        except Exception as exc:
            last_error = repr(exc)
            if attempt < max_retries - 1:
                await asyncio.sleep(2 * (attempt + 1))

    return {
        "rollout_id": rollout_id,
        "response": "",
        "final_answer": "",
        "has_answer_tag": False,
        "finish_reason": "",
        "response_chars": 0,
        "error": last_error,
    }


async def generate_one_sample(
    local_idx: int,
    example: dict[str, Any],
    session: aiohttp.ClientSession,
    api_url: str,
    cfg: dict[str, Any],
    prompt_template: str,
) -> dict[str, Any]:
    record = metadata_record(example, local_idx)
    prompt = render_prompt(prompt_template, example)
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if bool(cfg.get("use_image", True)):
        content.append({"type": "image_url", "image_url": {"url": encode_image(example["image"])}})

    rollouts_per_sample = int(cfg.get("rollouts_per_sample", 4))
    tasks = [
        asyncio.create_task(request_one_rollout(rollout_id, session, api_url, cfg, content))
        for rollout_id in range(rollouts_per_sample)
    ]
    rollouts = await asyncio.gather(*tasks)

    failed_rollouts = [item for item in rollouts if item.get("error")]
    if len(failed_rollouts) == rollouts_per_sample:
        return {
            **record,
            "rollout_valid": False,
            "num_rollouts": rollouts_per_sample,
            "rollouts": rollouts,
            "error": "all rollouts failed",
            "failed_rollout_ids": [item.get("rollout_id", -1) for item in failed_rollouts],
            "failed_rollout_errors": {str(item.get("rollout_id", -1)): item.get("error") for item in failed_rollouts},
        }

    return {
        **record,
        "rollout_valid": True,
        "num_rollouts": rollouts_per_sample,
        "rollouts": rollouts,
        "num_failed_rollouts": len(failed_rollouts),
        "failed_rollout_ids": [item.get("rollout_id", -1) for item in failed_rollouts],
    }


async def process_split(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    data_dir = Path(cfg["data_dir"])
    output_dir = Path(cfg["output_dir"])
    input_file = data_dir / "splits" / f"{args.index}.parquet"
    success_name = str(cfg.get("success_dir_name", "rollouts"))
    failed_name = str(cfg.get("failed_dir_name", "failed_rollouts"))
    valid_key = str(cfg.get("valid_key", "rollout_valid"))
    jsonl_dir = output_dir / success_name
    failed_dir = output_dir / failed_name
    stop_dir = output_dir / "stop_files"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)
    stop_dir.mkdir(parents=True, exist_ok=True)

    output_file = jsonl_dir / f"{args.index}.jsonl"
    failed_file = failed_dir / f"{args.index}.jsonl"
    prompt_template = load_prompt(cfg)
    cache_dir = Path(cfg.get("hf_datasets_cache_dir", output_dir / "hf_datasets_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{cfg.get('name', '?')}] Loading split {args.index}: {input_file}", flush=True)
    ds = load_dataset("parquet", data_files=str(input_file), split="train", cache_dir=str(cache_dir))
    expected_rollouts = int(cfg.get("rollouts_per_sample", 4))
    existing = load_existing_indexes(output_file, valid_key, expected_rollouts)
    work = [i for i in range(len(ds)) if int(ds[i].get("index", i)) not in existing]
    print(f"rows={len(ds)} done={len(existing)} remaining={len(work)}", flush=True)

    if not work:
        (stop_dir / f"rollout_{args.index}.flag").write_text("stop\n", encoding="utf-8")
        print("Split already complete; wrote stop flag.", flush=True)
        return

    api_url = f"http://{args.url}:{args.port}/v1"
    timeout = aiohttp.ClientTimeout(total=int(cfg.get("request_timeout", 600)))
    max_concurrent = int(cfg.get("max_concurrent", 16))
    max_pending = max_concurrent * int(cfg.get("max_pending_multiplier", 1))

    ok = 0
    fail = 0
    started = time.time()
    iterator = iter(work)
    active: set[asyncio.Task] = set()

    def schedule(session: aiohttp.ClientSession) -> bool:
        try:
            i = next(iterator)
        except StopIteration:
            return False
        task = asyncio.create_task(generate_one_sample(i, ds[i], session, api_url, cfg, prompt_template))
        active.add(task)
        return True

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for _ in range(min(max_pending, len(work))):
            schedule(session)

        with output_file.open("a", encoding="utf-8") as out_f, failed_file.open("a", encoding="utf-8") as fail_f:
            pbar = tqdm(total=len(work), desc=f"split {args.index}", unit="sample")
            while active:
                done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    record = task.result()
                    if record.get(valid_key):
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out_f.flush()
                        ok += 1
                    else:
                        fail_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fail_f.flush()
                        fail += 1
                    pbar.update(1)
                    pbar.set_postfix(ok=ok, fail=fail)
                    schedule(session)
            pbar.close()

    elapsed = time.time() - started
    print(f"Done split {args.index}: ok={ok} fail={fail} elapsed={elapsed:.1f}s", flush=True)
    if len(load_existing_indexes(output_file, valid_key, expected_rollouts)) >= len(ds):
        (stop_dir / f"rollout_{args.index}.flag").write_text("stop\n", encoding="utf-8")
        print("Split complete; wrote stop flag.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3.5 distill rollouts for one split.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--url", required=True, help="vLLM server host without port.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ[key] = ""

    cfg = load_config(args.config)
    asyncio.run(process_split(args, cfg))


if __name__ == "__main__":
    main()
