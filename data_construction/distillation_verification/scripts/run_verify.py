#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from common import as_bool, load_config



SYSTEM_PROMPT = (
    "You are an impartial grading expert. Evaluate whether the candidate response "
    "correctly solves the given problem. Final-answer correctness is the primary "
    "criterion, while the reasoning trajectory should be considered as a secondary "
    "consistency check."
)

DISTILLATION_VERIFICATION_PROMPT = """Role.
Given an original question, a standard answer, and a complete candidate response, determine whether the response should be regarded as correct. The candidate response may contain both a reasoning trajectory and a final answer.

Evaluation Protocol.

1. Response Validity.
   - Reject responses that are incomplete, severely truncated, repetitive, or explicit refusals.
   - Minor stylistic issues, redundancy, or harmless formatting differences should not affect the judgment.

2. Final-Answer Correctness — Primary Criterion.
   - First determine whether the candidate's final answer is semantically equivalent to the standard answer.
   - Equivalent mathematical expressions, reasonable numerical precision differences, and semantically equivalent textual answers should be accepted.
   - For multiple-choice questions, compare the selected option and its corresponding content with the standard answer.
   - For multi-part questions, all required parts must be correctly answered.
   - If the final answer is incorrect, classify the response as incorrect regardless of the preceding reasoning.

3. Reasoning Consistency — Secondary Criterion.
   - If the final answer is correct, further examine whether the reasoning trajectory is generally consistent with the question, visual evidence, and final answer.
   - Check for major logical contradictions, mathematical errors, unsupported visual claims, or factual errors that substantially undermine the validity of the solution.
   - Minor omissions, shortcuts, imprecise wording, or non-essential intermediate imperfections may be tolerated if they do not affect the validity of the overall solution.
   - If the trajectory contains a substantive reasoning error that invalidates the solution process, classify the response as incorrect even if the final answer matches the standard answer.

4. Overall Judgment.
   - Classify the response as correct when the final answer is correct and the reasoning contains no major inconsistency.
   - Otherwise, classify the response as incorrect.

Original Question.
<Original Question Begin>
{question}
<Original Question End>

Standard Answer.
<Standard Answer Begin>
{gold_answer}
<Standard Answer End>

Candidate Response.
<Candidate Response Begin>
{llm_response}
<Candidate Response End>

Output Format.
Return exactly one valid JSON object on a single line. Do not output Markdown, code fences, or any additional text.

{{"verdict":"correct|incorrect","reasoning_consistent":true|false,"reason":"brief justification"}}

Final Instruction.
Evaluate the final answer first and use the reasoning trajectory as a secondary consistency check. Return only the required single-line JSON object.
"""


def truncate(value: Any, max_chars: int = 50000, preserve_final_answer: bool = False) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars] + "\n[TRUNCATED]"
    if preserve_final_answer:
        matches = list(re.finditer(r"<answer>.*?</answer>", text, flags=re.DOTALL | re.IGNORECASE))
        if matches:
            final_answer = matches[-1].group(0)
            if not text[matches[-1].end() :].strip() and final_answer not in truncated:
                truncated += "\n" + final_answer
    return truncated


def process_judgment(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    verdict = parsed.get("verdict")
    reasoning_consistent = parsed.get("reasoning_consistent")
    reason = parsed.get("reason")
    if verdict not in {"correct", "incorrect"}:
        return None
    if not isinstance(reasoning_consistent, bool) or not isinstance(reason, str):
        return None
    return {
        "verdict": verdict,
        "reasoning_consistent": reasoning_consistent,
        "reason": reason.strip(),
    }


def check_rollout_format(response: Any) -> tuple[bool, str]:
    text = "" if response is None else str(response).strip()
    if not re.search(r"<think>.*?</think>", text, flags=re.DOTALL | re.IGNORECASE):
        return False, "missing_think_tag"
    if not re.search(r"<answer>.*?</answer>", text, flags=re.DOTALL | re.IGNORECASE):
        return False, "missing_answer_tag"
    if not re.search(r"<answer>.*?</answer>\s*$", text, flags=re.DOTALL | re.IGNORECASE):
        return False, "bad_final_line"
    return True, ""


def format_prompt(tokenizer: AutoTokenizer, question: str, answer: str, candidate: str) -> str:
    prompt = DISTILLATION_VERIFICATION_PROMPT.format(
        question=truncate(question),
        gold_answer=truncate(answer),
        llm_response=truncate(candidate, preserve_final_answer=True),
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if tokenizer.bos_token and formatted.startswith(tokenizer.bos_token):
        formatted = formatted.removeprefix(tokenizer.bos_token)
    return formatted


def record_key(item: dict[str, Any]) -> str:
    return str(item.get("id", item.get("index", "")))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] skip bad json {path}:{line_no}: {exc}", flush=True)
    return rows


def dump_jsonl_line(f, obj: dict[str, Any]) -> None:
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    f.flush()


def build_tasks(item: dict[str, Any], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    rollouts = item.get("rollouts")
    if not isinstance(rollouts, list):
        rollouts = []
    tasks: list[dict[str, Any]] = []
    candidates: list[str] = []
    for ridx, rollout in enumerate(rollouts):
        if not isinstance(rollout, dict):
            rollout = {}
        candidate = rollout.get("response", "")
        candidate = str(candidate)
        format_valid, format_error = check_rollout_format(rollout.get("response", ""))
        candidates.append(candidate)
        tasks.append(
            {
                "rollout_id": rollout.get("rollout_id", ridx),
                "candidate": candidate,
                "final_answer": rollout.get("final_answer"),
                "format_valid": format_valid,
                "format_error": format_error,
                "finish_reason": rollout.get("finish_reason"),
                "response_chars": rollout.get("response_chars"),
                "has_answer_tag": rollout.get("has_answer_tag"),
            }
        )
    return tasks, candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify distillation trajectories with Qwen3.5-27B.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--index", type=int, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    index = args.index
    input_dir = Path(cfg["input_rollout_dir"])
    output_root = Path(cfg["output_dir"])
    verify_dir = output_root / str(cfg.get("success_dir_name", "verify"))
    failed_dir = output_root / str(cfg.get("failed_dir_name", "failed_verify"))
    stop_dir = output_root / str(cfg.get("stop_dir_name", "verify_stop_files"))
    verify_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)
    stop_dir.mkdir(parents=True, exist_ok=True)

    input_path = input_dir / f"{index}.jsonl"
    output_path = verify_dir / f"{index}.jsonl"
    failed_path = failed_dir / f"{index}.jsonl"
    stop_path = stop_dir / f"verify_{index}.flag"

    if stop_path.exists():
        print(f"stop flag exists: {stop_path}", flush=True)
        return
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    all_items = load_jsonl(input_path)
    done: dict[str, dict[str, Any]] = {}
    for item in load_jsonl(output_path):
        if item.get("verify_valid") and item.get("num_rollouts"):
            done[record_key(item)] = item
    todo = [item for item in all_items if record_key(item) not in done]

    print(f"split={index} input={len(all_items)} cached={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        stop_path.write_text(f"done {time.strftime('%F %T')}\n", encoding="utf-8")
        return

    model_path = str(cfg["verifier_model"])
    tp = int(cfg.get("vllm", {}).get("tensor_parallel_size", 1))
    model = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=tp,
        max_model_len=int(cfg.get("max_model_len", 32768)),
        enforce_eager=as_bool(cfg.get("enforce_eager", True)),
        gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.5)),
    )
    tokenizer = model.get_tokenizer()
    sampling = SamplingParams(
        temperature=float(cfg.get("temperature", 0.000001)),
        top_p=float(cfg.get("top_p", 0.9)),
        top_k=int(cfg.get("top_k", 1)),
        max_tokens=int(cfg.get("max_tokens", 2048)),
    )
    batch_size = int(cfg.get("batch_size", 64))
    expected_rollouts = int(cfg.get("expected_rollouts", 4))

    with output_path.open("a", encoding="utf-8") as out_f, failed_path.open("a", encoding="utf-8") as fail_f:
        for item in tqdm(todo, desc=f"verify split {index}", dynamic_ncols=True):
            try:
                meta_tasks, candidates = build_tasks(item, cfg)
                prompts: list[str] = []
                prompt_task_indices: list[int] = []
                for task_idx, (task, candidate) in enumerate(zip(meta_tasks, candidates)):
                    prompts.append(format_prompt(tokenizer, str(item.get("question", "")), str(item.get("answer", "")), candidate))
                    prompt_task_indices.append(task_idx)

                verifier_texts_by_task: dict[int, str] = {}
                for start in range(0, len(prompts), batch_size):
                    outputs = model.generate(prompts[start : start + batch_size], sampling)
                    for offset, output in enumerate(outputs):
                        task_idx = prompt_task_indices[start + offset]
                        verifier_texts_by_task[task_idx] = output.outputs[0].text

                judgments = []
                for task_idx, task in enumerate(meta_tasks):
                    verifier_response = verifier_texts_by_task.get(task_idx, "")
                    parsed = process_judgment(verifier_response)
                    parse_failed = parsed is None
                    if parse_failed:
                        verdict = "incorrect"
                        reasoning_consistent = False
                        reason = "Verifier output did not match the required JSON schema."
                    else:
                        verdict = parsed["verdict"]
                        reasoning_consistent = parsed["reasoning_consistent"]
                        reason = parsed["reason"]

                    # The protocol permits a correct verdict only when the trajectory
                    # is also consistent. Canonicalize contradictory verifier JSON.
                    is_correct = verdict == "correct" and reasoning_consistent
                    judgment = "correct" if is_correct else "wrong"
                    label = "A" if is_correct else "B"  # backward-compatible field
                    judgments.append(
                        {
                            **task,
                            "verifier_response": verifier_response,
                            "verdict": "correct" if is_correct else "incorrect",
                            "reasoning_consistent": reasoning_consistent,
                            "reason": reason,
                            "label": label,
                            "judgment": judgment,
                            "_parse_failed": parse_failed,
                        }
                    )
                correct = sum(1 for j in judgments if j["label"] == "A")
                wrong = sum(1 for j in judgments if j["label"] == "B")
                invalid = sum(1 for j in judgments if j["label"] == "C")
                parse_failed = sum(1 for j in judgments if j.pop("_parse_failed", False))
                verify_valid = len(judgments) == expected_rollouts and parse_failed < expected_rollouts
                result = {
                    "index": item.get("index"),
                    "id": item.get("id"),
                    "question": item.get("question"),
                    "answer": item.get("answer"),
                    "source": item.get("source"),
                    "original_id": item.get("original_id"),
                    "domain": item.get("domain"),
                    "type": item.get("type"),
                    "subtype": item.get("subtype"),
                    "verify_valid": verify_valid,
                    "num_rollouts": len(judgments),
                    "correct_count": correct,
                    "wrong_count": wrong,
                    "invalid_count": invalid,
                    "parse_failed_count": parse_failed,
                    "judgments": judgments,
                }
                dump_jsonl_line(out_f, result)
                if not result["verify_valid"]:
                    dump_jsonl_line(fail_f, result)
            except Exception as exc:
                err = {
                    "index": item.get("index"),
                    "id": item.get("id"),
                    "verify_valid": False,
                    "error": repr(exc),
                }
                dump_jsonl_line(fail_f, err)
                print(f"[ERROR] item {record_key(item)} failed: {exc}", file=sys.stderr, flush=True)

    final_rows = load_jsonl(output_path)
    valid_keys = {record_key(row) for row in final_rows if row.get("verify_valid")}
    valid_rows = len(valid_keys)
    if valid_rows >= len(all_items):
        stop_path.write_text(f"done {time.strftime('%F %T')}\n", encoding="utf-8")
        print(f"split={index} complete valid={valid_rows}/{len(all_items)}", flush=True)
    else:
        print(f"split={index} incomplete output_rows={len(final_rows)} valid={valid_rows}/{len(all_items)}", flush=True)


if __name__ == "__main__":
    main()
