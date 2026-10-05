#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parent


def resolve_path(path: str | Path, base: Path = PROJECT_DIR) -> Path:
    path = Path(path)
    return path if path.is_absolute() else base / path


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def flatten_config(obj: dict[str, Any], prefix: str = ""):
    for key, value in obj.items():
        env_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            yield from flatten_config(value, env_key)
        elif isinstance(value, list):
            continue
        else:
            if isinstance(value, bool):
                value = str(value).lower()
            yield env_key.upper(), value


def load_prompt(cfg: dict[str, Any]) -> str:
    return resolve_path(cfg["prompt"]).read_text(encoding="utf-8").strip()


def truncate_text(value: Any, max_chars: int = 20000) -> str:
    if value is None:
        return ""
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[TRUNCATED]"


def render_prompt(template: str, example: dict[str, Any]) -> str:
    prompt = template
    replacements = {
        "{image}": "",
        "<<IMAGE>>": "",
        "{question}": truncate_text(example.get("question", "")),
        "<<QUESTION>>": truncate_text(example.get("question", "")),
        "{answer}": truncate_text(example.get("answer", "")),
        "<<ANSWER>>": truncate_text(example.get("answer", "")),
        "{id}": truncate_text(example.get("id", example.get("index", ""))),
        "<<ID>>": truncate_text(example.get("id", example.get("index", ""))),
    }
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt.strip()


def extract_final_answer(response: str) -> tuple[str, bool]:
    if not response:
        return "", False
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", response, flags=re.DOTALL | re.IGNORECASE)
    if not matches:
        return "", False
    return matches[-1].strip(), True
