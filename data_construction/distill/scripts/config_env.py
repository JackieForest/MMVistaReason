#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
from typing import Any

from common import load_config


def flatten(obj: dict[str, Any], prefix: str = ""):
    for key, value in obj.items():
        env_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            yield from flatten(value, env_key)
        elif isinstance(value, list):
            continue
        else:
            if isinstance(value, bool):
                value = str(value).lower()
            yield env_key.upper(), value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    for key, value in flatten(cfg):
        print(f"{key}={shlex.quote(str(value))}")


if __name__ == "__main__":
    main()
