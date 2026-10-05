#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex

from common import flatten_config, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    for key, value in flatten_config(cfg):
        print(f"{key}={shlex.quote(str(value))}")


if __name__ == "__main__":
    main()
