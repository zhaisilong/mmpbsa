#!/usr/bin/env python3
"""Run this SP2016 case's validation-local source/model preflight."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from modeling.common.preflight import cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(cli(default_model=HERE.with_name("model.yaml")))
