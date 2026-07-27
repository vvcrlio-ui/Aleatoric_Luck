#!/usr/bin/env python3
"""Build and validate all FFCWS adapter artifacts for the shared engine."""

from __future__ import annotations

import sys
from pathlib import Path


ADAPTER_ROOT = Path(__file__).resolve().parent
PACKAGE_SRC = ADAPTER_ROOT / "src"
sys.path.insert(0, str(PACKAGE_SRC))

from ffcws_data_processor.pipeline import main as pipeline_main


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--config" not in arguments:
        arguments = [
            "--config",
            str(ADAPTER_ROOT / "contracts" / "ffc.yaml"),
            *arguments,
        ]
    pipeline_main(arguments)
