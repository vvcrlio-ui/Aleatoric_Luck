#!/usr/bin/env python3
"""CLI wrapper for the isolated FFCWS adapter package."""

from __future__ import annotations

import sys
from pathlib import Path


PACKAGE_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(PACKAGE_SRC))

from ffcws_data_processor.pipeline import main


if __name__ == "__main__":
    main()
