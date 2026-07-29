"""Logging helpers shared by NK grid entry points."""

from __future__ import annotations

import sys
from datetime import datetime


def log_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[nk_grid] {timestamp} {message}", file=sys.stderr, flush=True)
