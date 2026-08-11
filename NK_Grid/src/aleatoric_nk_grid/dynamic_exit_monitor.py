"""Operator-facing retry policy for dynamic afterany jobs.

Durable artefacts remain authoritative; this module only turns a completed
process exit code into a structured monitoring decision.  In particular,
superseded targets are terminal coordination outcomes, not retry candidates
and not production data-integrity alarms.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Sequence

from .generation_control import (
    PROTOCOL_EXIT_CODE,
    RETRYABLE_EXIT_CODE,
    SUCCESS_EXIT_CODE,
    SUPERSEDED_EXIT_CODE,
)


@dataclass(frozen=True)
class ExitDecision:
    exit_code: int
    classification: str
    retry: bool
    production_integrity_alarm: bool
    terminal: bool


def classify_dynamic_exit(exit_code: int) -> ExitDecision:
    """Return the sole automatic-retry policy shared by operators and tests."""

    code = int(exit_code)
    if code == SUCCESS_EXIT_CODE:
        return ExitDecision(code, "success", False, False, True)
    if code == PROTOCOL_EXIT_CODE:
        return ExitDecision(code, "protocol-or-corruption", False, True, True)
    if code == RETRYABLE_EXIT_CODE:
        return ExitDecision(code, "busy-or-recovery-required", True, False, False)
    if code == SUPERSEDED_EXIT_CODE:
        return ExitDecision(code, "superseded", False, False, True)
    return ExitDecision(code, "unexpected-exit", False, True, True)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Classify a completed dynamic queue job for monitoring/retry"
    )
    parser.add_argument("exit_code", type=int)
    parser.add_argument(
        "--phase",
        choices=("prep", "work", "close", "verify", "finalize"),
        required=True,
    )
    args = parser.parse_args(argv)
    payload = {"phase": args.phase, **asdict(classify_dynamic_exit(args.exit_code))}
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
