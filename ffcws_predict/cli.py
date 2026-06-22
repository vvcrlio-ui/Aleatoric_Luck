"""Command line interface for the FFCWS prediction framework."""

from __future__ import annotations

import argparse

from .pipeline import audit_features, learning_curve, prepare, summarize, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ffcws-predict")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["prepare", "audit-features", "train", "learning-curve", "summarize"]:
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        prepare(args.config)
    elif args.command == "audit-features":
        audit_features(args.config)
    elif args.command == "train":
        train(args.config)
    elif args.command == "learning-curve":
        learning_curve(args.config)
    elif args.command == "summarize":
        summarize(args.config)
    else:
        raise ValueError(args.command)
