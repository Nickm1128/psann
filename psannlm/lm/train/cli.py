"""0.x YAML entrypoint delegating to the canonical training command."""

from __future__ import annotations
import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PSANN-LM from YAML")
    parser.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    from ...cli import main as canonical_main

    print(
        "DeprecationWarning: python -m psannlm.lm.train.cli is deprecated; use python -m psannlm train --config.",
        file=sys.stderr,
    )
    return canonical_main(
        ["train", *(argv if argv is not None else sys.argv[1:])], _legacy_entry=True
    )


if __name__ == "__main__":
    raise SystemExit(main())
