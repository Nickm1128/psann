"""Legacy training command delegating to python -m psannlm train."""

from __future__ import annotations
import sys
from typing import Iterable
from ._train import build_parser, str2bool


def main(argv: Iterable[str] | None = None) -> int:
    from .cli import main as canonical_main

    print(
        "DeprecationWarning: python -m psannlm.train is deprecated; use python -m psannlm train.",
        file=sys.stderr,
    )
    return canonical_main(
        ["train", *(list(argv) if argv is not None else sys.argv[1:])], _legacy_entry=True
    )


__all__ = ["build_parser", "main", "str2bool"]

if __name__ == "__main__":
    raise SystemExit(main())
