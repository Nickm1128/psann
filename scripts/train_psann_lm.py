#!/usr/bin/env python
"""Backward-compatible shim for the psannlm.train entrypoint."""

from psannlm.train import main


if __name__ == "__main__":
    raise SystemExit(main())
