"""Command-line parsing for the PSANN-LM base benchmark."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PSANN-LM bases quickly on WikiText-103")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    parser.add_argument("--out", type=str, default=None, help="Override output directory")
    parser.add_argument("--run-name", type=str, default=None, help="Override run name suffix")
    parser.add_argument("--bases", type=str, default=None, help="Comma-separated base list")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seed list")
    parser.add_argument("--max-steps", type=int, default=None, help="Override training steps")
    parser.add_argument(
        "--tokens-target",
        type=int,
        default=None,
        help="Approximate token budget per run (overrides max-steps)",
    )
    parser.add_argument(
        "--with-lm-eval",
        action="store_true",
        help="Run lm-eval (opt-in; expects lm_eval installed)",
    )
    parser.add_argument("--lm-eval-tasks", type=str, default=None)
    parser.add_argument("--lm-eval-limit", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned sweep matrix and exit without running training.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs that already have a metrics.json with status=ok.",
    )
    parser.add_argument(
        "--save-run-logs",
        action="store_true",
        help="Tee stdout/stderr to run_dir/stdout.log for each run.",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile for training runs (single GPU only; skipped under DDP/FSDP).",
    )
    parser.add_argument(
        "--torch-compile-mode",
        type=str,
        default=None,
        choices=["default", "reduce-overhead", "max-autotune"],
        help="Optional torch.compile mode override.",
    )
    return parser.parse_args()
