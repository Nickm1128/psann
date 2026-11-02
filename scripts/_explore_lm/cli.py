"""Small commands supporting inspect-before-train exploration workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from psannlm.lm.data.tokenizer import Tokenizer
from torch.utils.data import DataLoader

from .config import preset, read_config, trials
from .data import Blocks, load_cache, prepare
from .diagnostics import causality_probe, probe
from .runner import model_for, plan, run, summarize, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Bounded PSANN-LM architecture and training exploration"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Write a smoke-test or FineWeb proxy configuration")
    init.add_argument("--preset", choices=["smoke", "fineweb-scout"], default="smoke")
    init.add_argument("--out", required=True)
    prepare_parser = sub.add_parser("prepare", help="Cache tokens with a document-hash holdout")
    prepare_parser.add_argument("--config", required=True)
    prepare_parser.add_argument("--out", required=True)
    for command in ("plan", "probe", "run"):
        item = sub.add_parser(command)
        item.add_argument("--config", required=True)
        item.add_argument("--cache", required=True)
        item.add_argument("--out", required=True)
        if command in {"plan", "probe"}:
            item.add_argument(
                "--target",
                action="store_true",
                help="Use target-model dimensions for resource inspection",
            )
        if command != "plan":
            item.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
        if command == "probe":
            item.add_argument(
                "--limit", type=int, default=0, help="Limit trial count; zero probes all trials"
            )
        if command == "run":
            item.add_argument("--max-seconds", type=float)
    report = sub.add_parser("report", help="Rebuild leaderboard and shortlist for an existing run")
    report.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.command == "report":
        summarize(args.out)
        return 0
    output = Path(args.out)
    if args.command != "run" and output.exists():
        raise ValueError("Output already exists; choose a new file.")
    if args.command == "init":
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump(preset(args.preset), sort_keys=False), encoding="utf-8")
        return 0
    config = read_config(args.config)
    if args.command == "prepare":
        cache = prepare(config)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, output)
        print(json.dumps(cache["metadata"], indent=2))
        return 0
    cache = load_cache(args.cache, config)
    if args.command == "plan":
        write_json(output, plan(config, cache, target=args.target))
    elif args.command == "probe":
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable in this Python environment.")
        tokenizer = Tokenizer.from_state(cache["tokenizer"])
        candidates = trials(config, tokenizer.vocab_size, target=args.target)
        if args.limit < 0:
            raise ValueError("--limit must be nonnegative.")
        if args.limit:
            candidates = candidates[: args.limit]
        batch = next(iter(DataLoader(Blocks(cache["train"]), batch_size=1)))
        rows = []
        for trial in candidates:
            model = model_for(trial, args.device)
            rows.append(
                {
                    "trial": trial,
                    "causality": causality_probe(model, batch["input_ids"]),
                    "probe": probe(model, batch),
                }
            )
            del model
        write_json(output, rows)
    else:
        results = run(config, cache, output, device=args.device, max_seconds=args.max_seconds)
        summary = json.loads((output / "summary.json").read_text())
        return (
            0
            if results
            and all(row["status"] == "ok" for row in results)
            and summary["not_started"] == 0
            else 2
        )
    return 0
