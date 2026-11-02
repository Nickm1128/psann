"""Run a selected native model with explicit time bounds and checkpoint retention.

This local research driver never uploads or publishes. Use --max-seconds to set
an explicit compute budget. A bounded run may produce an unfinished candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import time
from pathlib import Path

import torch
from torch.utils.data import IterableDataset

from psannlm import LMTrainer, PSANNLM, TrainConfig
from psannlm.architectures import build_lm_model
from psannlm.lm.data.tokenizer import Tokenizer
from scripts._explore_lm.data import Blocks, RepeatBlocks
from scripts._explore_lm.config import fingerprint
from scripts._explore_lm.runner import evaluate, seed_all, write_json


class BoundedStream(IterableDataset):
    def __init__(self, source, deadline, skip):
        self.source, self.deadline, self.skip = source, deadline, skip
        self.stop_requested = False

    def __iter__(self):
        iterator = iter(self.source)
        try:
            for index, item in enumerate(iterator):
                if self.stop_requested or time.monotonic() >= self.deadline:
                    return
                if index >= self.skip:
                    yield item
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()


class LocalParquetTokens(IterableDataset):
    """Use Arrow's native local-file reader, a bounded shuffle, and fixed IDs."""

    def __init__(self, data, tokenizer, length, seed, excluded):
        self.data, self.tokenizer, self.length = data, tokenizer, length
        self.seed, self.excluded = seed, excluded

    def _documents(self):
        import pyarrow.parquet as parquet
        from huggingface_hub import hf_hub_download

        generator = random.Random(self.seed)
        files = list(self.data["files"])
        bootstrap = self.data.get("bootstrap_shard")
        if bootstrap:
            files.remove(bootstrap)
        generator.shuffle(files)
        if bootstrap:
            files.insert(0, bootstrap)
        buffer = []
        for filename in files:
            path = hf_hub_download(
                repo_id=self.data["dataset"],
                repo_type="dataset",
                revision=self.data["revision"],
                filename=filename,
            )
            reader = parquet.ParquetFile(path)
            try:
                for batch in reader.iter_batches(
                    batch_size=1024, columns=["text"], use_threads=False
                ):
                    for text in batch.column(0).to_pylist():
                        text = str(text).strip()
                        if not text or hashlib.sha256(text.encode()).hexdigest() in self.excluded:
                            continue
                        if len(buffer) < self.data["shuffle_buffer"]:
                            buffer.append(text)
                        else:
                            index = generator.randrange(len(buffer))
                            chosen, buffer[index] = buffer[index], text
                            yield chosen
            finally:
                reader.close()
        generator.shuffle(buffer)
        yield from buffer

    def __iter__(self):
        documents, tokens = self._documents(), []
        try:
            for text in documents:
                tokens.extend(self.tokenizer.encode(text, add_specials=True))
                while len(tokens) >= self.length + 1:
                    chunk = tokens[: self.length + 1]
                    del tokens[: self.length]
                    yield {
                        "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                        "labels": torch.tensor(chunk[1:], dtype=torch.long),
                    }
        finally:
            documents.close()


class RetainedTrainer(LMTrainer):
    """Retain the last three periodic checkpoints and the native final artifact."""

    def _save_checkpoint(self, *args, **kwargs):
        super()._save_checkpoint(*args, **kwargs)
        directory = Path(self.cfg.checkpoint_dir)
        periodic = sorted(directory.glob("ckpt_step[0-9][0-9][0-9][0-9][0-9][0-9].pt"))
        for path in periodic[:-3]:
            path.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-seconds", required=True, type=float)
    parser.add_argument("--resume", help="Explicit native trainer checkpoint to resume")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if args.max_seconds <= 0 or not math.isfinite(args.max_seconds):
        raise ValueError("--max-seconds must be finite and positive.")
    started = time.monotonic()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This driver supports a single local process.")
    recipe = json.loads(Path(args.recipe).read_text())
    if recipe.get("schema") != "psannlm.selected-training.v1":
        raise ValueError("Unsupported selected-training recipe.")
    output = Path(args.out)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Choose a new output directory or explicitly resume its checkpoint.")
    if args.resume:
        checkpoint = Path(args.resume).resolve()
        if checkpoint.parent != (output / "checkpoints").resolve():
            raise ValueError("Resume checkpoint must belong to this output directory.")
        previous = json.loads((output / "recipe.json").read_text())
        if previous != recipe:
            raise ValueError(
                "Recipe differs from the original run; preserve its training schedule."
            )
    seed_all(int(recipe["seed"]))
    model = build_lm_model(recipe["model"]).model
    parameters = sum(value.numel() for value in model.parameters())
    if parameters != int(recipe["parameters"]):
        raise ValueError("Recipe parameter count does not match the model.")
    cache = torch.load(recipe["token_cache"], map_location="cpu", weights_only=True)
    if fingerprint(cache["tokenizer"]) != recipe["tokenizer_sha256"]:
        raise ValueError("Tokenizer state differs from the selected recipe.")
    tokenizer = Tokenizer.from_state(cache["tokenizer"])
    if tokenizer.vocab_size != model.lm_config.vocab_size:
        raise ValueError("Tokenizer vocabulary conflicts with the model.")
    length = int(recipe["sequence_length"])
    train = TrainConfig(**dict(recipe["train"], checkpoint_dir=str(output / "checkpoints")))
    if (
        train.epochs != 1
        or train.steps_per_epoch is None
        or train.ddp != "off"
        or train.fsdp != "off"
    ):
        raise ValueError("Use one epoch, explicit optimizer steps, and single-device training.")
    tokens_per_step = max(1, train.batch_tokens // length) * length * train.grad_accum_steps
    resume_step = 0
    if args.resume:
        metadata = torch.load(args.resume, map_location="cpu", weights_only=True, mmap=True)
        resume_step = int(metadata["state"]["step"])
        del metadata
    data = recipe["data"]
    if data["kind"] == "cache-smoke":
        if cache["metadata"]["sequence_length"] != length:
            raise ValueError("Smoke cache sequence length differs from the recipe.")
        source = RepeatBlocks(Blocks(cache["train"]), int(recipe["seed"]))
    elif data["kind"] == "local-parquet":
        excluded_bytes = Path(data["excluded_hashes"]).read_bytes()
        if hashlib.sha256(excluded_bytes).hexdigest() != data["excluded_hashes_sha256"]:
            raise ValueError("Excluded document hashes differ from the recipe.")
        excluded = set(json.loads(excluded_bytes))
        source = LocalParquetTokens(data, tokenizer, length, int(recipe["seed"]), excluded)
    else:
        raise ValueError("Unsupported training data kind.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "recipe.json", recipe)
    if not args.resume:
        model.to(args.device)
        write_json(
            output / "initial_metrics.json",
            evaluate(
                model,
                Blocks(cache["validation"]),
                train.batch_tokens,
                int(recipe["validation_tokens"]),
            ),
        )
    deadline = started + args.max_seconds
    dataset = BoundedStream(
        source,
        deadline,
        resume_step * max(1, train.batch_tokens // length) * train.grad_accum_steps,
    )
    trainer = RetainedTrainer(train)

    def observe(event):
        if not math.isfinite(float(event["loss"])):
            raise ValueError("Training loss is nonfinite; inspect the last good checkpoint.")
        measured = dict(event, elapsed_s=args.max_seconds - (deadline - time.monotonic()))
        if event["step"] % int(recipe["validation_interval"]) == 0:
            measured.update(
                evaluate(
                    model,
                    Blocks(cache["validation"]),
                    train.batch_tokens,
                    int(recipe["validation_tokens"]),
                )
            )
        with (output / "steps.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(measured) + "\n")

    stopped_by = []

    def stop_after_batch(signum, frame):
        dataset.stop_requested = True
        stopped_by.append(signal.Signals(signum).name)

    previous_handlers = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, stop_after_batch)
    try:
        trainer.train(
            model,
            dataset,
            max_length=length,
            device=args.device,
            resume_checkpoint=args.resume,
            on_step=observe,
        )
        lm = PSANNLM(config=model.lm_config, device=args.device)
        lm._model = model
        lm.attach_tokenizer(tokenizer)
        lm.save(output / "model.pt")
        status = "complete" if trainer.state.step >= train.steps_per_epoch else "incomplete"
        write_json(
            output / "completion.json",
            {
                "status": status,
                "step": trainer.state.step,
                "trained_tokens": trainer.state.step * tokens_per_step,
                "planned_tokens": train.steps_per_epoch * tokens_per_step,
                "stopped_by": stopped_by,
                "wall_time_limit_reached": time.monotonic() >= deadline,
                "note": "Candidate weights require held-out quality evaluation and instruction tuning before distribution.",
            },
        )
    finally:
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
