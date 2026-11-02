"""Use the canonical model registry and trainer for bounded, comparable trials."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
import time
from pathlib import Path

import torch
from psannlm import PSANNLM, LMTrainer, TrainConfig
from psannlm.architectures import build_lm_model
from psannlm.lm.data.tokenizer import Tokenizer
from torch.utils.data import DataLoader

from scripts._bench_lm_bases.config import _system_info

from .config import fingerprint, trials
from .data import Blocks, RepeatBlocks
from .diagnostics import causality_probe, gradient_groups, probe


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # JSON consumers receive null rather than nonstandard NaN/Infinity.
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [clean(v) for v in value]
        return None if isinstance(value, float) and not math.isfinite(value) else value

    path.write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_for(trial, device):
    seed_all(trial["seed"])
    return build_lm_model(trial["model"]).model.to(device)


def plan(config, cache, target=False):
    tokenizer = Tokenizer.from_state(cache["tokenizer"])
    rows = []
    counts = {}
    for trial in trials(config, tokenizer.vocab_size, target=target):
        key = fingerprint(trial["model"])
        if key not in counts:
            model = model_for(trial, "cpu")
            counts[key] = (
                sum(p.numel() for p in model.parameters()),
                sum(p.numel() for p in model.parameters() if p.requires_grad),
            )
            del model
        parameters, trainable = counts[key]
        length = cache["metadata"]["sequence_length"]
        batch_size = max(1, int(trial["train"].get("batch_tokens", length)) // length)
        tokens = (
            batch_size
            * length
            * int(trial["train"].get("grad_accum_steps", 1))
            * int(trial["train"].get("max_steps", 32))
        )
        rows.append(
            {
                **trial,
                "parameters": parameters,
                "trainable_parameters": trainable,
                "fp32_adamw_parameter_state_bytes_lower_bound": parameters * 4 + trainable * 12,
                "planned_tokens": tokens,
                "training_cache_passes": tokens / cache["metadata"]["train_tokens"],
                "within_300m_target_band": (
                    abs(
                        parameters
                        / config.get("exploration", {}).get("target_parameters", 304565264)
                        - 1
                    )
                    <= 0.05
                    if target
                    else None
                ),
            }
        )
    return {
        "purpose": config.get("exploration", {}).get("purpose"),
        "scale": "target" if target else "proxy",
        "cache": cache["metadata"],
        "trials": rows,
        "notes": [
            "Memory estimate excludes activations, attention, allocator/workspace and temporary buffers.",
            "Proxy rankings require larger-scale, multiple-seed confirmation before a 300M training run.",
        ],
    }


def evaluate(model, dataset, batch_tokens, max_tokens):
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    loss, correct, count = 0.0, 0, 0
    if max_tokens <= 0:
        raise ValueError("Evaluation token limit must be positive.")
    size = max(1, batch_tokens // (dataset.blocks.shape[1] - 1))
    try:
        with torch.inference_mode():
            for batch in DataLoader(
                dataset, batch_size=size, generator=torch.Generator().manual_seed(0)
            ):
                ids, labels = batch["input_ids"].to(device), batch["labels"].to(device)
                # Use exactly the requested prefix of validation tokens.
                logits = model(ids).float().flatten(0, 1)
                targets = labels.flatten()
                take = min(targets.numel(), max_tokens - count)
                logits, targets = logits[:take], targets[:take]
                loss += float(torch.nn.functional.cross_entropy(logits, targets, reduction="sum"))
                correct += int((logits.argmax(-1) == targets).sum())
                count += take
                if count >= max_tokens:
                    break
        if count == 0:
            raise ValueError("Evaluation processed no tokens.")
        mean = loss / count
        if not math.isfinite(mean):
            raise ValueError("Validation loss is nonfinite.")
        return {
            "val_loss": mean,
            "val_ppl": math.exp(mean) if mean < 700 else None,
            "val_accuracy": correct / count,
            "val_tokens": count,
        }
    finally:
        model.train(was_training)


class BudgetExceeded(RuntimeError):
    pass


def run(config, cache, output, device="cpu", max_seconds=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Trial output must be absent or empty; choose a new run directory.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA is unavailable in this Python environment; use a CUDA-capable environment."
        )
    if device not in {"cpu", "cuda"}:
        raise ValueError("Exploration currently supports single-device cpu or cuda trials.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "Exploration is a single-process workflow; do not launch it with torchrun."
        )
    length = cache["metadata"]["sequence_length"]
    tokenizer = Tokenizer.from_state(cache["tokenizer"])
    planned = trials(config, tokenizer.vocab_size)
    train, validation = Blocks(cache["train"]), Blocks(cache["validation"])
    settings = config.get("exploration", {})
    interval = int(settings.get("eval_interval", 32))
    eval_tokens = int(settings.get("eval_tokens", 32768))
    budget = float(max_seconds if max_seconds is not None else settings.get("max_seconds", 7200))
    if interval <= 0 or eval_tokens <= 0 or budget <= 0 or not math.isfinite(budget):
        raise ValueError(
            "Evaluation interval, token limit and time budget must be finite and positive."
        )
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "experiment.json",
        {
            "config": config,
            "cache": cache["metadata"],
            "system": _system_info(),
            "fingerprint": fingerprint({"config": config, "cache": cache["metadata"]}),
            "device": device,
            "max_seconds": budget,
            "planned_trials": len(planned),
        },
    )
    results = []
    write_json(output / "results.json", results)
    for trial in planned:
        if time.monotonic() - started >= budget:
            break
        directory = output / trial["id"]
        directory.mkdir()
        write_json(directory / "trial.json", trial)
        model, trainer, lm, observe = None, None, None, None
        record = {
            "id": trial["id"],
            "name": trial["name"],
            "seed": trial["seed"],
            "status": "failed",
        }
        try:
            model = model_for(trial, device)
            record["parameters"] = sum(p.numel() for p in model.parameters())
            batch_tokens = int(trial["train"].get("batch_tokens", 2048))
            batch_size = max(1, batch_tokens // length)
            batch = next(iter(DataLoader(train, batch_size=batch_size)))
            causal = causality_probe(model, batch["input_ids"])
            write_json(directory / "causality_initial.json", causal)
            if not causal["passed"]:
                raise ValueError(
                    "Causality check failed: prefix logits depend on future tokens or context length."
                )
            initial_probe = probe(model, batch)
            write_json(directory / "probe_initial.json", initial_probe)
            if (
                not initial_probe["finite_logits"]
                or not initial_probe["finite_gradients"]
                or not math.isfinite(initial_probe["loss"])
            ):
                raise ValueError("Initial forward/backward probe is nonfinite.")
            initial_eval = evaluate(model, validation, batch_tokens, eval_tokens)
            curve = [{"step": 0, "train_tokens": 0, **initial_eval}]
            processed, callback_seconds = 0, 0.0
            values = dict(trial["train"])
            max_steps = int(values.pop("max_steps", 32))
            if device == "cpu":
                values["amp"] = "fp32"
            if values.get("amp", "bf16") not in {"fp32", "bf16"}:
                raise ValueError(
                    "Exploration supports fp32 or bf16; fp16 probes are not implemented."
                )
            cfg = TrainConfig(
                **dict(
                    values,
                    epochs=1,
                    steps_per_epoch=max_steps,
                    ddp="off",
                    fsdp="off",
                    checkpoint_dir=str(directory / "checkpoints"),
                    dataloader_num_workers=0,
                    log_interval_steps=max_steps,
                    save_interval_steps=max_steps + 1,
                )
            )
            write_json(directory / "train_resolved.json", cfg.__dict__)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            training_started = time.monotonic()

            def observe(
                event,
                *,
                training_started=training_started,
                max_steps=max_steps,
                model=model,
                batch_tokens=batch_tokens,
                curve=curve,
                directory=directory,
            ):
                nonlocal processed, callback_seconds
                callback_started = time.monotonic()
                processed += int(event["tokens"])
                row = dict(
                    event, train_tokens=processed, elapsed_s=time.monotonic() - training_started
                )
                if not math.isfinite(float(event["loss"])):
                    raise ValueError("Training loss became nonfinite.")
                if event["step"] % interval == 0 or event["step"] == max_steps:
                    row.update(evaluate(model, validation, batch_tokens, eval_tokens))
                    row["gradient_groups"] = gradient_groups(model)
                curve.append(row)
                write_json(directory / "curve.json", curve)
                callback_seconds += time.monotonic() - callback_started
                if time.monotonic() - started >= budget:
                    raise BudgetExceeded(
                        "Experiment wall-time budget reached; this trial is incomplete."
                    )

            trainer = LMTrainer(cfg)
            trainer.train(
                model,
                RepeatBlocks(train, trial["seed"]),
                max_length=length,
                device=device,
                on_step=observe,
            )
            elapsed = time.monotonic() - training_started
            record.update(
                status="ok",
                train_steps=trainer.state.step,
                train_tokens=processed,
                train_wall_s=elapsed,
                training_tokens_per_s=processed / max(elapsed, 1e-12),
                callback_s=callback_seconds,
                initial_val_loss=initial_eval["val_loss"],
                **evaluate(model, validation, batch_tokens, eval_tokens),
            )
            record["loss_improvement"] = record["initial_val_loss"] - record["val_loss"]
            record["peak_cuda_allocated_bytes"] = (
                torch.cuda.max_memory_allocated() if device == "cuda" else None
            )
            record["cache_passes"] = processed / cache["metadata"]["train_tokens"]
            causal = causality_probe(model, batch["input_ids"])
            write_json(directory / "causality_final.json", causal)
            if not causal["passed"]:
                raise ValueError(
                    "Final causality check failed; exclude this trial from model selection."
                )
            write_json(directory / "probe_final.json", probe(model, batch))
            lm = PSANNLM(config=model.lm_config, device="cpu")
            lm._model = model
            lm.attach_tokenizer(tokenizer)
            lm.save(str(directory / "model.pt"))
        except Exception as exc:  # noqa: BLE001 - retain a failed trial and continue the sweep.
            record.update(
                status="budget_limited" if isinstance(exc, BudgetExceeded) else "failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            results.append(record)
            write_json(directory / "metrics.json", record)
            write_json(output / "results.json", results)
            print(
                f"[explore] {record['id']} status={record['status']} loss={record.get('val_loss')}",
                flush=True,
            )
            model, trainer, lm, observe = None, None, None, None
            if device == "cuda":
                torch.cuda.empty_cache()
    summarize(output)
    return results


def summarize(output):
    output = Path(output)
    results = json.loads((output / "results.json").read_text())
    good = [
        row
        for row in results
        if row["status"] == "ok"
        and row.get("val_loss") is not None
        and math.isfinite(row["val_loss"])
    ]
    for row in good:
        row["pareto_efficient"] = not any(
            other["val_loss"] <= row["val_loss"]
            and other["train_tokens"] == row["train_tokens"]
            and other["val_tokens"] == row["val_tokens"]
            and other["training_tokens_per_s"] >= row["training_tokens_per_s"]
            and (
                other["val_loss"] < row["val_loss"]
                or other["training_tokens_per_s"] > row["training_tokens_per_s"]
            )
            for other in good
        )
    good.sort(key=lambda row: row["val_loss"])
    fields = [
        "id",
        "status",
        "parameters",
        "train_tokens",
        "val_loss",
        "val_ppl",
        "loss_improvement",
        "training_tokens_per_s",
        "pareto_efficient",
    ]
    with (output / "leaderboard.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(good)
    lines = [
        "# Bounded LM exploration",
        "",
        "Proxy results require larger-scale confirmation across seeds. Synthetic smoke results cannot select a production architecture.",
        "",
        "| Trial | Train tokens | Validation loss | Perplexity | Train tokens/s | Pareto efficient |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in good:
        perplexity = f"{row['val_ppl']:.2f}" if row.get("val_ppl") is not None else "overflow"
        lines.append(
            f"| {row['id']} | {row['train_tokens']} | {row['val_loss']:.4f} | {perplexity} | {row['training_tokens_per_s']:.1f} | {row['pareto_efficient']} |"
        )
    grouped = {}
    for row in good:
        trial = json.loads((output / row["id"] / "trial.json").read_text())
        key = fingerprint(
            {
                "model": trial["model"],
                "train": trial["train"],
                "train_tokens": row["train_tokens"],
                "val_tokens": row["val_tokens"],
            }
        )
        grouped.setdefault(key, []).append(row)
    groups = []
    for key, rows in grouped.items():
        losses = [row["val_loss"] for row in rows]
        groups.append(
            {
                "variant": key,
                "name": rows[0]["name"],
                "trial_ids": [row["id"] for row in rows],
                "seeds": [row["seed"] for row in rows],
                "train_tokens": rows[0]["train_tokens"],
                "val_tokens": rows[0]["val_tokens"],
                "val_loss_mean": statistics.mean(losses),
                "val_loss_std": statistics.stdev(losses) if len(losses) > 1 else None,
                "next_step": (
                    "confirm at larger scale" if len(losses) > 1 else "replicate across seeds"
                ),
            }
        )
    groups.sort(key=lambda row: row["val_loss_mean"])
    experiment = json.loads((output / "experiment.json").read_text())
    planned = experiment.get("planned_trials", len(results))
    summary = {
        "planned": planned,
        "completed": len(good),
        "failed_or_incomplete": len(results) - len(good),
        "not_started": planned - len(results),
        "variants": groups,
    }
    lines.extend(
        [
            "",
            f"Completed: {len(good)}. Failed/incomplete: {len(results) - len(good)}. Not started: {planned - len(results)}.",
            "",
            "Throughput includes trainer setup, callbacks, and checkpoint writing; it is not a steady-state kernel benchmark.",
            "Pareto comparisons are restricted to equal train/evaluation token counts. Shortlist variants aggregate available seeds; a single-seed result is provisional.",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "summary.json", summary)
    write_json(output / "shortlist.json", groups[:3])
    return good
