# ruff: noqa: F403,F405
from __future__ import annotations

from .config import RunResult
from .shared import *


def aggregate_results(results: List[RunResult]) -> List[Dict[str, object]]:
    log_progress("Aggregating run results.")
    by_label: Dict[str, List[RunResult]] = collections.defaultdict(list)
    for res in results:
        by_label[res.label].append(res)
    rows: List[Dict[str, object]] = []
    for label, runs in sorted(by_label.items(), key=lambda kv: kv[0]):
        tokens_per_step = runs[0].tokens_per_step if runs else 0
        target_params = runs[0].target_params if runs else 0
        landed_params = statistics.mean(r.landed_params for r in runs)
        avg_tokens_sec = statistics.mean(r.avg_tokens_sec for r in runs)
        std_tokens_sec = statistics.pstdev(r.avg_tokens_sec for r in runs) if len(runs) > 1 else 0.0
        final_loss = statistics.mean(r.final_train_loss for r in runs)
        eval_ppl = (
            statistics.mean(r.eval_ppl for r in runs if r.eval_ppl is not None)
            if any(r.eval_ppl is not None for r in runs)
            else None
        )
        wall_clock = statistics.mean(r.wall_clock_s for r in runs)
        peak_mem = statistics.mean(r.peak_mem_gb for r in runs)
        log_progress(
            f"Aggregate {label}: avg_tokens_sec={avg_tokens_sec:.0f} wall_clock={wall_clock:.1f}s seeds={len(runs)}"
        )
        rows.append(
            {
                "size_label": label,
                "target_params": target_params,
                "landed_params": landed_params,
                "tokens_per_step": tokens_per_step,
                "avg_tokens_sec": avg_tokens_sec,
                "std_tokens_sec": std_tokens_sec,
                "p50_ms": statistics.mean(r.p50_ms for r in runs),
                "p90_ms": statistics.mean(r.p90_ms for r in runs),
                "p99_ms": statistics.mean(r.p99_ms for r in runs),
                "peak_mem_gb": peak_mem,
                "final_train_loss": final_loss,
                "eval_ppl": eval_ppl,
                "wall_clock_s": wall_clock,
                "seeds": [r.seed for r in runs],
                "participation_ratio": (
                    statistics.mean(
                        [r.participation_ratio for r in runs if r.participation_ratio is not None]
                    )
                    if any(r.participation_ratio is not None for r in runs)
                    else None
                ),
            }
        )
    return rows


def save_summary(rows: List[Dict[str, object]], out_dir: Path) -> None:
    csv_path = out_dir / "summary.csv"
    json_path = out_dir / "summary.json"
    if not rows:
        return
    log_progress(f"Saving summary artifacts to {out_dir}")
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def plot_scaling(rows: List[Dict[str, object]], results: List[RunResult], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("[warn] matplotlib not available; skipping plots.", flush=True)
        return
    if not rows:
        return
    log_progress("Generating scaling plots.")
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(rows, key=lambda r: r["target_params"])
    sizes = [r["target_params"] / 1e6 for r in sorted_rows]
    labels = [r["size_label"] for r in sorted_rows]
    throughput = [r["avg_tokens_sec"] for r in sorted_rows]
    wall_clock = [r["wall_clock_s"] for r in sorted_rows]

    plt.figure(figsize=(7, 4))
    plt.plot(sizes, throughput, marker="o")
    plt.xlabel("Parameters (Millions)")
    plt.ylabel("Tokens / second")
    plt.title("Throughput vs Model Size")
    plt.grid(True, alpha=0.2)
    plt.xticks(sizes, labels, rotation=20)
    plt.tight_layout()
    plt.savefig(plot_dir / "throughput_vs_size.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(labels, wall_clock)
    plt.ylabel("Wall-clock (s) to steps")
    plt.title("Wall-clock vs Model Size")
    plt.tight_layout()
    plt.savefig(plot_dir / "wallclock_vs_size.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    for label in labels:
        run = next((r for r in results if r.label == label), None)
        if not run or not run.loss_trace:
            continue
        xs = [step * run.tokens_per_step / 1e6 for step, _ in run.loss_trace]
        ys = [loss for _, loss in run.loss_trace]
        plt.plot(xs, ys, label=label)
    plt.xlabel("Tokens seen (Millions)")
    plt.ylabel("Train loss (xe)")
    plt.title("Loss vs Tokens")
    plt.grid(True, alpha=0.2)
    if any(run.loss_trace for run in results):
        plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "loss_vs_tokens.png", dpi=200)
    plt.close()

    if any(r["participation_ratio"] for r in rows):
        pr_values = [r["participation_ratio"] or 0.0 for r in sorted_rows]
        plt.figure(figsize=(7, 4))
        plt.bar(labels, pr_values)
        plt.ylabel("Participation Ratio")
        plt.title("Jacobian Participation Ratio (post-train)")
        plt.tight_layout()
        plt.savefig(plot_dir / "participation_ratio.png", dpi=200)
        plt.close()
