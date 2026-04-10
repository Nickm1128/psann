# ruff: noqa: F403,F405
from __future__ import annotations

from .bench import train_one_size
from .config import log_progress, parse_size_targets, resolve_dataset, seed_everything
from .data import prepare_tokenizer
from .models import land_configs
from .reporting import aggregate_results, plot_scaling, save_summary
from .shared import *


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PSANN-LM throughput bench harness")
    parser.add_argument(
        "--sizes",
        type=str,
        required=True,
        help="Comma-separated target sizes (e.g., 15M,50M,125M).",
    )
    parser.add_argument(
        "--base", type=str, default="waveresnet", help="Model base (waveresnet or respsann)."
    )
    parser.add_argument(
        "--dataset", type=str, default="wikitext-103", help="Dataset alias or HF repo id."
    )
    parser.add_argument(
        "--dataset-hub-id",
        type=str,
        default=None,
        help="Explicit HF hub dataset id (overrides alias).",
    )
    parser.add_argument(
        "--dataset-name", type=str, default=None, help="HF dataset config/name if required."
    )
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--eval-split", type=str, default="validation")
    parser.add_argument("--dataset-cache", type=str, default=None, help="HF cache directory.")
    parser.add_argument(
        "--hf-token", type=str, default=None, help="HF auth token (if dataset gated)."
    )
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name or path.")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--tokens-per-step", type=int, default=32768)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--mlp-activation", type=str, default="sine")
    parser.add_argument("--sine-amp", type=float, default=1.0)
    parser.add_argument("--sine-freq", type=float, default=1.0)
    parser.add_argument("--sine-damp", type=float, default=0.01)
    parser.add_argument("--freeze-sine", action="store_true")
    parser.add_argument(
        "--amp-mode", type=str, default="bf16", choices=["bf16", "fp16", "fp32", "none"]
    )
    parser.add_argument("--bf16", action="store_true", help="Shortcut for --amp-mode bf16")
    parser.add_argument("--fp16", action="store_true", help="Shortcut for --amp-mode fp16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--stream", action="store_true", help="Enable HF streaming datasets.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset between epochs.")
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=8192,
        help="Shuffle buffer size for streaming datasets.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--multi-seed-sizes",
        type=str,
        default="",
        help="Comma-separated size labels to run multiple seeds for.",
    )
    parser.add_argument(
        "--multi-seed-repeats", type=int, default=1, help="Number of seeds for specified sizes."
    )
    parser.add_argument("--save-dir", type=str, default="runs/psannlm_bench")
    parser.add_argument(
        "--max-seconds", type=float, default=None, help="Optional wall-clock cap per run."
    )
    parser.add_argument(
        "--width-choices", type=str, default="384,512,640,768,896,1024,1152,1280,1536,1792,2048"
    )
    parser.add_argument("--min-layers", type=int, default=8)
    parser.add_argument("--max-layers", type=int, default=48)
    parser.add_argument("--layer-step", type=int, default=2)
    parser.add_argument("--max-heads", type=int, default=32)
    parser.add_argument("--max-param-error", type=float, default=0.03)
    parser.add_argument("--wave-kernel-size", type=int, default=3)
    parser.add_argument("--wave-dilation-growth", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--pr-samples", type=int, default=4)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--positional-encoding", type=str, default="rope")
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_progress(f"Arguments parsed: {args}")
    if args.bf16:
        args.amp_mode = "bf16"
    if args.fp16:
        args.amp_mode = "fp16"
    if args.amp_mode == "none":
        args.amp_mode = "fp32"
    width_choices = [int(w.strip()) for w in args.width_choices.split(",") if w.strip()]
    log_progress(f"Width choices: {width_choices}")
    targets = parse_size_targets(args.sizes)
    dataset_id, dataset_name = resolve_dataset(args.dataset, args.dataset_hub_id, args.dataset_name)
    log_progress(
        f"Preparing to load dataset id={dataset_id} name={dataset_name} split={args.train_split}"
    )
    train_ds = load_dataset(
        dataset_id,
        name=dataset_name,
        split=args.train_split,
        streaming=args.stream,
        use_auth_token=args.hf_token,
        cache_dir=args.dataset_cache,
    )
    try:
        val_ds = load_dataset(
            dataset_id,
            name=dataset_name,
            split=args.eval_split,
            streaming=args.stream,
            use_auth_token=args.hf_token,
            cache_dir=args.dataset_cache,
        )
    except Exception:
        print(
            f"[warn] Eval split '{args.eval_split}' unavailable; using train split for eval.",
            flush=True,
        )
        val_ds = None
    text_field = _detect_text_field(train_ds)
    tokenizer = prepare_tokenizer(args.tokenizer, args.seq_len)
    vocab_size = tokenizer.vocab_size
    log_progress(f"Tokenizer vocab size: {vocab_size}")

    train_stream = TextStream(
        train_ds,
        text_field,
        streaming=args.stream,
        shuffle=args.shuffle,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
    )
    val_stream = (
        TextStream(
            val_ds,
            text_field,
            streaming=args.stream,
            shuffle=False,
            seed=args.seed + 1,
            shuffle_buffer=args.shuffle_buffer,
        )
        if val_ds is not None
        else None
    )

    seed_everything(args.seed)

    landings = land_configs(
        targets,
        vocab_size=vocab_size,
        base=args.base,
        width_choices=width_choices,
        layer_min=args.min_layers,
        layer_max=args.max_layers,
        layer_step=args.layer_step,
        dropout=args.dropout,
        positional_encoding=args.positional_encoding,
        wave_interleave=True,
        wave_kernel_size=args.wave_kernel_size,
        wave_dilation_growth=args.wave_dilation_growth,
        max_heads=args.max_heads,
        max_error=args.max_param_error,
    )

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.save_dir) / f"{timestamp}_bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    params_path = out_dir / "params_landing.json"
    params_path.write_text(
        json.dumps([asdict(landing) for landing in landings], indent=2),
        encoding="utf-8",
    )
    log_progress(f"Saved parameter landing to {params_path}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    log_progress(f"Using device: {device}")
    results: List[RunResult] = []
    multi_seed_targets = {
        label.strip().upper() for label in args.multi_seed_sizes.split(",") if label.strip()
    }

    for landing in landings:
        seeds = [args.seed]
        if landing.label.upper() in multi_seed_targets and args.multi_seed_repeats > 1:
            seeds = [args.seed + idx for idx in range(args.multi_seed_repeats)]
        for run_seed in seeds:
            print(f"\n[info] === Running size {landing.label} (seed {run_seed}) ===", flush=True)
            log_progress(f"Dispatching run for {landing.label} seed={run_seed}")
            seed_everything(run_seed)
            run_dir = out_dir / f"{landing.label}_seed{run_seed}"
            run_result = train_one_size(
                landing,
                args=argparse.Namespace(**{**vars(args), "seed": run_seed}),
                tokenizer=tokenizer,
                train_stream=train_stream,
                val_stream=val_stream,
                device=device,
                run_dir=run_dir,
            )
            results.append(run_result)

    summary_rows = aggregate_results(results)
    save_summary(summary_rows, out_dir)
    if not args.no_plots:
        plot_scaling(summary_rows, results, out_dir)

    log_progress("Benchmark runs complete. Emitting summary to stdout.")
    print("\n[done] Benchmark complete. Summary rows:")
    for row in summary_rows:
        print(
            f"- {row['size_label']}: params={row['landed_params']/1e6:.1f}M "
            f"tokens/sec={row['avg_tokens_sec']:.0f} "
            f"p50={row['p50_ms']:.1f}ms wall={row['wall_clock_s']:.1f}s "
            f"loss={row['final_train_loss']:.3f} "
            f"eval_ppl={row['eval_ppl'] if row['eval_ppl'] else 'n/a'}",
            flush=True,
        )
    print(f"\nArtifacts stored under: {out_dir}")
