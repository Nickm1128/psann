# ruff: noqa: F403,F405
from __future__ import annotations

from .config import (
    _coerce_betas,
    _coerce_ddp,
    _deep_update,
    _default_config,
    _now_utc_tag,
    _system_info,
    _tee_run_logs,
    _write_json,
)
from .eval import _eval_model, _run_lm_eval
from .shared import *
from .sweep import _expand_sweep_configs, _parse_bases
from .tokenizer import _build_stream_dataset, _ensure_tokenizer


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark PSANN-LM bases quickly on WikiText-103")
    ap.add_argument("--config", type=str, default=None, help="Optional YAML config path")
    ap.add_argument("--out", type=str, default=None, help="Override output directory")
    ap.add_argument("--run-name", type=str, default=None, help="Override run name suffix")
    ap.add_argument("--bases", type=str, default=None, help="Comma-separated base list")
    ap.add_argument("--seeds", type=str, default=None, help="Comma-separated seed list")
    ap.add_argument("--max-steps", type=int, default=None, help="Override training steps")
    ap.add_argument(
        "--tokens-target",
        type=int,
        default=None,
        help="Approximate token budget per run (overrides max-steps)",
    )
    ap.add_argument(
        "--with-lm-eval",
        action="store_true",
        help="Run lm-eval (opt-in; expects lm_eval installed)",
    )
    ap.add_argument("--lm-eval-tasks", type=str, default=None)
    ap.add_argument("--lm-eval-limit", type=int, default=None)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned sweep matrix and exit without running training.",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs that already have a metrics.json with status=ok.",
    )
    ap.add_argument(
        "--save-run-logs",
        action="store_true",
        help="Tee stdout/stderr to run_dir/stdout.log for each run.",
    )
    ap.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile for training runs (single GPU only; skipped under DDP/FSDP).",
    )
    ap.add_argument(
        "--torch-compile-mode",
        type=str,
        default=None,
        choices=["default", "reduce-overhead", "max-autotune"],
        help="Optional torch.compile mode override.",
    )
    args = ap.parse_args()

    cfg = _default_config()
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise SystemExit(f"Config not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        _deep_update(cfg, loaded)

    if args.out:
        cfg["bench"]["out_dir"] = args.out
    if args.run_name:
        cfg["bench"]["run_name"] = args.run_name
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = int(args.max_steps)
    if args.tokens_target is not None:
        cfg["train"]["tokens_target"] = int(args.tokens_target)
    if args.with_lm_eval:
        cfg["bench"]["with_lm_eval"] = True
    if args.save_run_logs:
        cfg["bench"]["save_run_logs"] = True
    if args.lm_eval_tasks:
        cfg["bench"]["lm_eval_tasks"] = [
            t.strip() for t in args.lm_eval_tasks.split(",") if t.strip()
        ]
    if args.lm_eval_limit is not None:
        cfg["bench"]["lm_eval_limit"] = int(args.lm_eval_limit)
    if args.seeds:
        cfg["bench"]["seeds"] = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.torch_compile:
        cfg["train"]["torch_compile"] = True
    if args.torch_compile_mode:
        cfg["train"]["torch_compile_mode"] = str(args.torch_compile_mode)

    bases = _parse_bases(args.bases, cfg)
    bench_cfg = cfg.get("bench", {})
    seeds = bench_cfg.get("seeds") or [1337]
    if not isinstance(seeds, list):
        seeds = [int(seeds)]

    out_root = Path(str(bench_cfg.get("out_dir", "reports/benchmarks"))).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    tag = _now_utc_tag()
    run_name = str(bench_cfg.get("run_name", "base_shootout"))
    outdir = out_root / f"{tag}_{run_name}"
    outdir.mkdir(parents=True, exist_ok=True)

    (outdir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    _write_json(outdir / "system.json", _system_info())

    sweeps = _expand_sweep_configs(cfg)
    _write_json(
        outdir / "sweep_plan.json",
        {
            "count": len(sweeps),
            "bases": bases,
            "seeds": seeds,
            "sweeps": [
                {"id": int(s["id"]), "slug": str(s["slug"]), "overrides": s.get("overrides", {})}
                for s in sweeps
            ],
        },
    )

    if args.dry_run:
        print(f"[dry-run] outdir={outdir}")
        for s in sweeps:
            print(f"  sweep={int(s['id']):03d} slug={s['slug']} overrides={s.get('overrides', {})}")
            for base in bases:
                for seed in seeds:
                    print(f"    - base={base} seed={seed}")
        return 0

    results: List[Dict[str, Any]] = []
    tokenizers_by_sweep: Dict[str, Any] = {}
    runs_dir = outdir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    sweep_count = len(sweeps)
    for sweep in sweeps:
        sweep_id = int(sweep.get("id", 0))
        sweep_slug = str(sweep.get("slug", "default"))
        overrides = sweep.get("overrides", {}) or {}
        run_cfg: Dict[str, Any] = sweep["cfg"]

        bench_cfg_run = run_cfg.get("bench", {})
        data_cfg = run_cfg.get("data", {})
        train_cfg = run_cfg.get("train", {})
        eval_cfg = run_cfg.get("eval", {})
        distill_cfg = run_cfg.get("distill", {}) or {}

        sweep_root = runs_dir
        if sweep_count > 1:
            sweep_root = runs_dir / f"sweep{sweep_id:03d}_{sweep_slug}"
            sweep_root.mkdir(parents=True, exist_ok=True)
            (sweep_root / "config_resolved.yaml").write_text(
                yaml.safe_dump(run_cfg, sort_keys=False), encoding="utf-8"
            )

        tok_seed = int(seeds[0]) if seeds else 1337
        tokenizer, tokenizer_meta = _ensure_tokenizer(run_cfg, outdir=outdir, seed=tok_seed)
        tokenizers_by_sweep[str(sweep_id)] = tokenizer_meta
        tok_meta_path = (
            (outdir / "tokenizer_meta.json")
            if sweep_count == 1
            else (sweep_root / "tokenizer_meta.json")
        )
        if sweep_count > 1 or not tok_meta_path.exists():
            _write_json(tok_meta_path, tokenizer_meta)

        distill_enabled = bool(distill_cfg.get("enabled", False))
        teacher_base = str(distill_cfg.get("teacher_base", "transformer"))
        teacher_ckpt_override = distill_cfg.get("teacher_ckpt")
        teacher_max_steps_override = int(distill_cfg.get("teacher_max_steps", 0) or 0)
        distill_alpha = float(distill_cfg.get("alpha", 0.5))
        distill_temperature = float(distill_cfg.get("temperature", 2.0))

        for seed in seeds:
            pending: List[tuple[str, Path, Path]] = []
            for base in bases:
                run_dir = sweep_root / f"{base}_seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = run_dir / "metrics.json"
                if args.skip_existing and metrics_path.exists():
                    try:
                        cached = json.loads(metrics_path.read_text(encoding="utf-8"))
                        if isinstance(cached, dict) and cached.get("status") == "ok":
                            results.append(cached)
                            print(
                                f"[bench] skip existing base={base} seed={seed} sweep={sweep_id:03d}"
                            )
                            continue
                    except Exception:
                        pass
                pending.append((base, run_dir, metrics_path))

            if not pending:
                continue

            teacher_model: Optional[torch.nn.Module] = None
            teacher_ckpt_path: Optional[Path] = None
            teacher_run_dir: Optional[Path] = None

            if distill_enabled:
                if teacher_ckpt_override:
                    teacher_ckpt_path = Path(str(teacher_ckpt_override)).expanduser()
                    if not teacher_ckpt_path.is_absolute():
                        teacher_ckpt_path = (ROOT / teacher_ckpt_path).resolve()
                    if not teacher_ckpt_path.exists():
                        raise RuntimeError(f"Teacher checkpoint not found: {teacher_ckpt_path}")
                else:
                    teacher_run_dir = sweep_root / f"teacher_{teacher_base}_seed{seed}"
                    teacher_run_dir.mkdir(parents=True, exist_ok=True)
                    teacher_ckpt_path = teacher_run_dir / "final_model.pt"

                    if not teacher_ckpt_path.exists():
                        save_logs = bool(bench_cfg_run.get("save_run_logs", False))
                        teacher_log_path = teacher_run_dir / "stdout.log"

                        print(
                            f"[bench] teacher={teacher_base} seed={seed} sweep={sweep_id:03d} -> {teacher_run_dir}"
                        )
                        torch.manual_seed(int(seed))
                        random.seed(int(seed))
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(int(seed))

                        with _tee_run_logs(teacher_log_path, enabled=save_logs):
                            if save_logs:
                                print(
                                    f"[bench] logging stdout/stderr -> {teacher_log_path}",
                                    flush=True,
                                )
                            vocab_size = int(tokenizer.vocab_size)
                            factory = get_base(teacher_base)
                            sine_cfg = train_cfg.get("sine_params", {}) or {}
                            geosparse_kwargs: Dict[str, Any] = {}
                            if str(teacher_base).lower() == "geosparse":
                                for key in (
                                    "geosparse_shape",
                                    "geosparse_depth",
                                    "geosparse_k",
                                    "geosparse_pattern",
                                    "geosparse_radius",
                                    "geosparse_offsets",
                                    "geosparse_wrap_mode",
                                    "geosparse_activation",
                                    "geosparse_activation_types",
                                    "geosparse_activation_ratios",
                                    "geosparse_activation_ratio_sum_tol",
                                    "geosparse_activation_layout",
                                    "geosparse_norm",
                                    "geosparse_drop_path_max",
                                    "geosparse_residual_alpha_init",
                                    "geosparse_bias",
                                    "geosparse_compute_mode",
                                    "geosparse_seed",
                                    "geosparse_chunk_size",
                                ):
                                    if key in train_cfg and train_cfg.get(key) is not None:
                                        geosparse_kwargs[key] = train_cfg.get(key)
                            model = factory(
                                vocab_size=vocab_size,
                                d_model=int(train_cfg.get("d_model", 256)),
                                n_layers=int(train_cfg.get("n_layers", 4)),
                                n_heads=int(train_cfg.get("n_heads", 4)),
                                d_mlp=int(train_cfg.get("d_mlp", 1024)),
                                dropout=float(train_cfg.get("dropout", 0.0)),
                                positional_encoding=str(
                                    train_cfg.get("positional_encoding", "rope")
                                ),
                                mlp_activation=str(train_cfg.get("mlp_activation", "sine")),
                                sine=SineConfig(
                                    amp_init=float(sine_cfg.get("amp_init", 1.0)),
                                    amp_init_std=float(sine_cfg.get("amp_init_std", 0.0)),
                                    freq_init=float(sine_cfg.get("freq_init", 1.0)),
                                    freq_init_std=float(sine_cfg.get("freq_init_std", 0.0)),
                                    damp_init=float(sine_cfg.get("damp_init", 0.01)),
                                    damp_init_std=float(sine_cfg.get("damp_init_std", 0.0)),
                                    trainable=bool(sine_cfg.get("trainable", True)),
                                ),
                                attn_impl=str(train_cfg.get("attn_impl", "auto")),
                                **geosparse_kwargs,
                            )

                            train_ds = _build_stream_dataset(
                                data_cfg,
                                tokenizer,
                                split=str(data_cfg.get("train_split", "train")),
                                shuffle=bool(data_cfg.get("shuffle", True)),
                                seed=int(seed),
                            )

                            batch_tokens = int(train_cfg.get("batch_tokens", 32768))
                            grad_accum = int(train_cfg.get("grad_accum_steps", 1))
                            seq_len = int(data_cfg.get("max_length", 512))
                            batch_size = max(1, batch_tokens // seq_len)
                            tokens_per_step = batch_size * seq_len * grad_accum
                            tokens_target = int(train_cfg.get("tokens_target", 0) or 0)
                            max_steps = int(train_cfg.get("max_steps", 300))
                            if tokens_target > 0:
                                max_steps = max(1, tokens_target // max(1, tokens_per_step))
                            teacher_steps = (
                                teacher_max_steps_override
                                if teacher_max_steps_override > 0
                                else max_steps
                            )

                            tcfg = TrainConfig(
                                epochs=1,
                                batch_tokens=int(batch_tokens),
                                lr=float(train_cfg.get("lr", 2e-4)),
                                warmup_steps=int(train_cfg.get("warmup_steps", 200)),
                                weight_decay=float(train_cfg.get("weight_decay", 0.01)),
                                amp=str(train_cfg.get("amp", "bf16")),
                                optimizer=str(train_cfg.get("optimizer", "adamw")),
                                betas=_coerce_betas(train_cfg.get("betas")),
                                eps=float(train_cfg.get("eps", 1e-8)),
                                label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
                                grad_clip=float(train_cfg.get("grad_clip", 1.0)),
                                grad_checkpoint=bool(train_cfg.get("grad_checkpoint", False)),
                                log_gpu_mem=bool(train_cfg.get("log_gpu_mem", False)),
                                torch_compile=bool(train_cfg.get("torch_compile", False)),
                                torch_compile_mode=str(
                                    train_cfg.get("torch_compile_mode", "default")
                                ),
                                torch_compile_fullgraph=bool(
                                    train_cfg.get("torch_compile_fullgraph", False)
                                ),
                                torch_compile_dynamic=bool(
                                    train_cfg.get("torch_compile_dynamic", False)
                                ),
                                grad_accum_steps=int(grad_accum),
                                ddp=_coerce_ddp(train_cfg.get("ddp"), default="off"),
                                fsdp="off",
                                steps_per_epoch=int(teacher_steps),
                                checkpoint_dir=str(teacher_run_dir / "checkpoints"),
                                log_interval_steps=int(train_cfg.get("log_interval_steps", 50)),
                                save_interval_steps=int(train_cfg.get("save_interval_steps", 500)),
                                dataloader_num_workers=0,
                                dataloader_prefetch_factor=2,
                                dataloader_persistent_workers=False,
                                distill_alpha=0.0,
                                distill_temperature=1.0,
                            )
                            trainer = Trainer(tcfg)
                            trainer.train(model, train_ds, max_length=int(seq_len))

                            teacher_lm = psannLM(
                                base=teacher_base,
                                d_model=int(train_cfg.get("d_model", 256)),
                                n_layers=int(train_cfg.get("n_layers", 4)),
                                n_heads=int(train_cfg.get("n_heads", 4)),
                                d_mlp=int(train_cfg.get("d_mlp", 1024)),
                                vocab_size=vocab_size,
                                positional_encoding=str(
                                    train_cfg.get("positional_encoding", "rope")
                                ),
                                sine_params=sine_cfg,
                                dropout=float(train_cfg.get("dropout", 0.0)),
                                mlp_activation=str(train_cfg.get("mlp_activation", "sine")),
                                attn_impl=str(train_cfg.get("attn_impl", "auto")),
                                **geosparse_kwargs,
                            )
                            teacher_lm._model = model
                            teacher_lm.save(str(teacher_ckpt_path))

                            teacher_model = model

                if teacher_model is None and teacher_ckpt_path is not None:
                    teacher_lm = psannLM.load(str(teacher_ckpt_path))
                    if teacher_lm._model is None:
                        teacher_lm._ensure_model(int(tokenizer.vocab_size))
                    teacher_model = teacher_lm._model

                if teacher_model is None:
                    raise RuntimeError("Distillation enabled but teacher_model is None")

                try:
                    teacher_model.eval()
                    for p in teacher_model.parameters():
                        p.requires_grad_(False)
                except Exception:
                    pass

            for base, run_dir, metrics_path in pending:
                print(f"[bench] base={base} seed={seed} sweep={sweep_id:03d} -> {run_dir}")
                torch.manual_seed(int(seed))
                random.seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))

                record: Dict[str, Any] = {
                    "base": base,
                    "seed": int(seed),
                    "sweep_id": sweep_id,
                    "sweep_slug": sweep_slug,
                    "sweep_overrides": overrides,
                    "status": "ok",
                }

                save_logs = bool(bench_cfg_run.get("save_run_logs", False))
                run_log_path = run_dir / "stdout.log"
                record["run_log_path"] = str(run_log_path) if save_logs else None

                with _tee_run_logs(run_log_path, enabled=save_logs):
                    if save_logs:
                        print(f"[bench] logging stdout/stderr -> {run_log_path}", flush=True)
                        print(
                            f"[bench] sweep={sweep_id:03d} slug={sweep_slug} base={base} seed={seed}",
                            flush=True,
                        )

                    try:
                        vocab_size = int(tokenizer.vocab_size)
                        factory = get_base(base)
                        sine_cfg = train_cfg.get("sine_params", {}) or {}
                        geosparse_kwargs: Dict[str, Any] = {}
                        if str(base).lower() == "geosparse":
                            for key in (
                                "geosparse_shape",
                                "geosparse_depth",
                                "geosparse_k",
                                "geosparse_pattern",
                                "geosparse_radius",
                                "geosparse_offsets",
                                "geosparse_wrap_mode",
                                "geosparse_activation",
                                "geosparse_activation_types",
                                "geosparse_activation_ratios",
                                "geosparse_activation_ratio_sum_tol",
                                "geosparse_activation_layout",
                                "geosparse_norm",
                                "geosparse_drop_path_max",
                                "geosparse_residual_alpha_init",
                                "geosparse_bias",
                                "geosparse_compute_mode",
                                "geosparse_seed",
                                "geosparse_chunk_size",
                            ):
                                if key in train_cfg and train_cfg.get(key) is not None:
                                    geosparse_kwargs[key] = train_cfg.get(key)
                        model = factory(
                            vocab_size=vocab_size,
                            d_model=int(train_cfg.get("d_model", 256)),
                            n_layers=int(train_cfg.get("n_layers", 4)),
                            n_heads=int(train_cfg.get("n_heads", 4)),
                            d_mlp=int(train_cfg.get("d_mlp", 1024)),
                            dropout=float(train_cfg.get("dropout", 0.0)),
                            positional_encoding=str(train_cfg.get("positional_encoding", "rope")),
                            mlp_activation=str(train_cfg.get("mlp_activation", "sine")),
                            sine=SineConfig(
                                amp_init=float(sine_cfg.get("amp_init", 1.0)),
                                amp_init_std=float(sine_cfg.get("amp_init_std", 0.0)),
                                freq_init=float(sine_cfg.get("freq_init", 1.0)),
                                freq_init_std=float(sine_cfg.get("freq_init_std", 0.0)),
                                damp_init=float(sine_cfg.get("damp_init", 0.01)),
                                damp_init_std=float(sine_cfg.get("damp_init_std", 0.0)),
                                trainable=bool(sine_cfg.get("trainable", True)),
                            ),
                            attn_impl=str(train_cfg.get("attn_impl", "auto")),
                            **geosparse_kwargs,
                        )
                        param_count = sum(p.numel() for p in model.parameters())
                        record["param_count"] = int(param_count)

                        train_ds = _build_stream_dataset(
                            data_cfg,
                            tokenizer,
                            split=str(data_cfg.get("train_split", "train")),
                            shuffle=bool(data_cfg.get("shuffle", True)),
                            seed=int(seed),
                        )
                        val_ds = _build_stream_dataset(
                            data_cfg,
                            tokenizer,
                            split=str(data_cfg.get("val_split", "validation")),
                            shuffle=False,
                            seed=int(seed),
                        )

                        batch_tokens = int(train_cfg.get("batch_tokens", 32768))
                        grad_accum = int(train_cfg.get("grad_accum_steps", 1))
                        seq_len = int(data_cfg.get("max_length", 512))
                        batch_size = max(1, batch_tokens // seq_len)
                        tokens_per_step = batch_size * seq_len * grad_accum
                        tokens_target = int(train_cfg.get("tokens_target", 0) or 0)
                        max_steps = int(train_cfg.get("max_steps", 300))
                        if tokens_target > 0:
                            max_steps = max(1, tokens_target // max(1, tokens_per_step))

                        record["train_config"] = {
                            "batch_tokens": int(batch_tokens),
                            "grad_accum_steps": int(grad_accum),
                            "lr": float(train_cfg.get("lr", 2e-4)),
                            "warmup_steps": int(train_cfg.get("warmup_steps", 200)),
                            "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
                            "optimizer": str(train_cfg.get("optimizer", "adamw")),
                            "betas": list(_coerce_betas(train_cfg.get("betas"))),
                            "eps": float(train_cfg.get("eps", 1e-8)),
                            "label_smoothing": float(train_cfg.get("label_smoothing", 0.0)),
                            "grad_clip": float(train_cfg.get("grad_clip", 1.0)),
                            "amp": str(train_cfg.get("amp", "bf16")),
                            "grad_checkpoint": bool(train_cfg.get("grad_checkpoint", False)),
                            "log_gpu_mem": bool(train_cfg.get("log_gpu_mem", False)),
                            "ddp": _coerce_ddp(train_cfg.get("ddp"), default="off"),
                            "torch_compile": bool(train_cfg.get("torch_compile", False)),
                            "torch_compile_mode": str(
                                train_cfg.get("torch_compile_mode", "default")
                            ),
                            "torch_compile_fullgraph": bool(
                                train_cfg.get("torch_compile_fullgraph", False)
                            ),
                            "torch_compile_dynamic": bool(
                                train_cfg.get("torch_compile_dynamic", False)
                            ),
                            "log_interval_steps": int(train_cfg.get("log_interval_steps", 50)),
                            "save_interval_steps": int(train_cfg.get("save_interval_steps", 500)),
                            "max_steps": int(max_steps),
                            "tokens_target": int(tokens_target),
                            "tokens_per_step": int(tokens_per_step),
                            "distill_enabled": bool(distill_enabled),
                            "distill_alpha": float(distill_alpha) if distill_enabled else 0.0,
                            "distill_temperature": (
                                float(distill_temperature) if distill_enabled else 1.0
                            ),
                            "distill_teacher_base": teacher_base if distill_enabled else None,
                            "distill_teacher_ckpt": (
                                str(teacher_ckpt_path) if distill_enabled else None
                            ),
                        }
                        record["data_config"] = {
                            "dataset": str(data_cfg.get("dataset")),
                            "name": data_cfg.get("name"),
                            "data_files": data_cfg.get("data_files"),
                            "train_split": str(data_cfg.get("train_split", "train")),
                            "val_split": str(data_cfg.get("val_split", "validation")),
                            "text_key": str(data_cfg.get("text_key", "text")),
                            "max_length": int(seq_len),
                            "shuffle": bool(data_cfg.get("shuffle", True)),
                            "shuffle_buffer": int(data_cfg.get("shuffle_buffer", 10000)),
                        }

                        tcfg = TrainConfig(
                            epochs=1,
                            batch_tokens=int(batch_tokens),
                            lr=float(train_cfg.get("lr", 2e-4)),
                            warmup_steps=int(train_cfg.get("warmup_steps", 200)),
                            weight_decay=float(train_cfg.get("weight_decay", 0.01)),
                            amp=str(train_cfg.get("amp", "bf16")),
                            optimizer=str(train_cfg.get("optimizer", "adamw")),
                            betas=_coerce_betas(train_cfg.get("betas")),
                            eps=float(train_cfg.get("eps", 1e-8)),
                            label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
                            grad_clip=float(train_cfg.get("grad_clip", 1.0)),
                            grad_checkpoint=bool(train_cfg.get("grad_checkpoint", False)),
                            log_gpu_mem=bool(train_cfg.get("log_gpu_mem", False)),
                            torch_compile=bool(train_cfg.get("torch_compile", False)),
                            torch_compile_mode=str(train_cfg.get("torch_compile_mode", "default")),
                            torch_compile_fullgraph=bool(
                                train_cfg.get("torch_compile_fullgraph", False)
                            ),
                            torch_compile_dynamic=bool(
                                train_cfg.get("torch_compile_dynamic", False)
                            ),
                            grad_accum_steps=int(grad_accum),
                            ddp=_coerce_ddp(train_cfg.get("ddp"), default="off"),
                            fsdp="off",
                            steps_per_epoch=int(max_steps),
                            checkpoint_dir=str(run_dir / "checkpoints"),
                            log_interval_steps=int(train_cfg.get("log_interval_steps", 50)),
                            save_interval_steps=int(train_cfg.get("save_interval_steps", 500)),
                            dataloader_num_workers=0,
                            dataloader_prefetch_factor=2,
                            dataloader_persistent_workers=False,
                            distill_alpha=float(distill_alpha) if distill_enabled else 0.0,
                            distill_temperature=(
                                float(distill_temperature) if distill_enabled else 1.0
                            ),
                        )

                        trainer = Trainer(tcfg)
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()
                            torch.cuda.synchronize()
                        t0 = time.time()
                        trainer.train(
                            model,
                            train_ds,
                            max_length=int(seq_len),
                            val_dataset=None,
                            teacher_model=teacher_model if distill_enabled else None,
                        )
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        elapsed = time.time() - t0

                        steps_done = int(getattr(trainer.state, "step", 0))
                        world_size = int(os.environ.get("WORLD_SIZE", "1"))
                        total_tokens = steps_done * tokens_per_step * max(1, world_size)
                        tokens_per_s = (total_tokens / elapsed) if elapsed > 0 else 0.0
                        record.update(
                            {
                                "train_steps": steps_done,
                                "train_tokens": int(total_tokens),
                                "train_elapsed_s": round(elapsed, 4),
                                "train_tokens_per_s": round(tokens_per_s, 2),
                            }
                        )

                        peak_mem = None
                        peak_reserved = None
                        if torch.cuda.is_available():
                            try:
                                peak_mem = torch.cuda.max_memory_allocated() / float(1024**3)
                                peak_reserved = torch.cuda.max_memory_reserved() / float(1024**3)
                            except Exception:
                                peak_mem = None
                                peak_reserved = None
                        record["peak_cuda_mem_gb"] = (
                            None if peak_mem is None else round(peak_mem, 4)
                        )
                        record["peak_cuda_reserved_gb"] = (
                            None if peak_reserved is None else round(peak_reserved, 4)
                        )

                        eval_batch_tokens = int(eval_cfg.get("batch_tokens", batch_tokens))
                        eval_metrics = _eval_model(
                            model,
                            val_ds,
                            max_tokens=int(eval_cfg.get("max_tokens", 200000)),
                            max_batches=int(eval_cfg.get("max_batches", 0)),
                            batch_tokens=int(eval_batch_tokens),
                            seq_len=int(seq_len),
                            amp_mode=str(train_cfg.get("amp", "bf16")).lower(),
                        )
                        record.update(eval_metrics)

                        lm = psannLM(
                            base=base,
                            d_model=int(train_cfg.get("d_model", 256)),
                            n_layers=int(train_cfg.get("n_layers", 4)),
                            n_heads=int(train_cfg.get("n_heads", 4)),
                            d_mlp=int(train_cfg.get("d_mlp", 1024)),
                            vocab_size=vocab_size,
                            positional_encoding=str(train_cfg.get("positional_encoding", "rope")),
                            sine_params=sine_cfg,
                            dropout=float(train_cfg.get("dropout", 0.0)),
                            mlp_activation=str(train_cfg.get("mlp_activation", "sine")),
                            attn_impl=str(train_cfg.get("attn_impl", "auto")),
                            **geosparse_kwargs,
                        )
                        lm._model = model  # reuse trained weights
                        ckpt_path = run_dir / "final_model.pt"
                        lm.save(str(ckpt_path))
                        record["final_model_path"] = str(ckpt_path)

                        if bench_cfg_run.get("with_lm_eval", False):
                            if not tokenizer_meta.get("model_path"):
                                record["lm_eval"] = {
                                    "status": "skipped",
                                    "reason": "tokenizer_model_path missing; lm-eval requires tokenizer parity",
                                }
                            else:
                                lm_eval = _run_lm_eval(
                                    run_dir,
                                    ckpt_path=str(ckpt_path),
                                    tokenizer_meta=tokenizer_meta,
                                    tasks=[str(t) for t in bench_cfg_run.get("lm_eval_tasks", [])],
                                    limit=int(bench_cfg_run.get("lm_eval_limit", 256)),
                                    num_fewshot=int(bench_cfg_run.get("lm_eval_num_fewshot", 0)),
                                    device="cuda" if torch.cuda.is_available() else "cpu",
                                )
                                record["lm_eval"] = lm_eval

                    except Exception as exc:
                        record["status"] = "error"
                        record["error"] = str(exc)
                        print(f"[bench] ERROR: {exc}", file=sys.stderr, flush=True)

                results.append(record)
                _write_json(metrics_path, record)

                try:
                    del model
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if teacher_run_dir is not None:
                try:
                    teacher_metrics = {
                        "status": "ok",
                        "teacher_base": teacher_base,
                        "seed": int(seed),
                        "teacher_ckpt_path": str(teacher_ckpt_path),
                    }
                    _write_json(teacher_run_dir / "metrics.json", teacher_metrics)
                except Exception:
                    pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        "timestamp_utc": tag,
        "run_name": run_name,
        "bases": bases,
        "seeds": seeds,
        "system": _system_info(),
        "sweeps": [
            {"id": int(s["id"]), "slug": str(s["slug"]), "overrides": s.get("overrides", {})}
            for s in sweeps
        ],
        "tokenizers_by_sweep": tokenizers_by_sweep,
        "results": results,
    }
    _write_json(outdir / "summary.json", summary)

    # Write summary.csv
    sweep_keys = sorted({k for s in sweeps for k in (s.get("overrides", {}) or {}).keys()})
    sweep_cols = [k.replace(".", "__") for k in sweep_keys]
    header = [
        "base",
        "seed",
        "sweep_id",
        "sweep_slug",
        *sweep_cols,
        "status",
        "param_count",
        "train_steps",
        "train_tokens",
        "train_tokens_per_s",
        "train_elapsed_s",
        "peak_cuda_mem_gb",
        "val_loss",
        "val_ppl",
        "val_top1_acc",
    ]
    csv_rows = [",".join(header)]
    for row in results:
        sweep_vals = row.get("sweep_overrides", {}) if isinstance(row, dict) else {}
        csv_rows.append(
            ",".join(
                [
                    str(row.get("base", "")),
                    str(row.get("seed", "")),
                    str(row.get("sweep_id", "")),
                    str(row.get("sweep_slug", "")),
                    *[str(sweep_vals.get(k, "")) for k in sweep_keys],
                    str(row.get("status", "")),
                    str(row.get("param_count", "")),
                    str(row.get("train_steps", "")),
                    str(row.get("train_tokens", "")),
                    str(row.get("train_tokens_per_s", "")),
                    str(row.get("train_elapsed_s", "")),
                    str(row.get("peak_cuda_mem_gb", "")),
                    str(row.get("val_loss", "")),
                    str(row.get("val_ppl", "")),
                    str(row.get("val_top1_acc", "")),
                ]
            )
        )
    (outdir / "summary.csv").write_text("\n".join(csv_rows) + "\n", encoding="utf-8")

    # Leaderboard markdown
    ok_rows = [r for r in results if r.get("status") == "ok"]
    ok_rows.sort(key=lambda r: (r.get("val_ppl") is None, r.get("val_ppl", float("inf"))))
    lines = [
        "# Base Estimator Shootout",
        "",
        "| sweep | base | seed | val_ppl | val_loss | val_top1_acc | tokens/s | peak_cuda_mem_gb |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in ok_rows:
        lines.append(
            "| {sweep} | {base} | {seed} | {val_ppl} | {val_loss} | {val_top1_acc} | {train_tokens_per_s} | {peak_cuda_mem_gb} |".format(
                sweep=r.get("sweep_slug", ""),
                base=r.get("base"),
                seed=r.get("seed"),
                val_ppl=r.get("val_ppl"),
                val_loss=r.get("val_loss"),
                val_top1_acc=r.get("val_top1_acc"),
                train_tokens_per_s=r.get("train_tokens_per_s"),
                peak_cuda_mem_gb=r.get("peak_cuda_mem_gb"),
            )
        )
    (outdir / "leaderboard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[bench] Wrote summary -> {outdir / 'summary.json'}")
    print(f"[bench] Leaderboard -> {outdir / 'leaderboard.md'}")
    return 0
