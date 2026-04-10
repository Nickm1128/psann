# ruff: noqa: F403,F405
from __future__ import annotations

from .config import LandingConfig, RunResult, log_progress
from .data import SequenceBatcher, TextStream
from .models import build_scheduler, tie_embeddings
from .shared import *


class StepLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")

    def log(self, payload: Dict[str, object]) -> None:
        self._fh.write(json.dumps(payload) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class GPUStats:
    def __init__(self) -> None:
        self.enabled = False
        self._handle = None
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.enabled = True
            self.max_gpu_util = 0.0
            self.max_mem_util = 0.0
        except Exception:
            self.enabled = False
            self._pynvml = None
            self.max_gpu_util = 0.0
            self.max_mem_util = 0.0
        log_progress(f"GPUStats init -> enabled={self.enabled}")

    def sample(self) -> None:
        if not self.enabled or self._handle is None:
            return
        try:
            util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self.max_gpu_util = max(self.max_gpu_util, float(util.gpu))
            mem_util = 100.0 * float(mem.used) / float(mem.total)
            self.max_mem_util = max(self.max_mem_util, mem_util)
        except Exception:
            pass

    def close(self) -> None:
        if self.enabled:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass


def compute_step_quantiles(times_ms: List[float]) -> Tuple[float, float, float]:
    if not times_ms:
        return 0.0, 0.0, 0.0
    p50 = statistics.median(times_ms)
    if len(times_ms) == 1:
        return p50, p50, p50
    p90 = statistics.quantiles(times_ms, n=10)[8] if len(times_ms) >= 10 else max(times_ms)
    p99 = statistics.quantiles(times_ms, n=100)[98] if len(times_ms) >= 100 else max(times_ms)
    return p50, p90, p99


def evaluate_model(
    model: nn.Module,
    batcher: SequenceBatcher,
    *,
    device: torch.device,
    vocab_size: int,
    eval_batches: int,
) -> Optional[Dict[str, float]]:
    if eval_batches <= 0:
        return None
    log_progress(f"Starting eval pass -> batches={eval_batches}")
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for _ in range(eval_batches):
            inputs, labels = batcher.next_batch()
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(inputs)
            loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
            total_loss += float(loss.item())
            total_tokens += inputs.numel()
    if total_tokens == 0:
        return None
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    log_progress(f"Eval finished -> loss={avg_loss:.4f} ppl={ppl:.2f}")
    return {"loss": avg_loss, "ppl": ppl}


def snapshot_participation_ratio(
    model: nn.Module,
    batcher: SequenceBatcher,
    *,
    device: torch.device,
    samples: int,
) -> Optional[float]:
    if samples <= 0:
        return None
    log_progress(f"Running participation ratio diagnostic -> samples={samples}")
    collected: List[torch.Tensor] = []
    handle = None

    def _hook(_module, _inp, output):
        collected.append(output.detach().float().cpu())
        return output

    if hasattr(model, "ln_f"):
        handle = model.ln_f.register_forward_hook(_hook)  # type: ignore[attr-defined]
    model.eval()
    with torch.no_grad():
        for _ in range(samples):
            inputs, _ = batcher.next_batch()
            inputs = inputs.to(device, non_blocking=True)
            model(inputs)
    if handle is not None:
        handle.remove()
    if not collected:
        return None
    feats = torch.cat([f.mean(dim=1) for f in collected], dim=0)
    if feats.ndim != 2:
        feats = feats.reshape(feats.size(0), -1)
    pr = participation_ratio(feats)
    log_progress(f"Participation ratio computed -> {pr:.4f}")
    return pr


def get_git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT))
        commit = out.decode().strip()
        log_progress(f"Git commit detected -> {commit}")
        return commit
    except Exception:
        log_progress("Git commit unavailable.")
        return None


def train_one_size(
    landing: LandingConfig,
    *,
    args,
    tokenizer,
    train_stream: TextStream,
    val_stream: Optional[TextStream],
    device: torch.device,
    run_dir: Path,
) -> RunResult:
    log_progress(f"train_one_size -> label={landing.label} seed={args.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_progress(f"Output directory: {run_dir}")
    step_log = StepLogger(run_dir / "step_metrics.jsonl")
    loss_curve_path = run_dir / "loss_curve.csv"

    micro_batch = max(1, args.tokens_per_step // (args.seq_len * args.grad_accum))
    tokens_per_step = micro_batch * args.seq_len * args.grad_accum
    if tokens_per_step != args.tokens_per_step:
        print(
            f"[warn] Requested tokens/step={args.tokens_per_step} but "
            f"achievable value is {tokens_per_step} "
            f"(seq_len={args.seq_len}, grad_accum={args.grad_accum}, micro_batch={micro_batch}).",
            flush=True,
        )
    log_progress(
        f"Train setup -> micro_batch={micro_batch} tokens_per_step={tokens_per_step} grad_accum={args.grad_accum}"
    )

    train_batcher = SequenceBatcher(
        train_stream,
        tokenizer,
        seq_len=args.seq_len,
        micro_batch_size=micro_batch,
    )
    eval_batcher = None
    if val_stream is not None:
        eval_batcher = SequenceBatcher(
            val_stream,
            tokenizer,
            seq_len=args.seq_len,
            micro_batch_size=micro_batch,
        )

    factory = get_base(args.base)
    log_progress(f"Instantiating model base='{args.base}' for {landing.label}")
    sine_cfg = SineConfig(
        amp_init=args.sine_amp,
        freq_init=args.sine_freq,
        damp_init=args.sine_damp,
        trainable=not args.freeze_sine,
    )
    model_kwargs = dict(
        vocab_size=tokenizer.vocab_size,
        d_model=landing.d_model,
        n_layers=landing.n_layers,
        n_heads=landing.n_heads,
        d_mlp=landing.d_mlp,
        dropout=landing.dropout,
        positional_encoding=landing.positional_encoding,
        mlp_activation=args.mlp_activation,
        sine=sine_cfg,
        wave_interleave=landing.wave_interleave,
        wave_kernel_size=landing.wave_kernel_size,
        wave_dilation_growth=landing.wave_dilation_growth,
    )
    model = factory(**model_kwargs)
    tie_embeddings(model)
    if args.grad_checkpoint and hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing(True)  # type: ignore[attr-defined]
        log_progress("Gradient checkpointing enabled on model.")
    model.to(device)
    log_progress(f"Model moved to device {device}.")
    if args.compile:
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
            log_progress("torch.compile succeeded.")
        except Exception as exc:
            print(f"[warn] torch.compile failed ({exc}); continuing without compilation.")
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=1e-8,
        weight_decay=args.weight_decay,
        fused=(device.type == "cuda"),
    )
    scheduler = build_scheduler(optimizer, args.steps, args.warmup_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp_mode == "fp16" and device.type == "cuda"))
    amp_enabled = args.amp_mode in {"bf16", "fp16"} and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.amp_mode == "bf16" else torch.float16
    criterion = nn.CrossEntropyLoss()
    log_progress(
        f"Optimization ready -> lr={args.lr} amp={args.amp_mode} grad_clip={args.grad_clip} tf32={args.tf32}"
    )

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32

    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None

    tokens_total = 0
    loss_history: List[Tuple[int, float]] = []
    throughput_history: List[float] = []
    step_times_ms: List[float] = []
    moving_tps = collections.deque(maxlen=20)
    grad_norms: List[float] = []
    nan_flag = False
    oom_flag = False
    best_loss = float("inf")
    start_time = time.perf_counter()
    gpu_stats = GPUStats()

    loss_curve_fh = loss_curve_path.open("w", newline="", encoding="utf-8")
    loss_writer = csv.writer(loss_curve_fh)
    loss_writer.writerow(["step", "loss"])

    try:
        log_progress(f"Starting training loop for {landing.label} with {args.steps} steps.")
        for step in range(1, args.steps + 1):
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            step_tokens = 0
            micro_losses = []
            for _ in range(args.grad_accum):
                inputs, labels = train_batcher.next_batch()
                step_tokens += inputs.numel()
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype if amp_enabled else torch.float32,
                    enabled=amp_enabled,
                ):
                    logits = model(inputs)
                    loss = criterion(
                        logits.view(-1, tokenizer.vocab_size),
                        labels.view(-1),
                    )
                    loss = loss / args.grad_accum
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_flag = True
                    raise RuntimeError("NaN detected in loss.")
                micro_losses.append(float(loss.item()))
                if scaler is not None and amp_enabled and args.amp_mode == "fp16":
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
            )
            grad_norms.append(grad_norm)
            if scaler is not None and amp_enabled and args.amp_mode == "fp16":
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_time = time.perf_counter() - step_start
            step_ms = step_time * 1000.0
            step_times_ms.append(step_ms)
            tokens_total += step_tokens
            tokens_per_sec = step_tokens / max(step_time, 1e-6)
            moving_tps.append(tokens_per_sec)
            throughput_history.append(tokens_per_sec)
            avg_loss = float(sum(micro_losses) / max(1, len(micro_losses)))
            loss_history.append((step, avg_loss))
            best_loss = min(best_loss, avg_loss)
            loss_writer.writerow([step, avg_loss])
            gpu_stats.sample()
            payload = {
                "step": step,
                "loss": avg_loss,
                "tokens_per_sec": tokens_per_sec,
                "moving_avg_tokens_per_sec": sum(moving_tps) / max(1, len(moving_tps)),
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": grad_norm,
                "ms_per_step": step_ms,
                "tokens_this_step": step_tokens,
            }
            if step % args.log_interval == 0 or step == 1:
                print(
                    f"[{landing.label}] step {step}/{args.steps} "
                    f"loss={avg_loss:.4f} "
                    f"tps={tokens_per_sec:,.0f} "
                    f"lr={payload['lr']:.2e}",
                    flush=True,
                )
            step_log.log(payload)
            if args.max_seconds and (time.perf_counter() - start_time) > args.max_seconds:
                print(
                    f"[info] Max wall-clock {args.max_seconds}s reached; stopping early.",
                    flush=True,
                )
                break
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            oom_flag = True
            print("[error] CUDA OOM encountered; aborting run.", flush=True)
            torch.cuda.empty_cache()
            log_progress("Encountered CUDA OOM; emptied cache.")
        else:
            log_progress(f"Runtime error encountered: {exc}")
            raise
    finally:
        step_log.close()
        loss_curve_fh.close()
        gpu_stats.close()
        log_progress("Training loop finished; files closed.")

    steps_completed = len(loss_history)
    wall_clock = time.perf_counter() - start_time
    avg_tokens_sec = sum(throughput_history) / max(1, len(throughput_history))
    p50_ms, p90_ms, p99_ms = compute_step_quantiles(step_times_ms)
    peak_mem_gb = 0.0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device)
        peak_mem_gb = peak_mem / (1024**3)
    eval_stats = None
    if eval_batcher is not None and steps_completed > 0:
        log_progress("Running evaluation on validation stream.")
        eval_stats = evaluate_model(
            model,
            eval_batcher,
            device=device,
            vocab_size=tokenizer.vocab_size,
            eval_batches=args.eval_batches,
        )
    pr_value = None
    if args.pr_samples > 0 and val_stream is not None and steps_completed > 0:
        log_progress("Running participation ratio diagnostic.")
        diag_batcher = SequenceBatcher(
            val_stream,
            tokenizer,
            seq_len=args.seq_len,
            micro_batch_size=max(1, min(4, micro_batch)),
        )
        pr_value = snapshot_participation_ratio(
            model,
            diag_batcher,
            device=device,
            samples=args.pr_samples,
        )

    metrics = {
        "label": landing.label,
        "seed": args.seed,
        "target_params": landing.target_params,
        "landed_params": landing.landed_params,
        "param_error_pct": landing.error_pct,
        "config": asdict(landing),
        "training": {
            "steps_requested": args.steps,
            "steps_completed": steps_completed,
            "tokens_per_step": tokens_per_step,
            "tokens_total": tokens_total,
            "tokens_per_sec_avg": avg_tokens_sec,
            "wall_clock_s": wall_clock,
            "grad_accum": args.grad_accum,
            "seq_len": args.seq_len,
            "grad_clip": args.grad_clip,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "amp_mode": args.amp_mode,
            "compile": args.compile,
        },
        "optimizer": {
            "beta1": args.beta1,
            "beta2": args.beta2,
        },
        "loss": {
            "final": loss_history[-1][1] if loss_history else None,
            "best": best_loss if best_loss < float("inf") else None,
        },
        "eval": eval_stats,
        "diagnostics": {
            "participation_ratio": pr_value,
            "grad_norms": grad_norms[-min(len(grad_norms), 16) :],
            "step_time_ms": {
                "p50": p50_ms,
                "p90": p90_ms,
                "p99": p99_ms,
            },
            "gpu_util_max": gpu_stats.max_gpu_util,
            "gpu_mem_util_max": gpu_stats.max_mem_util,
        },
        "stability": {
            "nan": nan_flag,
            "oom": oom_flag,
        },
        "environment": {
            "git_commit": get_git_commit(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch_version": torch.__version__,
        },
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log_progress(f"Wrote metrics.json for {landing.label} at {metrics_path}")

    final_loss = loss_history[-1][1] if loss_history else float("nan")
    eval_ppl = eval_stats["ppl"] if eval_stats else None
    return RunResult(
        label=landing.label,
        seed=args.seed,
        target_params=landing.target_params,
        landed_params=landing.landed_params,
        steps_completed=steps_completed,
        tokens_per_step=tokens_per_step,
        tokens_total=tokens_total,
        avg_tokens_sec=avg_tokens_sec,
        p50_ms=p50_ms,
        p90_ms=p90_ms,
        p99_ms=p99_ms,
        peak_mem_gb=peak_mem_gb,
        final_train_loss=final_loss,
        eval_ppl=eval_ppl,
        wall_clock_s=wall_clock,
        metrics_path=metrics_path,
        stability={"nan": nan_flag, "oom": oom_flag},
        throughput_trace=throughput_history,
        loss_trace=loss_history,
        participation_ratio=pr_value,
    )
