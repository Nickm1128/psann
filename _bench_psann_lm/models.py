# ruff: noqa: F403,F405
from __future__ import annotations

from .config import LandingConfig, log_progress
from .shared import *


def tie_embeddings(model: nn.Module) -> None:
    if hasattr(model, "embed") and hasattr(model, "lm_head"):
        embed = getattr(model, "embed")
        head = getattr(model, "lm_head")
        if isinstance(embed, nn.Embedding) and isinstance(head, nn.Linear):
            head.weight = embed.weight  # type: ignore[assignment]
            log_progress("Tied input embedding weights to LM head.")


def count_model_params(
    base: str,
    vocab_size: int,
    *,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_mlp: int,
    dropout: float,
    positional_encoding: str,
    wave_interleave: bool,
    wave_kernel_size: int,
    wave_dilation_growth: int,
) -> int:
    factory = get_base(base)
    cfg = dict(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_mlp=d_mlp,
        dropout=dropout,
        positional_encoding=positional_encoding,
        wave_interleave=wave_interleave,
        wave_kernel_size=wave_kernel_size,
        wave_dilation_growth=wave_dilation_growth,
    )
    log_progress(
        f"Counting params -> base={base} d_model={d_model} n_layers={n_layers} n_heads={n_heads}"
    )
    model = factory(**cfg)
    total = sum(p.numel() for p in model.parameters())
    del model
    return int(total)


def suggest_head_count(d_model: int, max_heads: int) -> int:
    target = max(4, min(max_heads, d_model // 64 or 1))
    while target > 1 and d_model % target != 0:
        target -= 1
    heads = max(1, target)
    log_progress(f"suggest_head_count -> d_model={d_model} heads={heads}")
    return heads


def land_configs(
    targets: List[Tuple[str, int]],
    *,
    vocab_size: int,
    base: str,
    width_choices: Sequence[int],
    layer_min: int,
    layer_max: int,
    layer_step: int,
    dropout: float,
    positional_encoding: str,
    wave_interleave: bool,
    wave_kernel_size: int,
    wave_dilation_growth: int,
    max_heads: int,
    max_error: float,
) -> List[LandingConfig]:
    landings: List[LandingConfig] = []
    cache: Dict[Tuple[int, int, int], int] = {}
    for label, target in targets:
        best: Optional[LandingConfig] = None
        log_progress(f"Landing config for size {label} ({target/1e6:.2f}M params target)")
        for d_model in width_choices:
            n_heads = suggest_head_count(d_model, max_heads)
            if d_model % n_heads != 0:
                continue
            d_mlp = 4 * d_model
            for n_layers in range(layer_min, layer_max + 1, layer_step):
                key = (d_model, n_layers, n_heads)
                params = cache.get(key)
                if params is None:
                    params = count_model_params(
                        base,
                        vocab_size,
                        d_model=d_model,
                        n_layers=n_layers,
                        n_heads=n_heads,
                        d_mlp=d_mlp,
                        dropout=dropout,
                        positional_encoding=positional_encoding,
                        wave_interleave=wave_interleave,
                        wave_kernel_size=wave_kernel_size,
                        wave_dilation_growth=wave_dilation_growth,
                    )
                    cache[key] = params
                error = abs(params - target) / target
                candidate = LandingConfig(
                    label=label,
                    target_params=target,
                    landed_params=params,
                    error_pct=error,
                    d_model=d_model,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    d_mlp=d_mlp,
                    dropout=dropout,
                    positional_encoding=positional_encoding,
                    wave_interleave=wave_interleave,
                    wave_kernel_size=wave_kernel_size,
                    wave_dilation_growth=wave_dilation_growth,
                )
                if (
                    best is None
                    or error < best.error_pct
                    or (math.isclose(error, best.error_pct) and params < best.landed_params)
                ):
                    best = candidate
        if best is None:
            raise RuntimeError(f"Unable to land configuration for size {label}.")
        if best.error_pct > max_error:
            print(
                f"[warn] Size {label}: landed error {best.error_pct*100:.2f}% exceeds tolerance "
                f"({max_error*100:.2f}%). Consider adjusting width/layer search.",
                flush=True,
            )
        log_progress(
            f"Landed {label}: params={best.landed_params} "
            f"error={best.error_pct*100:.2f}% d_model={best.d_model} layers={best.n_layers}"
        )
        landings.append(best)
    return landings


def build_scheduler(optimizer: AdamW, total_steps: int, warmup_steps: int) -> LambdaLR:
    warmup = max(0, int(warmup_steps))
    log_progress(f"Building scheduler -> total_steps={total_steps} warmup={warmup}")

    def lr_lambda(step: int) -> float:
        s = step + 1
        if warmup > 0 and s <= warmup:
            return float(s) / float(max(1, warmup))
        if total_steps <= warmup:
            return 1.0
        progress = float(s - warmup) / float(max(1, total_steps - warmup))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
