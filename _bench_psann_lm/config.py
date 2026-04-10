# ruff: noqa: F403,F405
from __future__ import annotations

from .shared import *

DATASET_ALIASES = {
    "wikitext-103": ("iohadrubin/wikitext-103-raw-v1", None),
    "wikitext103": ("iohadrubin/wikitext-103-raw-v1", None),
    "wikitext_103": ("iohadrubin/wikitext-103-raw-v1", None),
    "wikitext": ("iohadrubin/wikitext-103-raw-v1", None),
    "wikitext-2": ("wikitext", "wikitext-2-raw-v1"),
    "wikitext2": ("wikitext", "wikitext-2-raw-v1"),
    "wikitext_2": ("wikitext", "wikitext-2-raw-v1"),
}


def log_progress(message: str) -> None:
    """Emit a flushed progress line for long-running RunPod jobs."""
    print(f"[bench] {message}", flush=True)


@dataclass
class LandingConfig:
    label: str
    target_params: int
    landed_params: int
    error_pct: float
    d_model: int
    n_layers: int
    n_heads: int
    d_mlp: int
    dropout: float
    positional_encoding: str
    wave_interleave: bool
    wave_kernel_size: int
    wave_dilation_growth: int


@dataclass
class RunResult:
    label: str
    seed: int
    target_params: int
    landed_params: int
    steps_completed: int
    tokens_per_step: int
    tokens_total: int
    avg_tokens_sec: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    peak_mem_gb: float
    final_train_loss: float
    eval_ppl: Optional[float]
    wall_clock_s: float
    metrics_path: Path
    stability: Dict[str, bool]
    throughput_trace: List[float]
    loss_trace: List[Tuple[int, float]]
    participation_ratio: Optional[float]


def parse_size_targets(arg: str) -> List[Tuple[str, int]]:
    if not arg:
        raise ValueError("At least one size must be provided.")
    log_progress(f"parse_size_targets -> raw='{arg}'")
    tokens = [tok.strip() for tok in arg.split(",") if tok.strip()]
    results: List[Tuple[str, int]] = []
    for tok in tokens:
        label = tok.upper().replace(" ", "")
        mult = 1.0
        numeric = tok
        if label.endswith("M"):
            mult = 1_000_000
            numeric = label[:-1]
        elif label.endswith("B"):
            mult = 1_000_000_000
            numeric = label[:-1]
        try:
            value = float(numeric)
        except ValueError as exc:
            raise ValueError(f"Could not parse size token '{tok}'.") from exc
        params = int(value * mult)
        if params <= 0:
            raise ValueError(f"Size token '{tok}' resolved to non-positive params.")
        results.append((label, params))
    log_progress(f"parse_size_targets -> landed={results}")
    return results


def seed_everything(seed: int) -> None:
    log_progress(f"Seeding RNGs with seed={seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:  # pragma: no cover - optional dep
        pass


def resolve_dataset(
    dataset: str, explicit_hub_id: Optional[str], dataset_name: Optional[str]
) -> Tuple[str, Optional[str]]:
    log_progress(
        f"resolve_dataset -> dataset={dataset} explicit_hub_id={explicit_hub_id} dataset_name={dataset_name}"
    )
    if explicit_hub_id:
        return explicit_hub_id, dataset_name
    key = dataset.strip().lower()
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    if "/" in dataset:
        return dataset, dataset_name
    return dataset, dataset_name
