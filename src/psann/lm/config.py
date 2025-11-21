"""Typed configuration shells for PSANN-LM.

These dataclasses are intentionally minimal and will evolve alongside
the trainer and model implementations. They provide a clear place to
hold options that also maps cleanly to CLI/YAML.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

POS_ENCODING_CHOICES = ("rope", "alibi", "sinusoidal")


def normalize_positional_encoding(value: Optional[str]) -> str:
    enc = "rope" if value is None else str(value).strip().lower()
    if enc not in POS_ENCODING_CHOICES:
        raise ValueError(
            f"positional_encoding must be one of {POS_ENCODING_CHOICES}; received '{value}'."
        )
    return enc


@dataclass
class ModelConfig:
    base: str = "waveresnet"  # or "respsann"
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_mlp: Optional[int] = None
    vocab_size: Optional[int] = None
    positional_encoding: str = "rope"
    # Sine params kept flat for YAML friendliness
    sine_amp_init: float = 1.0
    sine_freq_init: float = 1.0
    sine_damp_init: float = 0.01
    sine_trainable: bool = True

    def __post_init__(self) -> None:
        if self.base.lower() not in {"waveresnet", "respsann"}:
            raise ValueError("base must be 'waveresnet' or 'respsann'")
        if self.d_model <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("d_model, n_layers, n_heads must be positive")
        if self.d_mlp is not None and self.d_mlp <= 0:
            raise ValueError("d_mlp must be positive when provided")
        if self.vocab_size is not None and self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive when provided")
        self.positional_encoding = normalize_positional_encoding(self.positional_encoding)


@dataclass
class DataConfig:
    tokenizer: str = "auto"
    max_length: int = 1024
    pack_sequences: bool = True
    val_split: float = 0.01
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if not (0.0 <= float(self.val_split) <= 0.5):
            raise ValueError("val_split should be in [0.0, 0.5]")


@dataclass
class TrainConfig:
    epochs: int = 1
    batch_tokens: int = 32768
    lr: float = 2e-4
    warmup_steps: int = 2000
    weight_decay: float = 0.01
    amp: str = "bf16"  # bf16 | fp16 | fp32 | none
    optimizer: str = "adamw"  # adamw | adamw8bit | adafactor
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    label_smoothing: float = 0.0
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    ddp: str = "auto"  # auto | on | off
    fsdp: str = "off"  # off | full_shard
    fsdp_cpu_offload: bool = False
    fsdp_use_orig_params: bool = True
    fsdp_auto_wrap_policy: str = "size"  # size | none
    fsdp_min_params: int = 1_000_000
    steps_per_epoch: int | None = None
    checkpoint_dir: str = "runs/lm/exp"
    log_interval_steps: int = 50
    save_interval_steps: int = 500
    # Memory/perf knobs
    grad_checkpoint: bool = False
    log_gpu_mem: bool = False
    dataloader_num_workers: int = 8
    dataloader_prefetch_factor: int = 2
    dataloader_persistent_workers: bool = True
    hf_cache_limit_gb: float | None = None

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.batch_tokens <= 0:
            raise ValueError("batch_tokens must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.warmup_steps < 0 or self.save_interval_steps <= 0 or self.log_interval_steps <= 0:
            raise ValueError("warmup/log/save steps must be non-negative/positive respectively")
        if self.grad_clip < 0:
            raise ValueError("grad_clip must be >= 0")
        if self.grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be positive")
        if self.label_smoothing < 0 or self.label_smoothing >= 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        if self.amp.lower() not in {"bf16", "fp16", "fp32", "none"}:
            raise ValueError("amp must be one of {'bf16','fp16','fp32','none'}")
        if self.ddp.lower() not in {"auto", "on", "off"}:
            raise ValueError("ddp must be one of {'auto','on','off'}")
        if self.optimizer.lower() not in {"adamw", "adamw8bit", "adafactor"}:
            raise ValueError("optimizer must be one of {'adamw','adamw8bit','adafactor'}")
        if self.fsdp.lower() not in {"off", "full_shard"}:
            raise ValueError("fsdp must be one of {'off','full_shard'}")
        if self.fsdp_auto_wrap_policy.lower() not in {"size", "none"}:
            raise ValueError("fsdp_auto_wrap_policy must be one of {'size','none'}")
        if self.dataloader_num_workers < 0:
            raise ValueError("dataloader_num_workers must be >= 0")
        if self.dataloader_prefetch_factor < 1:
            raise ValueError("dataloader_prefetch_factor must be >= 1")
        if self.hf_cache_limit_gb is not None and self.hf_cache_limit_gb <= 0:
            raise ValueError("hf_cache_limit_gb must be positive when provided")
