"""Typed configuration shells for PSANN-LM.

These dataclasses are intentionally minimal and will evolve alongside
the trainer and model implementations. They provide a clear place to
hold options that also maps cleanly to CLI/YAML.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional

POS_ENCODING_CHOICES = ("rope", "alibi", "sinusoidal")
_DEFAULT_NUM_WORKERS = 0 if os.name == "nt" else 8


def normalize_positional_encoding(value: Optional[str]) -> str:
    enc = "rope" if value is None else str(value).strip().lower()
    if enc not in POS_ENCODING_CHOICES:
        raise ValueError(
            f"positional_encoding must be one of {POS_ENCODING_CHOICES}; received '{value}'."
        )
    return enc


@dataclass
class ModelConfig:
    """0.x flat configuration shell; use LMConfig for maintained construction."""

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
        from ..architectures.compat import BASE_KINDS, compatibility_warning

        if self.base.lower() not in BASE_KINDS:
            raise ValueError(f"base must be one of {tuple(BASE_KINDS)}")
        if self.d_model <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("d_model, n_layers, n_heads must be positive")
        if self.d_mlp is not None and self.d_mlp <= 0:
            raise ValueError("d_mlp must be positive when provided")
        if self.vocab_size is not None and self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive when provided")
        self.positional_encoding = normalize_positional_encoding(self.positional_encoding)
        compatibility_warning("ModelConfig is deprecated; use LMConfig or to_lm_config().")

    def to_lm_config(self):
        from ..architectures.compat import legacy_lm_config

        return legacy_lm_config(
            self.base,
            {
                "d_model": self.d_model,
                "n_layers": self.n_layers,
                "n_heads": self.n_heads,
                "d_mlp": self.d_mlp,
                "vocab_size": self.vocab_size,
                "positional_encoding": self.positional_encoding,
                "sine": {
                    "amp_init": self.sine_amp_init,
                    "freq_init": self.sine_freq_init,
                    "damp_init": self.sine_damp_init,
                    "trainable": self.sine_trainable,
                },
            },
            warn=False,
        )


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


@dataclass(frozen=True)
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
    # Knowledge distillation (optional). Activated when `distill_alpha > 0`
    # and a teacher model is provided to Trainer.train(...).
    distill_alpha: float = 0.0
    distill_temperature: float = 1.0
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
    dataloader_num_workers: int = _DEFAULT_NUM_WORKERS
    dataloader_prefetch_factor: int = 2
    dataloader_persistent_workers: bool = True
    hf_cache_limit_gb: float | None = None
    # Eval (optional). When val_dataset is provided, trainer can periodically report ppl.
    eval_interval_steps: int = 0
    eval_max_batches: int = 0
    # torch.compile (PyTorch 2.x). Only enabled when explicitly requested.
    torch_compile: bool = False
    torch_compile_mode: str = "default"  # default | reduce-overhead | max-autotune
    torch_compile_fullgraph: bool = False
    torch_compile_dynamic: bool = False
    # CUDA memory QoL (optional)
    cuda_empty_cache_after_init: bool = False
    cuda_empty_cache_interval_steps: int = 0

    def __post_init__(self) -> None:
        from ..architectures.config import integer, real

        positive_ints = {
            "epochs",
            "batch_tokens",
            "grad_accum_steps",
            "fsdp_min_params",
            "log_interval_steps",
            "save_interval_steps",
            "dataloader_prefetch_factor",
        }
        nonnegative_ints = {
            "warmup_steps",
            "dataloader_num_workers",
            "eval_interval_steps",
            "eval_max_batches",
            "cuda_empty_cache_interval_steps",
        }
        for field in fields(self):
            name, value = field.name, getattr(self, field.name)
            if name in positive_ints | nonnegative_ints:
                integer(value, "train." + name, 1 if name in positive_ints else 0)
            elif name == "steps_per_epoch":
                if value is not None:
                    integer(value, "train." + name)
            elif field.type == "bool":
                if not isinstance(value, bool):
                    raise TypeError(f"train.{name} must be a boolean.")
            elif field.type in {"float", "float | None"}:
                if value is not None:
                    real(value, "train." + name)
            elif field.type == "str" and not isinstance(value, str):
                raise TypeError(f"train.{name} must be a string.")
        if not isinstance(self.betas, (list, tuple)) or len(self.betas) != 2:
            raise TypeError("train.betas must be a pair.")
        betas = tuple(real(v, "train.betas") for v in self.betas)
        if any(not 0 <= v < 1 for v in betas):
            raise ValueError("train.betas must be in [0, 1).")
        object.__setattr__(self, "betas", betas)
        if self.eps <= 0 or self.weight_decay < 0:
            raise ValueError("train.eps must be positive and train.weight_decay non-negative.")
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
        if self.distill_alpha < 0 or self.distill_alpha > 1:
            raise ValueError("distill_alpha must be in [0, 1]")
        if self.distill_temperature <= 0:
            raise ValueError("distill_temperature must be > 0")
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
        if self.eval_interval_steps < 0:
            raise ValueError("eval_interval_steps must be >= 0")
        if self.eval_max_batches < 0:
            raise ValueError("eval_max_batches must be >= 0")
        if self.cuda_empty_cache_interval_steps < 0:
            raise ValueError("cuda_empty_cache_interval_steps must be >= 0")
        if self.torch_compile_mode.strip() not in {
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        }:
            raise ValueError("train.torch_compile_mode is invalid.")
        object.__setattr__(self, "torch_compile_mode", self.torch_compile_mode.strip())


def normalize_train_config(value: TrainConfig | Mapping[str, Any]) -> TrainConfig:
    """Normalize one immutable train policy, accepting the optional kind='train' tag."""
    if isinstance(value, TrainConfig):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("train must be TrainConfig or a mapping.")
    raw = dict(value)
    if "kind" in raw and raw.pop("kind") != "train":
        raise ValueError("train.kind must be 'train'.")
    known = {f.name for f in fields(TrainConfig)}
    for key in raw:
        if key not in known:
            raise ValueError(f"train.{key} is unknown.")
    try:
        return TrainConfig(**raw)
    except (ValueError, TypeError) as exc:
        raise type(exc)("train: " + str(exc)) from exc
