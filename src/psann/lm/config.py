"""Typed configuration shells for PSANN-LM.

These dataclasses are intentionally minimal and will evolve alongside
the trainer and model implementations. They provide a clear place to
hold options that also maps cleanly to CLI/YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    base: str = "waveresnet"  # or "respsann"
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_mlp: Optional[int] = None
    vocab_size: Optional[int] = None
    rope: bool = True
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
    batch_tokens: int = 131072
    lr: float = 2e-4
    warmup_steps: int = 2000
    weight_decay: float = 0.01
    amp: str = "bf16"  # bf16 | fp16 | fp32 | none
    label_smoothing: float = 0.0
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    ddp: str = "auto"  # auto | on | off
    checkpoint_dir: str = "runs/lm/exp"
    log_interval_steps: int = 50
    save_interval_steps: int = 500

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
