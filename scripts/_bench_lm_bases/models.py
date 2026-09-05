"""Resolve benchmark model mappings; old configuration files remain readable."""

from copy import deepcopy
from psannlm.architectures import LMConfig, normalize_lm_config
from psannlm.architectures.compat import legacy_lm_config
from psannlm.lm.models.sine import SineConfig


def legacy_benchmark_model(train_cfg, name, vocab_size=None):
    """Compatibility translation of the historical benchmark defaults."""
    sine_cfg = train_cfg.get("sine_params", {}) or {}
    geosparse_kwargs = (
        {k: v for k, v in train_cfg.items() if k.startswith("geosparse_") and v is not None}
        if name == "geosparse"
        else {}
    )
    return legacy_lm_config(
        name,
        dict(
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
        ),
        warn=False,
    )


def _merge(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = deepcopy(value)
    return target


def benchmark_model_config(cfg, name, vocab_size=None):
    """Normalize a named LMConfig after applying shared canonical model overrides."""
    models = cfg.get("models", {})
    if models:
        if name not in models:
            raise ValueError(f"Unknown benchmark model {name!r}; choose from {tuple(models)}")
        model = _merge(deepcopy(models[name]), cfg.get("model_overrides", {}))
        model["vocab_size"] = vocab_size
        return normalize_lm_config(model)
    if cfg.get("bench", {}).get("bases") or name in (
        "respsann",
        "sgrpsann",
        "waveresnet",
        "geosparse",
    ):
        return legacy_benchmark_model(cfg.get("train", {}), name, vocab_size)
    return LMConfig(architecture=name, vocab_size=vocab_size)
