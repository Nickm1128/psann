"""Strict YAML configuration consumed by the canonical training command."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from .architectures import normalize_lm_config
from .architectures.compat import compatibility_warning, legacy_lm_config
from .lm.api import PSANNLM, PSANNLMDataPrep
from .lm.config import normalize_train_config


def run_yaml(
    path: str | Path, *, resume_checkpoint: str | None = None, warn_legacy: bool = True
) -> PSANNLM:
    """Load, normalize, train and save a bounded YAML-configured model."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise TypeError("configuration must be a mapping.")
    unknown = set(raw) - {"model", "data", "train"}
    if unknown:
        raise ValueError(f"configuration.{sorted(unknown)[0]} is unknown.")
    for name in ("model", "data", "train"):
        if not isinstance(raw.get(name), Mapping):
            raise TypeError(f"configuration.{name} must be a mapping.")
    legacy = "architecture" not in raw["model"]
    inactive: list[str] = []
    data = dict(raw["data"])
    sources = data.pop("sources", [])
    if not isinstance(sources, list):
        raise TypeError("data.sources must be a list.")
    texts: list[str] = []
    for source in sources:
        if isinstance(source, Mapping) and set(source) == {"path"}:
            source = source["path"]
        if not isinstance(source, str):
            raise TypeError("data.sources entries must be paths or path mappings.")
        texts.extend(
            line for line in Path(source).read_text(encoding="utf-8").splitlines() if line.strip()
        )
    if not texts:
        raise ValueError("data.sources did not provide any training texts.")
    allowed = {
        "tokenizer",
        "tokenizer_model_path",
        "tokenizer_special_map_path",
        "max_length",
        "pack_sequences",
        "val_split",
        "seed",
    }
    unknown = set(data) - allowed
    if unknown:
        if not legacy:
            raise ValueError(f"data.{sorted(unknown)[0]} is unknown.")
        inactive.extend("data." + key for key in sorted(unknown))
        data = {key: value for key, value in data.items() if key in allowed}
    prepared = PSANNLMDataPrep(texts, **data)
    model = dict(raw["model"])
    legacy = "architecture" not in model
    if legacy:
        base = model.pop("base", "waveresnet")
        # The old YAML entrypoint explicitly forwarded this subset to the high-level API.
        active = {
            "d_model",
            "n_layers",
            "n_heads",
            "d_mlp",
            "vocab_size",
            "sine_params",
            "rope",
            "positional_encoding",
        }
        inactive.extend("model." + key for key in sorted(set(model) - active))
        model = {k: v for k, v in model.items() if k in active}
        model.setdefault("d_mlp", 2048)
        config = legacy_lm_config(base, model, high_level=True, warn=False)
    else:
        config = normalize_lm_config(model)
    if config.vocab_size is None:
        config = replace(config, vocab_size=prepared.vocab_size)
    train_values: dict[str, Any] = dict(raw["train"])
    if legacy:
        # Historical YAML forwarded only these five fit options. The output
        # directory selected the model artifact, not the trainer checkpoint path.
        active_train = {"epochs", "batch_tokens", "lr", "amp", "ddp"}
        inactive.extend(
            "train." + key for key in sorted(set(train_values) - active_train - {"checkpoint_dir"})
        )
        train_values = {key: value for key, value in train_values.items() if key in active_train}
        train_values.setdefault("batch_tokens", 131072)
        if "amp" in train_values:
            train_values["amp"] = str(train_values["amp"])
    if legacy and warn_legacy:
        compatibility_warning(
            "Legacy YAML model.base is deprecated; use model.architecture. "
            + ("Inactive YAML fields ignored: " + ", ".join(inactive) if inactive else "")
        )
    train = normalize_train_config(train_values)
    result = PSANNLM(config=config)
    result.fit(prepared, train=train, resume_checkpoint=resume_checkpoint)
    out = (
        Path(str(raw["train"].get("checkpoint_dir", "runs/lm/exp")))
        if legacy
        else Path(train.checkpoint_dir)
    )
    out.mkdir(parents=True, exist_ok=True)
    result.save(str(out / "final_model.pt"))
    return result
