"""Small, reproducible experiment presets and shared configuration contracts."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

import yaml
from psannlm import LMArchitectureConfig, LMConfig
from psannlm.architectures import LMActivationInitializationConfig, to_mapping

from psann.architectures import ActivationConfig
from scripts._bench_lm_bases.models import benchmark_model_config
from scripts._bench_lm_bases.sweep import _expand_sweep_configs


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def read_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("Exploration configuration must be a mapping.")
    unknown = set(config) - {
        "models",
        "model_overrides",
        "sweep",
        "bench",
        "train",
        "data",
        "tokenizer",
        "exploration",
    }
    if unknown:
        raise ValueError(f"Unknown exploration configuration fields: {sorted(unknown)}")
    if not config.get("models"):
        raise ValueError("models must contain named canonical LM configurations.")
    return config


def trials(config, vocab_size, target=False):
    seeds = [int(seed) for seed in config.get("bench", {}).get("seeds", [1337])]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("bench.seeds must contain unique seeds.")
    result = []
    for sweep in _expand_sweep_configs(config):
        resolved = sweep["cfg"]
        train = resolved.get("train", {})
        steps, warmup = int(train.get("max_steps", 32)), int(train.get("warmup_steps", 4))
        if steps <= 0 or warmup < 0 or warmup * 2 >= steps:
            raise ValueError("Short trials require max_steps > 2 * warmup_steps >= 0.")
        for name in resolved["models"]:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
                raise ValueError(
                    "Model labels must contain only letters, digits, dot, dash or underscore."
                )
            model = benchmark_model_config(resolved, name, vocab_size)
            model_values = to_mapping(model)
            if target:
                dimensions = config.get("exploration", {}).get("target_dimensions")
                if not dimensions:
                    raise ValueError("--target requires exploration.target_dimensions.")
                model_values.update(dimensions)
            for seed in seeds:
                identifier = f"sweep{sweep['id']:03d}_{name}_seed{seed}"
                result.append(
                    {
                        "id": identifier,
                        "name": name,
                        "seed": int(seed),
                        "model": deepcopy(model_values),
                        "train": deepcopy(train),
                        "overrides": sweep["overrides"],
                    }
                )
    return result


def preset(name):
    smoke = name == "smoke"
    dims = (
        {"d_model": 24, "n_layers": 2, "n_heads": 3, "d_mlp": 48}
        if smoke
        else {"d_model": 256, "n_layers": 4, "n_heads": 4, "d_mlp": 1024}
    )
    tuned = ActivationConfig(frequency_init=2.25, decay_init=0.001)
    architectures = {
        "transformer": LMArchitectureConfig.transformer(),
        "wave_default": LMArchitectureConfig.wave(),
        "wave_tuned": LMArchitectureConfig.wave(activation=tuned),
    }
    if not smoke:
        architectures["wave_spread"] = LMArchitectureConfig.wave(
            activation=tuned,
            activation_initialization=LMActivationInitializationConfig(frequency_std=0.25),
        )
    config = {
        "bench": {"seeds": [1337]},
        "models": {
            key: to_mapping(LMConfig(architecture=value, **dims))
            for key, value in architectures.items()
        },
        "train": {
            "max_steps": 12 if smoke else 256,
            "warmup_steps": 1 if smoke else 16,
            "lr": 0.003 if smoke else 0.0008,
            "batch_tokens": 64 if smoke else 2048,
            "grad_accum_steps": 1,
            "amp": "fp32" if smoke else "bf16",
            "weight_decay": 0.01,
        },
        "exploration": {
            "eval_interval": 4 if smoke else 32,
            "eval_tokens": 128 if smoke else 32768,
            "max_seconds": 120 if smoke else 7200,
            "target_parameters": 304565264,
            "target_dimensions": {"d_model": 1024, "n_layers": 16, "n_heads": 16, "d_mlp": 4096},
        },
        "data": {
            "sequence_length": 16 if smoke else 256,
            "validation_fraction": 0.2 if smoke else 0.05,
            "split_seed": 731,
            "max_documents": 500 if smoke else 10000,
            "train_tokens": 4096 if smoke else 2097152,
            "validation_tokens": 1024 if smoke else 131072,
        },
        "tokenizer": (
            {"backend": "simple"}
            if smoke
            else {
                "backend": "tokenizers",
                "model_path": "runs/tokenizer_300m_fineweb_edu_sample10BT/tokenizer.json",
                "special_tokens_map_path": "runs/tokenizer_300m_fineweb_edu_sample10BT/special_tokens_map.json",
                "hf_passthrough_ids": True,
            }
        ),
    }
    if smoke:
        config["data"]["texts"] = [
            f"Document {i}: waves and language models learn useful patterns from text. " * 4
            for i in range(40)
        ]
        config["exploration"][
            "purpose"
        ] = "Engineering smoke test only; synthetic text cannot select a production architecture."
    else:
        config["data"]["files"] = ["eval_data/fineweb_edu_sample10BT_eval.jsonl"]
        config["exploration"][
            "purpose"
        ] = "Proxy search on a newly hash-partitioned local FineWeb-Edu sample; larger-scale confirmation required."
        config["sweep"] = {"train.lr": [0.0002, 0.0008, 0.0012]}
    return config
