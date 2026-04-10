from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

from psannlm.lm.data.tokenizer import Tokenizer, TokenizerConfig

from .data import _tokenizer_sample_iterator


def _ensure_tokenizer_config(special_map_path: Optional[str], max_length: int) -> Optional[str]:
    if not special_map_path:
        return None
    special = Path(special_map_path)
    if not special.exists():
        return None
    cfg_path = special.with_name("tokenizer_config.json")
    if cfg_path.exists():
        return str(cfg_path)
    try:
        with special.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return None
    config = dict(data)
    config["model_max_length"] = int(max_length)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    return str(cfg_path)


def _prepare_tokenizer(
    args: argparse.Namespace,
    shard_paths: list[str],
) -> tuple[Tokenizer, dict]:
    backend = str(args.tokenizer_backend or "auto").lower()
    if args.train_tokenizer and backend != "tokenizers":
        raise SystemExit("--train-tokenizer currently supports --tokenizer-backend tokenizers.")

    tok_model = args.tokenizer_model_path
    tok_special = args.tokenizer_special_map_path
    if args.train_tokenizer:
        tok_model = None
        tok_special = None

    artifacts: dict[str, Optional[str] | bool] = {
        "model": tok_model,
        "special_map": tok_special,
        "config": None,
        "dir": None,
        "trained": bool(args.train_tokenizer),
    }

    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    is_rank0 = rank == 0

    if args.train_tokenizer:
        limit = (
            None
            if args.tokenizer_sample_limit is None or int(args.tokenizer_sample_limit) <= 0
            else int(args.tokenizer_sample_limit)
        )
        save_dir = Path(args.tokenizer_save_dir or os.path.join(args.checkpoint_dir, "tokenizer"))
        tok_json = save_dir / "tokenizer.json"
        special_map = save_dir / "special_tokens_map.json"
        done_flag = save_dir / ".done"
        save_dir.mkdir(parents=True, exist_ok=True)

        already_trained = tok_json.exists() and special_map.exists() and done_flag.exists()
        if already_trained:
            artifacts["model"] = str(tok_json)
            artifacts["special_map"] = str(special_map)
            artifacts["dir"] = str(save_dir)
            cfg_path = _ensure_tokenizer_config(artifacts["special_map"], args.max_length)
            if cfg_path:
                artifacts["config"] = cfg_path
            if is_rank0:
                print(f"[tokenizer] Reusing existing tokenizer at {tok_json}")
        elif is_rank0:
            train_cfg = TokenizerConfig(
                backend=backend,
                model_path=None,
                special_tokens_map_path=None,
                vocab_size=int(args.tokenizer_vocab_size),
                min_frequency=int(args.tokenizer_min_frequency),
                hf_passthrough_ids=(backend == "tokenizers"),
            )
            trainer_tok = Tokenizer(train_cfg)
            samples = _tokenizer_sample_iterator(args, shard_paths, limit)
            trainer_tok.fit(samples)
            trainer_tok.save(str(tok_json), special_tokens_map_path=str(special_map))
            artifacts["model"] = str(tok_json)
            artifacts["special_map"] = str(special_map)
            artifacts["dir"] = str(save_dir)
            cfg_path = _ensure_tokenizer_config(artifacts["special_map"], args.max_length)
            if cfg_path:
                artifacts["config"] = cfg_path
            done_flag.write_text("ok", encoding="utf-8")
            print(f"[tokenizer] Trained tokenizer saved to {tok_json}")
        else:
            wait_paths = [tok_json, special_map]
            while not all(p.exists() for p in wait_paths):
                if done_flag.exists():
                    break
                time.sleep(1.0)
            artifacts["model"] = str(tok_json)
            artifacts["special_map"] = str(special_map)
            artifacts["dir"] = str(save_dir)
            cfg_path = _ensure_tokenizer_config(artifacts["special_map"], args.max_length)
            if cfg_path:
                artifacts["config"] = cfg_path

    final_cfg = TokenizerConfig(
        backend=backend,
        model_path=artifacts["model"],
        special_tokens_map_path=artifacts["special_map"],
        vocab_size=int(args.tokenizer_vocab_size),
        min_frequency=int(args.tokenizer_min_frequency),
        hf_passthrough_ids=(backend == "tokenizers"),
    )
    tokenizer = Tokenizer(final_cfg)
    try:
        tokenizer.fit([""])
    except Exception:
        pass
    cfg_path = _ensure_tokenizer_config(artifacts["special_map"], args.max_length)
    if cfg_path:
        artifacts["config"] = cfg_path

    return tokenizer, artifacts
