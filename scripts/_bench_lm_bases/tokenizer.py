# ruff: noqa: F403,F405
from __future__ import annotations

from .shared import *


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


def _iter_hf_texts(
    data_cfg: Dict[str, Any],
    limit: Optional[int],
    seed: int,
) -> Iterable[str]:
    from datasets import load_dataset  # type: ignore

    text_filter = build_text_filter(
        ascii_only=bool(data_cfg.get("ascii_only", False)),
        languages=list(data_cfg.get("languages") or []),
        lang_threshold=float(data_cfg.get("lang_threshold", 0.8)),
    )
    data_files = data_cfg.get("data_files")
    if isinstance(data_files, str):
        df = data_files.strip()
        if "," in df:
            data_files = [s.strip() for s in df.split(",") if s.strip()]
        else:
            data_files = df
    if data_files:
        stream = load_dataset(
            data_cfg["dataset"],
            data_files=data_files,
            split=data_cfg.get("train_split", "train"),
            streaming=True,
            revision=data_cfg.get("revision"),
        )
    else:
        stream = load_dataset(
            data_cfg["dataset"],
            name=data_cfg.get("name"),
            split=data_cfg.get("train_split", "train"),
            streaming=True,
            revision=data_cfg.get("revision"),
        )
    if bool(data_cfg.get("shuffle", True)):
        try:
            stream = stream.shuffle(
                seed=int(seed), buffer_size=int(data_cfg.get("shuffle_buffer", 10000))
            )
        except Exception:
            pass
    yielded = 0
    for row in stream:
        try:
            text = str(row.get(data_cfg.get("text_key", "text"), "")).strip()
        except Exception:
            text = ""
        if not text:
            continue
        if not text_filter(text):
            continue
        yield text
        yielded += 1
        if limit is not None and yielded >= limit:
            break


def _ensure_tokenizer(
    cfg: Dict[str, Any],
    *,
    outdir: Path,
    seed: int,
) -> tuple[Tokenizer, Dict[str, Any]]:
    tok_cfg = cfg.get("tokenizer", {})
    data_cfg = cfg.get("data", {})
    backend = str(tok_cfg.get("backend", "tokenizers")).lower()
    reuse = bool(cfg.get("bench", {}).get("reuse_tokenizer", True))
    save_dir = Path(tok_cfg.get("save_dir") or outdir / "tokenizer")
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = tok_cfg.get("model_path")
    special_map_path = tok_cfg.get("special_tokens_map_path")
    tok_json = save_dir / "tokenizer.json"
    special_map = save_dir / "special_tokens_map.json"

    if not model_path and reuse and tok_json.exists():
        model_path = str(tok_json)
        if special_map.exists():
            special_map_path = str(special_map)

    if backend == "tokenizers" and not model_path:
        limit = tok_cfg.get("sample_limit")
        limit_val = None if limit is None or int(limit) <= 0 else int(limit)
        train_cfg = TokenizerConfig(
            backend=backend,
            model_path=None,
            special_tokens_map_path=None,
            vocab_size=int(tok_cfg.get("vocab_size", 16384)),
            min_frequency=int(tok_cfg.get("min_frequency", 2)),
            hf_passthrough_ids=True,
        )
        trainer_tok = Tokenizer(train_cfg)
        samples = _iter_hf_texts(data_cfg, limit=limit_val, seed=seed)
        trainer_tok.fit(samples)
        trainer_tok.save(str(tok_json), special_tokens_map_path=str(special_map))
        model_path = str(tok_json)
        special_map_path = str(special_map)

    if backend == "tokenizers":
        hf_passthrough = True
    else:
        hf_passthrough = False

    final_cfg = TokenizerConfig(
        backend=backend,
        model_path=str(model_path) if model_path else None,
        special_tokens_map_path=str(special_map_path) if special_map_path else None,
        vocab_size=int(tok_cfg.get("vocab_size", 16384)),
        min_frequency=int(tok_cfg.get("min_frequency", 2)),
        hf_passthrough_ids=hf_passthrough,
    )
    tokenizer = Tokenizer(final_cfg)
    try:
        tokenizer.fit([""])
    except Exception:
        pass

    cfg_path = _ensure_tokenizer_config(
        special_map_path,
        int(data_cfg.get("max_length", 512)),
    )
    meta = {
        "backend": backend,
        "model_path": model_path,
        "special_tokens_map_path": special_map_path,
        "tokenizer_config_path": cfg_path,
        "vocab_size": tokenizer.vocab_size,
        "save_dir": str(save_dir),
    }
    return tokenizer, meta


def _build_stream_dataset(
    data_cfg: Dict[str, Any],
    tokenizer: Tokenizer,
    *,
    split: str,
    shuffle: bool,
    seed: int,
) -> HFTextStreamingLMDataset:
    pack = PackingConfig(
        max_length=int(data_cfg.get("max_length", 512)),
        pack_sequences=True,
    )
    return HFTextStreamingLMDataset(
        dataset=str(data_cfg.get("dataset")),
        name=data_cfg.get("name"),
        data_files=data_cfg.get("data_files"),
        revision=data_cfg.get("revision"),
        split=str(split),
        text_key=str(data_cfg.get("text_key", "text")),
        shuffle=bool(shuffle),
        seed=int(seed),
        shuffle_buffer=int(data_cfg.get("shuffle_buffer", 10000)),
        tokenizer=tokenizer,
        cfg=pack,
        ascii_only=bool(data_cfg.get("ascii_only", False)),
        languages=list(data_cfg.get("languages") or []),
        lang_threshold=float(data_cfg.get("lang_threshold", 0.8)),
    )
