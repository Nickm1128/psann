from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional


def _export_bundle(
    args: argparse.Namespace,
    *,
    final_ckpt: Path,
    tokenizer_artifacts: dict,
    shard_paths: list[str],
) -> None:
    if not args.export_dir:
        return
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    def _copy(src: Path, dst_name: Optional[str] = None) -> Optional[Path]:
        if not src.exists():
            return None
        target = export_dir / (dst_name or src.name)
        shutil.copy2(src, target)
        return target

    copied = []
    model_copy = _copy(final_ckpt, "model.pt")
    if model_copy:
        copied.append(str(model_copy))
    for key in ("model", "special_map", "config"):
        path = tokenizer_artifacts.get(key)
        if not path:
            continue
        copied_path = _copy(Path(path))
        if copied_path:
            copied.append(str(copied_path))

    meta = {
        "model": {
            "base": args.base,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "d_mlp": args.d_mlp if args.d_mlp is not None else 4 * args.d_model,
            "positional_encoding": args.pos_enc,
        },
        "tokenizer": {
            "backend": args.tokenizer_backend,
            "trained": bool(tokenizer_artifacts.get("trained")),
            "files": {
                k: Path(v).name
                for k, v in tokenizer_artifacts.items()
                if k in {"model", "special_map", "config"} and v
            },
        },
        "data": (
            {
                "type": "hf_dataset",
                "dataset": args.hf_dataset,
                "name": args.hf_name,
                "split": args.hf_split,
                "revision": args.hf_revision,
                "text_key": args.hf_text_key,
                "filters": {
                    "ascii_only": bool(args.hf_keep_ascii_only),
                    "languages": args.hf_lang or [],
                    "lang_threshold": args.hf_lang_threshold,
                },
            }
            if args.hf_dataset
            else {
                "type": "manifest",
                "path": args.data_manifest,
                "num_shards": len(shard_paths),
            }
        ),
        "training": {
            "epochs": args.epochs,
            "batch_tokens": args.batch_tokens,
            "grad_accum_steps": args.grad_accum_steps,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "amp": args.amp,
            "fsdp": args.fsdp,
            "grad_checkpoint": bool(args.grad_checkpoint),
            "max_length": args.max_length,
        },
        "artifacts": copied,
    }
    meta_path = export_dir / "psann_artifacts.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[export] Assets copied to {export_dir} (metadata: {meta_path})")
