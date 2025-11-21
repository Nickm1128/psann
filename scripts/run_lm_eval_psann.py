#!/usr/bin/env python
"""Run lm-eval for a PSANN-LM checkpoint via local registry shim.

Example:
  python scripts/run_lm_eval_psann.py \
    --ckpt runs/lm/wrn_cpu_local/final_model.pt \
    --tokenizer-backend sentencepiece \
    --tokenizer-model-path examples/lm/tokenizer/sample_texts.model \
    --tasks wikitext,lambada_openai,hellaswag,piqa,winogrande \
    --device cuda --limit 1500 --num-fewshot 0 \
    --output eval_out/smoke_quick.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from pathlib import PurePosixPath
import warnings

from lm_eval import evaluator
from lm_eval.api import registry

# Ensure repository root is on sys.path so we can import psann_adapter
import sys
from pathlib import Path as _Path

_here = _Path(__file__).resolve()
_root = _here.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Import adapter and register under a name
from psann_adapter import PSANNLM


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=False, type=str)
    p.add_argument(
        "--hf-repo", required=False, type=str, help="Hugging Face repo id, e.g. user/model"
    )
    p.add_argument(
        "--hf-filename",
        required=False,
        type=str,
        help="Filename inside repo (default: final_model.pt)",
    )
    p.add_argument("--hf-revision", required=False, type=str, help="Branch/tag/commit")
    p.add_argument("--hf-token", required=False, type=str, help="HF token if the repo is private")
    p.add_argument("--device", default="auto", type=str)
    p.add_argument("--tokenizer-backend", default="auto", type=str)
    p.add_argument("--tokenizer-model-path", default=None, type=str)
    p.add_argument(
        "--tokenizer-special-map-path",
        default=None,
        type=str,
        help="Path to special_tokens_map.json for Hugging Face tokenizers backends",
    )
    p.add_argument("--hf-tokenizer-repo", default=None, type=str)
    p.add_argument("--hf-tokenizer-filename", default=None, type=str)
    p.add_argument("--hf-tokenizer-revision", default=None, type=str)
    p.add_argument(
        "--hf-tokenizer-special-map",
        default=None,
        type=str,
        help="Filename within the tokenizer repo for special_tokens_map.json (default: sibling to tokenizer json)",
    )
    p.add_argument("--tasks", default="wikitext", type=str)
    p.add_argument("--limit", default=1500, type=int)
    p.add_argument("--num-fewshot", default=0, type=int)
    p.add_argument("--batch-size", default="auto", type=str)
    p.add_argument("--max-ctx", default=2048, type=int)
    p.add_argument("--max-batch-size", default=8, type=int)
    p.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Apply chat template for tasks that benefit from chat-style formatting",
    )
    p.add_argument(
        "--fewshot-as-multiturn",
        action="store_true",
        help="Render few-shot examples as multi-turn chat",
    )
    p.add_argument("--output", default="eval_out/psann_eval.json", type=str)
    args = p.parse_args()

    # Register locally
    registry.register_model("psann")(PSANNLM)

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not args.ckpt and not args.hf_repo:
        raise SystemExit("Provide --ckpt or --hf-repo to locate the model.")

    # Optionally download tokenizer assets from HF Hub
    tok_model_path = args.tokenizer_model_path
    tok_special_map = args.tokenizer_special_map_path
    if args.hf_tokenizer_repo:
        try:
            from huggingface_hub import hf_hub_download
        except Exception as e:  # pragma: no cover
            raise SystemExit(f"Failed to import huggingface_hub: {e}")

        if tok_model_path is None and args.hf_tokenizer_filename:
            try:
                tok_model_path = hf_hub_download(
                    repo_id=args.hf_tokenizer_repo,
                    filename=args.hf_tokenizer_filename,
                    revision=args.hf_tokenizer_revision,
                )
            except Exception as e:
                raise SystemExit(f"Failed to download tokenizer from HF Hub: {e}")

        if tok_special_map is None:
            special_filename = args.hf_tokenizer_special_map
            if special_filename is None and args.hf_tokenizer_filename:
                special_filename = str(
                    PurePosixPath(args.hf_tokenizer_filename).with_name("special_tokens_map.json")
                )
            if special_filename:
                try:
                    tok_special_map = hf_hub_download(
                        repo_id=args.hf_tokenizer_repo,
                        filename=special_filename,
                        revision=args.hf_tokenizer_revision,
                    )
                except Exception as exc:
                    warnings.warn(
                        f"Could not download special tokens map '{special_filename}' from "
                        f"{args.hf_tokenizer_repo}: {exc}",
                        RuntimeWarning,
                    )

    cfg = {
        "batch_size": args.batch_size,
        "device": args.device,
        "model": "psann",
        "model_args": ",".join(
            [
                (f"ckpt={args.ckpt}" if args.ckpt else ""),
                (f"hf_repo={args.hf_repo}" if args.hf_repo else ""),
                (f"hf_filename={args.hf_filename}" if args.hf_filename else ""),
                (f"hf_revision={args.hf_revision}" if args.hf_revision else ""),
                (f"hf_token={args.hf_token}" if args.hf_token else ""),
                f"tokenizer_backend={args.tokenizer_backend}",
                f"tokenizer_model_path={tok_model_path}" if tok_model_path else "",
                f"tokenizer_special_map_path={tok_special_map}" if tok_special_map else "",
                f"max_ctx={int(args.max_ctx)}",
                f"max_batch_size={int(args.max_batch_size)}",
            ]
        ).strip(","),
        "num_fewshot": int(args.num_fewshot),
        "tasks": task_list,
        "limit": int(args.limit),
    }
    # Optional chat-template flags pass-through (supported in recent lm-eval)
    if args.apply_chat_template:
        cfg["apply_chat_template"] = True
    if args.fewshot_as_multiturn:
        cfg["fewshot_as_multiturn"] = True

    results = evaluator.simple_evaluate(**cfg)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Wrote results -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
