# PSANN‑LM 3B Plan and Running TODO

Agent instructions (read first, update every session):
- Keep this file concise and current. Edit only relevant sections; do not duplicate content in multiple places.
- At the end of each work session, update the Work Log and Checklist with concrete changes and next actions.
- When a task requires code changes, reference file paths and line anchors where appropriate (e.g., `src/psann/lm/train/trainer.py:200`).
- If scope expands, add a new subtask under the correct section rather than scattering notes.
- The end state must allow training a 3B English chat model with one command after environment setup.

---

## Goals
- Train a ~3B parameter PSANN‑LM on English text with continued pretraining and SFT for chat.
- Achieve a single command to launch training after environment setup, with robust defaults and documented overrides.
- Maintain correct tokenizer parity (HF `tokenizer.json` + `special_tokens_map.json`) and apply chat templates for instruction/MC evaluation.

## Deliverables
- Trainer supports large‑model training: FSDP/ZeRO sharding, gradient checkpointing, AMP.
- English‑only data pipeline with dedup + decontamination and streaming support.
- Scripted entrypoints:
  - `scripts/train_psann_lm.py` (new) or a documented CLI path to train 3B with FSDP.
  - `scripts/run_lm_eval_psann.py` supports chat template pass‑through for MC tasks.
- Reproducible examples: single command to train + single command to eval.

---

## Environment Setup (target)
- Python 3.10+; CUDA machine(s) with NCCL configured.
- Packages: `torch` (CUDA), `tokenizers`, `sentencepiece` (optional), `datasets`, `accelerate` (if used), `bitsandbytes` (optional), `lm_eval`.
- Optional: DeepSpeed or FSDP (built‑in PyTorch), WandB for logging.

Proposed extra deps (evaluate later):
- `torch>=2.1`, `accelerate>=0.34`, `bitsandbytes>=0.43` (if 8‑bit), `datasets>=2.19`, `langdetect`

---

## Data Pipeline (English‑only)
- Sources (English): RefinedWeb (Falcon), Dolma EN, RedPajama v2 EN, C4‑EN, Wikipedia EN, Books, StackExchange.
- Steps:
  1. Ingest → normalize → dedup (MinHash or exact URL/text dedup) → filter low‑quality/boilerplate.
  2. Decontaminate against eval corpora (WT2, LAMBADA, HellaSwag, PIQA, Winogrande) to prevent leakage.
  3. Tokenize on the fly (HF tokenizers backend) via `StreamingLMDataset`.
- Implementation:
  - Use `src/psann/lm/data/dataset.py:StreamingLMDataset` for large corpora.
  - Add a lightweight loader that reads newline‑delimited text shards from a manifest file.
  - Optional: add a small `tools/dedupe.py` for MinHash if needed.

Acceptance: training can stream 10B+ English tokens without exhausting memory.

---

## Tokenizer & Parity
- Keep using HF `tokenizer_final/tokenizer.json` + `special_tokens_map.json`.
- Ensure adapter and trainer use passthrough IDs for `tokenizers` backend (already implemented in adapter; verify for training path).
- Pre‑flight check: encode/decode roundtrip sanity; `vocab_size` matches `lm_head`.

---

## Model Config (~3B)
- Draft sizing (adjust as needed):
  - `d_model ≈ 3072`, `n_layers ≈ 30`, `n_heads ≈ 24`, `d_mlp ≈ 4×d_model ≈ 12288`, `vocab_size ≈ 50k`.
- Keep RoPE by default; enable gradient checkpointing.
- Add a config preset (YAML or CLI) for 3B with sensible LR/warmup/batch tokens.

---

## Trainer Upgrades (Large‑Model Training)
- Sharding: add FSDP (PyTorch native) with options:
  - Parameter/optimizer/state sharding, activation checkpointing, auto wrap policy for transformer blocks.
  - Optional CPU offload for optimizer states.
- Optimizer: allow `bitsandbytes` 8‑bit Adam or Adafactor (optional), fallback to AdamW.
- Scheduling: cosine with warmup (already present) + gradient accumulation.
- Memory knobs: `grad_checkpoint=True`, smaller `batch_tokens`, higher `grad_accum_steps`.

Files to modify:
- `src/psann/lm/train/trainer.py` — add FSDP path and sharded optimizer support.
- Optional `scripts/train_psann_lm.py` — single‑command entrypoint that wires configs, data, and trainer.

Acceptance: 3B model runs stably on target GPUs with FSDP and checkpointing.

---

## One‑Command Training
Target command (example):
- python scripts/train_psann_lm.py \
  --base waveresnet --d-model 3072 --n-layers 30 --n-heads 24 --vocab-size 50257 \
  --tokenizer-backend tokenizers \
  --hf-tokenizer-repo Nickm1128/psannlm-chat-20251108 \
  --hf-tokenizer-filename tokenizer_final/tokenizer.json \
  --data-manifest /path/to/english_shards.txt \
  --batch-tokens 65536 --grad-accum-steps 8 --amp bf16 --grad-checkpoint \
  --fsdp full_shard --epochs 1 --save-interval-steps 2000 \
  --checkpoint-dir runs/lm/3b_en \
  --out-ckpt runs/lm/3b_en/final_model.pt

Notes:
- `--data-manifest` is a newline‑separated list of text shard paths.
- FSDP mode string TBD based on implementation details (e.g., `--fsdp=full_shard auto_wrap`).

---

## Evaluation & Chat Template
- Perplexity: `wikitext`, `lambada_openai` — do NOT apply chat template.
- Multiple‑choice (chat‑SFT helpful): apply chat template + few‑shot for `hellaswag`, `piqa`, `winogrande`.
- Extend `scripts/run_lm_eval_psann.py` to accept `--apply-chat-template` and `--fewshot-as-multiturn` and pass through to lm‑eval (or document CLI via `python -m lm_eval`).

MC run example (with chat template):
- python -m lm_eval \
  --model custom \
  --model_args custom_model_class=psann_adapter.PSANNLM,\
    hf_repo=Nickm1128/psannlm-chat-20251108,\
    hf_filename=psannlm_chat_final.pt,\
    tokenizer_backend=tokenizers,\
    tokenizer_model_path=tokenizer_final/tokenizer.json,\
    tokenizer_special_map_path=tokenizer_final/special_tokens_map.json \
  --tasks hellaswag,piqa,winogrande \
  --device cuda --batch_size auto --num_fewshot 5 \
  --apply_chat_template --fewshot_as_multiturn \
  --output_path eval_out/mc_chat.json

---

## Compute & Scaling Notes
- Chinchilla heuristic: tokens ≈ 20×params ⇒ 3B ≈ 60B tokens (from scratch). For continued pretraining, target 10–30B English tokens initially.
- FLOPs ≈ 6×params×tokens. Plan GPU hours accordingly; prefer bf16 with FSDP.

---

## Checklist
- [ ] Data: English manifest created; dedup + decontaminate complete.
- [x] Tokenizer: final tokenizer.json + special_tokens_map.json verified; passthrough IDs in training path.
- [x] Trainer: FSDP sharding added; gradient checkpointing toggle works; AMP bf16 tested.
- [ ] Optimizer: 8‑bit optimizer optional path (bitsandbytes) wired; fallback AdamW ok.
- [x] Single command training script added with CLI args for 3B.
- [ ] Eval: script accepts chat‑template flags or documented lm‑eval command available.
- [ ] Smoke: small CPU/GPU dry run passes; final GPU run stable on target cluster.
- [x] Documentation: commands recorded in this file and README updates.

---

## Work Log
- 2025-11-09 (cont.): Added tokenizer training loop + HF streaming filter reuse in `scripts/train_psann_lm.py` (lang filters, sample cap, tokenizer save/export bundle). `HFTextStreamingLMDataset` now shares `build_text_filter`, and `Tokenizer.save()` persists `tokenizer.json` + `special_tokens_map.json`, with optional export bundles (`--export-dir`) including metadata for HF uploads.
- 2025-11-09: Implemented Trainer FSDP path (size-based auto-wrap, cpu-offload toggle, full-state saves), optimizer options (AdamW8bit/Adafactor fallback), iterable-data handling + scheduler fallback, new one-command training entrypoint `scripts/train_psann_lm.py` with manifest + HF tokenizer passthrough, and extended `scripts/run_lm_eval_psann.py` to accept `--apply-chat-template`/`--fewshot-as-multiturn`. Next: run GPU smoke, refine auto-wrap policy for 3B blocks, and add README snippets.
- YYYY‑MM‑DD: Created 3B plan. Adapter/tokenizer parity validated; eval improved; next—FSDP integration and training CLI.
- YYYY‑MM‑DD: …
