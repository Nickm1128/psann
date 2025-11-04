# PSANN-LM Module Plan & Working TODO

> Goal: Add a production-ready language modeling module to PSANN with a clean, minimal API:
>
> ```python
> texts = ["test1", "test2"]
> model = psannLM(base="waveresnet")
> train_data = psannLMDataPrep(texts)
> model.fit(train_data)
> ```
>
> Internals should leverage ResPSANN/WaveResNet with trainable sine amplitude, frequency, and damping; scale cleanly to multi-GPU; and be easy to extend.

---

## Progress Tracker (Codex MUST keep updated)

* **Tasks complete:** `59 / 80` - `73.75%`
* **Last edit (UTC):** `2025-11-04 16:45`
* **Editor:** `Codex`
* **Session Notes Summary (1-3 bullet points MAX):**

  * Recomputed checklist (59/80, 73.75%) and updated timestamp.
  * WRN cached conv state + test done; tokenizer/data checked; residual gating + sine ranges.
  * Ready for GPU block; recorded entry in Session History.

> **Codex:**
>
> * On every save, recompute the completed/total counts from all checkboxes in this file and update the percentage.
> * Update the timestamp and append 1–3 bullets to Session Notes Summary.
> * Do not remove historical notes; keep only the latest summary here and move older summaries to the “Session History” section.

---

## Success Criteria

* Clean, minimal **public API** (`psannLM`, `psannLMDataPrep`) with parity across bases (`respsann`, `waveresnet`).
* **Trainable sine parameters** (amplitude, frequency, damping) wired into transformer blocks when base supports it.
* **Tokenizer-agnostic** via plugin (defaults to WordPiece/BPE with sentencepiece or tokenizers).
* **Streaming data loader** for large corpora; supports memory-mapped shards and fast collation.
* **Scaling**: AMP (bf16/fp16), gradient accumulation, gradient checkpointing, DDP, optional DeepSpeed/FSDP.
* **Metrics**: loss, perplexity, tokens/sec, wall-clock, memory; checkpointing and resumable training.
* **Inference**: `generate()` with sampling/top-k/top-p, repetition penalty.
* **Docs & tests**: unit, smoke, and GPU integration tests; examples and README.

---

## Architecture Overview

* **Module path:** `psann/lm/`

  * `psann/lm/models/` – transformer stacks on ResPSANN/WaveResNet
  * `psann/lm/data/` – tokenization, dataset, collation, streaming
  * `psann/lm/train/` – trainer, loops, scaling utils
  * `psann/lm/infer/` – generation utilities
  * `psann/lm/api.py` – `psannLM`, `psannLMDataPrep`
  * `psann/lm/config.py` – typed configs for model/data/train
  * `psann/lm/tests/` – unit & integration tests
  * `examples/lm/` – notebooks and scripts
* **Key idea:** A PSANN-style transformer block with sine-activated MLPs (trainable amplitude/frequency/damping) and optional WaveResNet residual pathways; registry pattern to select base.

---

## Public API Spec (first-class UX)

```python
from psann.lm import psannLM, psannLMDataPrep

texts = ["hello world", "goodnight moon"]
train_data = psannLMDataPrep(texts, tokenizer="auto", max_length=1024, pack_sequences=True)

model = psannLM(
    base="waveresnet",       # or "respsann"
    d_model=768,
    n_layers=12,
    n_heads=12,
    vocab_size=train_data.vocab_size,
    sine_params=dict(amp_init=1.0, freq_init=1.0, damp_init=0.01, trainable=True),
    rope=True,               # rotary embeddings
)

model.fit(
    train_data,
    val_data=None,           # or psannLMDataPrep([...])
    epochs=3,
    batch_tokens=131072,
    lr=2e-4,
    amp="bf16",
    ddp="auto",
)

out = model.generate("Once upon a time", max_new_tokens=128, top_p=0.9)
```

---

## Implementation Plan (High-Level)

1. **Scaffold module** (paths, registries, configs, stubs).
2. **Sine activation core**: shared layer exposing amplitude/frequency/damping.
3. **Transformer blocks** for **ResPSANN** and **WaveResNet** bases.
4. **Tokenizer plugins** (sentencepiece/tokenizers), vocab build/load.
5. **DataPrep** object: chunking, packing, streaming, shard IO, collation.
6. **Trainer**: AMP, grad accumulation, grad clip, LR schedule, checkpointing, logging.
7. **DDP/FSDP/DeepSpeed** hooks; config-driven.
8. **Inference**: fast generation utils.
9. **Tests**, **docs**, **examples**.
10. **Benchmark script** for tokens/s, loss curve on a tiny dataset.

---

## GPU-REQUIRED WORK BLOCK

* [ ] **GPU-01:** Set up environment (CUDA/cuDNN, PyTorch versions) and verify with a 1-step forward/backward on a tiny model.
* [ ] **GPU-02:** AMP sanity check (bf16/fp16) on tiny model; compare loss parity with fp32 on a dummy batch.
* [ ] **GPU-03:** Throughput benchmark on synthetic data for both bases (`respsann`, `waveresnet`); log tokens/s vs batch_tokens.
* [ ] **GPU-04:** Activate gradient checkpointing; measure memory and wall-clock deltas.
* [ ] **GPU-05:** DDP on 2+ GPUs; confirm loss/repro parity with single-GPU.
* [ ] **GPU-06:** Optional DeepSpeed/FSDP hooks for large models.
* [ ] **GPU-07:** Generation smoke test with top-k/top-p sampling; verify no NaNs and reasonable outputs.
* [ ] **GPU-08:** Save/load checkpoints; verify resume produces loss continuity within tolerance.

---

## Work Items

### 0) Project Scaffolding & Docs

* [x] **PH-01:** Create `psann/lm/` scaffolding and `__init__.py` exports for public API.
* [x] **PH-02:** Add `pyproject.toml` updates for new extras: `psann[lm]` installs tokenizer deps.
* [x] **PH-03:** Add `docs/lm.md` page with getting started, config, and examples.
* [x] **PH-04:** Add `examples/lm/minimal_train.py` and `examples/lm/generate.py`.

### 1) Config, Registry, and API Stubs (CPU)

* [x] **CFG-01:** Implement `psann/lm/config.py`:
  * [x] `ModelConfig`, `DataConfig`, `TrainConfig` dataclasses with validation.
* [x] **REG-01:** Implement base registry in `psann/lm/models/registry.py`:
  * [x] Register `"respsann"` and `"waveresnet"`.
* [x] **API-01:** Implement `psannLM` high-level wrapper in `psann/lm/api.py`:
  * [x] `__init__` takes `base`, model kwargs, auto-bridges `vocab_size`, `max_length`.
  * [x] `fit(train_data, val_data=None, **train_kwargs)`.
  * [x] `generate(prompt, **gen_kwargs)`.
  * [x] `save(path)`, `load(path)`.
* [x] **API-02:** Implement `psannLMDataPrep` in `psann/lm/api.py`:
  * [x] Accepts `List[str]` or path(s); builds tokenizer if needed.
  * [x] Exposes `.dataset`, `.vocab_size`, `.tokenizer`, `.pad_id`.
  * [x] Supports validation split and packed sequences.

### 2) Sine Activation Core (CPU)

* [x] **SINE-01:** Create `psann/lm/models/sine.py`:
  * [x] Parametric sine with amplitude/frequency/damping parameters.
  * [x] Optional residual scaling; configurable init ranges.
  * [x] Unit tests for forward/backward, shape, dtype, grad flow.
* [x] **SINE-02:** Export a factory for use inside MLP/ResNets.

### 3) Transformer Stacks on PSANN Bases (CPU)

* [x] **TBLK-01:** Implement shared components:
  * [x] Embedding, rotary/absolute positional encodings, attention (MHA), norm (RMSNorm/LayerNorm).
  * [x] MLP with PSANN sine activation; toggles for GELU comparison.
* [x] **TBLK-02:** Implement **ResPSANNTransformer**:
  * [x] Residual structure using PSANN residual sine MLPs.
  * [x] Config: `n_layers`, `d_model`, `n_heads`, `d_mlp`, `dropout`, sine params.
* [x] **TBLK-03:** Implement **WaveResNetTransformer**:
  * [x] Scaffolding: temporal Conv1d residual block (depthwise) with sine/gelu activation; interleave option (off by default).
  * [x] Add replace-MLP mode and per-layer dilation growth; keep KV-cache path functional.
  * [x] Config parity polish and mask-aware conv state for cached gen.
* [x] **TBLK-04:** Weight init strategy; tests for parameter counts and forward pass.

### 4) Tokenization & Data (CPU)

* [x] **TOK-01:** Tokenizer plugin interface `psann/lm/data/tokenizer.py`:
  * [x] Adapters for `sentencepiece` and `tokenizers` (BPE/Unigram).
  * [x] `auto` mode picks installed backend; can train from corpus or load prebuilt.
* [x] **TOK-02:** Vocab building utilities with min-freq, special tokens, `unk`, `pad`, `bos`, `eos`.
* [x] **TOK-03:** CLI/DataPrep expose `tokenizer_model_path` (load prebuilt tokenizer models).
* [x] **DATA-01:** Dataset and collation in `psann/lm/data/dataset.py`:
  * [x] Sequence packing (contiguous stream, `pack_sequences=True`).
  * [x] Streaming from files (mmap/shards), deterministic shuffling.
  * [x] `batch_tokens` sampler for efficient token-based batching.
* [x] **DATA-02:** Validation dataset splitter; supports fixed seed and stratified split by length.

### 5) Training Loop (CPU baseline; GPU features toggled later)

* [x] **TRN-01:** Implement trainer in `psann/lm/train/trainer.py`:
  * [x] Cross-entropy LM loss (shifted), label smoothing optional.
  * [x] Optimizers: AdamW default; schedulers: cosine w/ warmup.
  * [x] Gradient clipping, accumulation.
  * [x] Checkpoint save/load, best-val tracking.
  * [x] Logging hooks: throughput, loss, perplexity, lr, grad-norm.
* [x] **TRN-02:** CLI helpers `psann/lm/train/cli.py` to run from YAML config.

### 6) Inference & Sampling (CPU ok)

* [x] **INF-01:** `generate()` with greedy, top-k, top-p, temperature, repetition penalty, max_new_tokens, eos stop.
* [x] **INF-02:** Batched generation w/ KV-cache; ensure CPU fallback.
* [x] **INF-03:** Mixed-length batched generation via length bucketing (no masks required).

### 7) Tests (CPU and GPU separation)

* [x] **TEST-01 (CPU):** Unit tests for tokenizer build/load, dataset packing, sine layer math, transformer forward, loss shape.
* [x] **TEST-02 (CPU):** Save/load roundtrip; deterministic seed tests.
* [x] **TEST-04 (CPU):** WRN temporal block interleave/replace forward shape tests.
* [ ] **TEST-03 (GPU):** (Place in GPU block) AMP parity, DDP parity, generation sanity, throughput assertions.

### 8) Documentation & Examples

* [x] **DOC-01:** `docs/lm.md` usage, config table, scaling tips, caveats.
* [x] **DOC-02:** `examples/lm/minimal_train.py` trains on toy corpus.
* [x] **DOC-03:** `examples/lm/generate.py` shows prompt to text.

### 9) Benchmark & Validation (GPU)

* [ ] **BMRK-01:** Tiny corpus (e.g., ~50MB) baseline: loss curve, perplexity target.
* [ ] **BMRK-02:** Throughput table: tokens/s for base configs and batch_tokens variants.
* [ ] **BMRK-03:** Memory profile snapshot under AMP + checkpointing.

---

## Design Notes

* **Parametric Sine:** prefer a numerically stable damping like `exp(-d*|x|)` to avoid exploding outputs; clamp or softplus `d`.
* **Positional Encodings:** default to **RoPE** (rotary) for attention; allow absolute sinusoidal fallback.
* **Sequence Packing:** for efficiency on LM, implement packed contiguous token streams with attention masks per sample.
* **Batching by tokens:** prioritize `batch_tokens` over `batch_size` for stable throughput across sequence lengths.
* **Scaling:** recommend bf16; enable grad checkpointing by default on large models; document tradeoffs.
* **Registry:** keep base selection orthogonal to model dimensions; both bases must satisfy a shared interface.

---

## Example Minimal YAML (for CLI)

```yaml
# examples/lm/configs/waveresnet_small.yaml
model:
  base: waveresnet
  d_model: 512
  n_layers: 8
  n_heads: 8
  d_mlp: 2048
  vocab_size: 32000
  rope: true
  sine_params:
    amp_init: 1.0
    freq_init: 1.0
    damp_init: 0.01
    trainable: true

data:
  sources:
    - path: data/sample_texts.txt
  tokenizer: auto
  max_length: 1024
  pack_sequences: true
  val_split: 0.01
  seed: 1337

train:
  epochs: 1
  batch_tokens: 131072
  lr: 0.0002
  warmup_steps: 2000
  weight_decay: 0.01
  amp: bf16
  grad_clip: 1.0
  grad_accum_steps: 1
  ddp: auto
  checkpoint_dir: runs/lm/wrn_small
  log_interval_steps: 50
  save_interval_steps: 500
```

---

## Codex Operating Instructions (Very Important)

1. **Before starting a session:**
   * Read this file top to bottom.
   * Update **Progress Tracker** timestamp.
   * Write 1–3 bullets in **Session Notes Summary** about intent.

2. **During the session:**
   * Work top-down by sections unless a dependency forces reordering.
   * For any new files, create them with clear headers and docstrings.
   * Keep commits small and scoped to one subsection when possible.

3. **When encountering GPU-required items:**
   * Pause coding.
   * Add a note in **Session Notes Summary** specifying the exact GPU step and prerequisites.
   * Wait for Nick to acknowledge/prepare GPU environment before proceeding to the next GPU item.
   * Once GPU work is green, proceed with the rest of the GPU block continuously.

4. **On every save/commit:**
   * Recompute checklist completion counts and update the **Progress Tracker**.
   * Keep **Session Notes Summary** to 1–3 bullets; move older bullets to **Session History**.

5. **When blocking on ambiguity or missing dependencies:**
   * Add a short “Open Questions” bullet list at the end of this file.
   * Propose defaults and proceed with the safest assumption; mark with `// ASSUMPTION`.

---

## Session History (latest at top)

* [2025-11-04 16:45 UTC] WRN cached conv state+test; tokenizer/data checked; residual gating + sine ranges; counts updated.

* [2025-11-04 15:28 UTC] Streaming dataset, prebuilt tokenizer load, min-freq; weight init; tests; TODO updates.
* [2025-11-04 15:05 UTC] KV-cache + generate_batch; HuggingFace tokenizers backend; sine unit tests.
* [2025-11-04 14:46 UTC] Trainer: cosine LR + label smoothing + grad accumulation/logging; sampling util; docs expanded.
* [2025-11-04 14:22 UTC] Added GELU toggle + sine wiring; validation split; continued RoPE + generate + CLI + checkpointing.
* [2025-11-04 14:16 UTC] Added RoPE attention, generate() sampling, trainer checkpointing/best-val, and CLI YAML.
* [2025-11-04 14:05 UTC] Scaffolded `psann/lm` package, public API, and core stubs.

---

## Open Questions

* [ ] Prefer `sentencepiece` vs `tokenizers` as default backend?
* [ ] Default positional encodings: RoPE everywhere, or allow ALiBi toggle?
* [ ] Do we need a fast C++/CUDA KV-cache path now, or defer to later?

---

## Acceptance Checklist (final sign-off)

* [ ] Public API works as spec and is covered in docs.
* [ ] Trainable sine params visibly affect training (ablations logged).
* [ ] End-to-end example trains on a sample corpus and generates text.
* [ ] GPU block completed with throughput and memory numbers recorded.
* [ ] Tests (CPU+GPU) passing in CI or local matrix.
* [ ] README/docs updated with installation and quickstart.





