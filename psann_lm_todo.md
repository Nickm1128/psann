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

* **Tasks complete:** `73 / 84` - `86.90%`
* **Last edit (UTC):** `2025-11-07 01:36`
* **Editor:** `Codex`
* **Session Notes Summary (1-3 bullet points MAX):**
  * Added `scripts/run_outstanding_gpu_tests.sh` for combined GPU validation + GPU-03 sweeps with logs.
  * Aggregation now auto-writes throughput/memory under tagged benchmark dirs to simplify pull-backs.
  * Documented RunPod commands for executing the new script on the pod.

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

* [x] **GPU-01:** Set up environment (CUDA/cuDNN, PyTorch versions) and verify with a 1-step forward/backward on a tiny model.
  - Result (20251104_232550): OK, elapsed ≈1.85s on L40S.
* [x] **GPU-02:** AMP sanity check (bf16/fp16) on tiny model; compare loss parity with fp32 on a dummy batch.
  - Result: bf16 vs fp32 rel_diff ≈ 0.001243 (ok), torch 2.9.0+cu128 on L40S.
* [x] **GPU-03:** Throughput benchmark on synthetic data for both bases (`respsann`, `waveresnet`); log tokens/s vs batch_tokens.
  - Result (multi-runs 20251105_21xx): respsann � 279�293k tok/s; waveresnet � 278�293k tok/s (20,480 tokens; B=4, T=256).
* [x] **GPU-04:** Activate gradient checkpointing; measure memory and wall-clock deltas.
  - Implemented: model-level checkpointing toggled via Trainer config (`grad_checkpoint=True`).
  - Tests: added unit tests for forward/backward with checkpointing on both bases.
  - Runner: GPU-04 tiny fit elapsed_s � 0.057�0.067 in full runs; isolated benches reported � 1.57�1.62s (different batch), see reports/gpu/20251105_213755,_221656,_222454.
* [x] **GPU-05:** DDP on 2+ GPUs; confirm loss/repro parity with single-GPU.
  - Result (20251105_204844, 2x L40S via `torchrun`): single vs DDP loss 3.999884 with `rel_diff=0.0`; deterministic seeding + DistributedSampler wiring merged.
* [x] **GPU-06:** Optional DeepSpeed/FSDP hooks for large models.
  - Result: torch FSDP wrapper validated in same run (loss 3.999884, `rel_diff=0.0`, world_size=2); DeepSpeed shim stands ready but not required for current configs.
* [x] **GPU-07:** Generation smoke test with top-k/top-p sampling; verify no NaNs and reasonable outputs.
  - Result (20251105_204844): length=19; sample: `hqixqtqixqqxxzqxqut`.
* [x] **GPU-08:** Save/load checkpoints; verify resume produces loss continuity within tolerance.
  - Result (20251105_204844): params_equal=True, gen_equal=True; checkpoint at `reports/gpu/20251105_204844/checkpoints/lm.pt`.

---

## Work Items

### Immediate Next Steps

- [x] Full CUDA suite green (`reports/tests/20251105_002930`); push artifacts.
- [x] Run GPU validation for GC metrics: `python scripts/run_gpu_validation.py --out reports/gpu`; inspected GPU-04 elapsed/memory in `reports/gpu/20251105_204844`.
- [x] (Optional) Install `pytest-json-report` in the pod to produce `pytest_report.json` alongside `junit.xml` (via `pip install .[dev]` or `python -m pip install pytest-json-report`).
- [x] Begin DDP baseline (GPU-05): torchrun init, DistributedSampler, grad sync, per-rank seeding now live; GPU validation 20251105_204844 covers parity.

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

* [x] **BMRK-01:** Tiny corpus (e.g., ~50MB) baseline: loss curve, perplexity target.
  - Plan documented in `benchmarks/lm_plan.md` (dataset: `datasets/lm/tiny_books.txt`, run `python -m psann.lm.train.cli --config examples/lm/configs/tiny_corpus_benchmark.yaml`, record `loss_curve.png` + `metrics.json`).
  - Status: artifacts verified at `reports/benchmarks/20251106_140525/` (metrics.csv/json + loss_curve.png).
  - Metrics JSON path: `reports/benchmarks/20251106_140525/metrics.json`.
* [ ] **BMRK-02:** Throughput table: tokens/s for base configs and batch_tokens variants.
  - Use `scripts/run_gpu_validation.py --only GPU-03 --out reports/gpu` (included in `scripts/next_gpu_batch.sh`); aggregate into `reports/benchmarks/<ts>/throughput.csv`.
  - Status: aggregated to `reports/benchmarks/20251106_140525/throughput.csv`; best tokens/s respsann=293354.05 (ts=20251105_221652), waveresnet=292731.23 (ts=20251105_213736). 262k sweep pending on GPU.
  - RunPod command: `./scripts/run_outstanding_gpu_tests.sh` to run final validation + GPU-03 sweeps with logs under `reports/benchmarks/<tag>/`.
* [x] **BMRK-03:** Memory profile snapshot under AMP + checkpointing.
  - Capture `torch.cuda.max_memory_allocated()` + elapsed from GPU-04 run (also wired via `scripts/next_gpu_batch.sh`); see `benchmarks/lm_plan.md` for expected `memory.json` schema.
  - Result: memory.json at `reports/benchmarks/20251106_124936/memory.json` (max_alloc=24.01MB, max_reserved=28.0MB).

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

* [2025-11-07 01:31 UTC] Verified BMRK-01 artifacts at `reports/benchmarks/20251106_140525/`, aggregated throughput (best respsann=293354.05, waveresnet=292731.23); 262k GPU-03 sweep still pending.
* [2025-11-06 13:15 UTC] Closed BMRK-03; prepped BMRK-01 (tiny corpus synthesized, metrics.csv present; plot/metrics.json next).
* [2025-11-06 13:08 UTC] GPU-04 memory (20251106_130503) OK; 65k throughput (20251106_124912) ~273.4k tok/s; full suite (20251106_124858) green.
* [2025-11-06 12:13 UTC] CPU LM tests passed (18/18), ran minimal_train on CPU, and added CPU YAML + sample texts for local CLI run before GPU.

* [2025-11-06 13:15 UTC] Closed BMRK-03 (memory snapshot at `reports/benchmarks/20251106_124936/memory.json`); prepped BMRK-01 (tiny corpus + metrics.csv, needs loss_curve.png + metrics.json); throughput 65k row recorded; 131k/262k sweeps pending.
* [2025-11-06 12:42 UTC] RunPod prep: throughput sweeps clamp B/T, AMP warnings resolved, tiny benchmark YAML fixed.
* [2025-11-05 22:28 UTC] GPU runs 20251105_213736/_221637/_222307 full green; throughput-only and checkpoint-only runs captured (GPU-03 ~278-293k tok/s; GPU-04 ~0.057-0.067s full, ~1.57-1.62s isolated).

* [2025-11-05 21:18 UTC] Queued next GPU batch (run_gpu_validation --only, new tiny corpus config, next_gpu_batch.sh); docs/plan updated.
* [2025-11-05 21:13 UTC] Added CPU trainer + data-boundary tests; targeted pytest slice (tokenizer+trainer) green with json-report plugin.
* [2025-11-05 21:07 UTC] Documented benchmark plan/docs refresh; pytest-json-report now in dev extras and optional reporting task closed.
* [2025-11-05 20:52 UTC] GPU validation 20251105_204844 (2x L40S): GPU-01..08 green, DDP/FSDP rel_diff=0, throughput 281k tok/s, checkpoints at `reports/gpu/20251105_204844`.
* [2025-11-05 20:35 UTC] Residual alpha params converted to 1D for FSDP; GPU validation/DDP left pending until hardware is available; pytest slice (psann_nn + conv_nets) green.
* [2025-11-05 17:46 UTC] GPU validation 20251105_174654 (2x RTX 4090): GPU-05/06 rel_diff=0; GPU-03 throughput dip noted.
* [2025-11-05 17:28 UTC] GPU validation 20251105_172515 (2x RTX 4090): GPU-05 parity ok; GPU-06 rel_diff=0.045 pending seed fix; TODO updated.
* [2025-11-05 16:41 UTC] RunPod GPU validation 20251105_155927 (2x L4): GPU-05/06 SIGABRT; patched `scripts/run_gpu_validation.py` with NCCL env fallbacks, deterministic seeding, and mp.spawn error guards.
* [2025-11-05 13:08 UTC] GPU validation 20251105_130530 (2x L40S, torch 2.8.0+cu128): GPU-01..08 ok; GPU-05/06 DDP+FSDP losses 3.999884, checkpoint at `reports/gpu/20251105_130530`.
* [2025-11-05 00:33 UTC] CUDA suite 20251105_002930: 170 passed, 1 skipped; green. L40S; torch 2.9.0+cu128; artifacts at `reports/tests/20251105_002930`.
* [2025-11-05 00:24 UTC] CUDA test suite 20251105_001704: 170 tests, 3 failures (json import), 1 skipped; L40S; torch 2.9.0+cu128; artifacts at `reports/tests/20251105_001704`.

* [2025-11-04 23:47 UTC] GPU smoke 20251104T234003Z: 5/5 passed; torch 2.9.0+cu128; 1 GPU; AMP fp16/bf16 OK; artifacts at `outputs/gpu_tests/20251104T234003Z`.

* [2025-11-04 23:27 UTC] GPU report 20251104_232550: GPU-01/02/03/07/08 OK; 04/05/06 skipped; throughput ≈225k tok/s; AMP bf16 rel_diff≈0.00124; TODO and counts updated.

* [2025-11-04 23:21 UTC] GPU report 20251104_231753: AMP parity ok (bf16 rel_diff≈0.000399), throughput ~226k tok/s, generation smoke ok; GPU-01/08 errors (multinomial num_samples=0); TODO updated.

* [2025-11-04 19:42 UTC] Full suite green (162 passed, 1 skipped); fixed LM save/load device; CUDA auto-select in trainer/generation; prepped to run GPU block on pod.

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

// ASSUMPTION � Proposed defaults for local work:
- Default tokenizer backend: `sentencepiece` when available; fall back to `tokenizers`.
- Positional encodings: RoPE by default; ALiBi toggle can be added later if needed.
- KV-cache path: keep PyTorch-only for now; defer C++/CUDA fast-path until after GPU benchmarks.

---

## Acceptance Checklist (final sign-off)

* [ ] Public API works as spec and is covered in docs.
* [ ] Trainable sine params visibly affect training (ablations logged).
* [ ] End-to-end example trains on a sample corpus and generates text.
* [ ] GPU block completed with throughput and memory numbers recorded.
* [ ] Tests (CPU+GPU) passing in CI or local matrix.
* [ ] README/docs updated with installation and quickstart.



