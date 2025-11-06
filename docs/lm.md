PSANN-LM
========

PSANN-LM packages a production-ready language-modeling stack on top of PSANN bases (ResPSANN
and WaveResNet). The public API keeps a minimal surface while still exposing tokenizer,
data, training, and inference utilities.

Quickstart
----------

```python
from psann.lm import psannLM, psannLMDataPrep

texts = ["hello world", "goodnight moon"]
dp = psannLMDataPrep(
    texts,
    tokenizer="auto",             # sentencepiece -> tokenizers -> char fallback
    tokenizer_model_path=None,
    max_length=256,
    pack_sequences=True,
)

model = psannLM(
    base="waveresnet",
    d_model=512,
    n_layers=8,
    n_heads=8,
    vocab_size=dp.vocab_size,
    rope=True,
    sine_params=dict(amp_init=1.0, freq_init=1.0, damp_init=0.01, trainable=True),
)

model.fit(dp, epochs=1, batch_tokens=65_536, lr=2e-4, amp="bf16")
print(model.generate("Once upon a time", top_p=0.9, max_new_tokens=64))
print(model.generate_batch(["hello", "goodnight"], max_new_tokens=32))
```

Quickstart (CLI, CPU)
---------------------

Run a minimal end-to-end training on CPU with a tiny sample corpus:

```
python -m psann.lm.train.cli --config examples/lm/configs/waveresnet_cpu.yaml
```

This uses `examples/lm/sample_texts.txt` and disables AMP/DDP for a fast local sanity check.

RunPod One-Sweep
----------------

After cloning on a GPU pod with CUDA-ready PyTorch installed:

```
pip install -e .[dev,lm]
chmod +x scripts/next_gpu_batch.sh
./scripts/next_gpu_batch.sh
```

This will:
- Run the full GPU validation suite into `reports/gpu/<ts>/`.
- Run throughput sweeps (GPU-03) at 65k/131k/262k tokens.
- Run gradient-checkpointing/memory (GPU-04) and record memory stats.
- Create a synthetic `datasets/lm/tiny_books.txt` (~50MB) if missing and run the tiny-corpus benchmark.
- Aggregate into `reports/benchmarks/<ts>/` with `throughput.csv`, `memory.json`, parsed `metrics.csv` and `metrics.json`.
- If `matplotlib` is installed (e.g., `pip install .[viz]`), a `loss_curve.png` will also be emitted.

Configuration
-------------

**Model (`psann.lm.config.ModelConfig`)**
- `base`: `waveresnet` | `respsann`
- `d_model`, `n_layers`, `n_heads`, `d_mlp`
- `vocab_size`: override (defaults to data prep vocab)
- `rope`: enable rotary embeddings
- `sine_params`: amplitude/frequency/damping settings

**Data (`psann.lm.config.DataConfig`)**
- `tokenizer`: `auto` | `simple` | `sentencepiece` | `tokenizers`
- `tokenizer_model_path`: optional SentencePiece `.model` or Hugging Face `.json`
- `max_length`: chunk length
- `pack_sequences`: contiguous stream packing
- `val_split`: float fraction for validation
- `seed`: RNG for shuffling/splitting

**Train (`psann.lm.config.TrainConfig`)**
- `epochs`, `batch_tokens`, `lr`, `warmup_steps`
- `weight_decay`, `label_smoothing`, `grad_clip`, `grad_accum_steps`
- `amp`: `bf16` | `fp16` | `fp32`
- `ddp`: `auto` | `on` | `off` (wraps torch.distributed)
- `checkpoint_dir`, `log_interval_steps`, `save_interval_steps`

Scaling Tips
------------
- Prefer `bf16` for stability and speed; fall back to `fp32` for debugging.
- Size `batch_tokens * grad_accum_steps` to fit GPU memory; gradient checkpointing is
  available through the trainer.
- Enable `pack_sequences=True` for throughput on smaller corpora.
- When running multi-GPU, launch via `torchrun` so the trainer can auto-initialize DDP/FSDP.

Benchmarks & Reporting
----------------------
- Benchmark targets live in `benchmarks/lm_plan.md` and mirror the TODO list items:
  - **BMRK-01** (tiny corpus baseline): ~50 MB text shard, track loss + perplexity curves,
    and log results under `reports/benchmarks/<timestamp>/loss_curve.png`.
  - **BMRK-02** (throughput table): measure tokens/sec for `{base} x {batch_tokens}` grids
    using `scripts/run_gpu_validation.py --only GPU-03`.
  - **BMRK-03** (memory snapshot): run with `grad_checkpoint=True`, capture
    `torch.cuda.max_memory_allocated()` + wall time, and store in
    `reports/benchmarks/<timestamp>/memory.json`.
- `scripts/next_gpu_batch.sh` queues the full validation suite, throughput-only, checkpoint-only,
  and tiny-corpus benchmark runs in one go once GPUs are available.
- Tiny corpus YAML config lives at `examples/lm/configs/tiny_corpus_benchmark.yaml`.
- `scripts/run_gpu_validation.py --out reports/gpu` produces standardized GPU reports
  (loss parity, throughput, checkpoint checks) for regression tracking.

Test Artifacts
--------------
- Install dev extras (`pip install .[dev]`) to pull in `pytest-json-report`.
- `scripts/run_cuda_tests.py` and `scripts/run_gpu_tests.py` automatically emit
  `pytest_report.json` alongside `junit.xml` whenever the plugin is available.
- Artifacts land under `reports/tests/<timestamp>/` with `system.json`, `summary.json`,
  `stdout.log`, and GPU test outputs (if applicable).

Caveats
-------
- The provided examples run on CPU by default; GPU performance requires CUDA-capable hardware.
- SentencePiece or Hugging Face tokenizers are auto-detected; falling back to the built-in
  char tokenizer may reduce quality on large corpora.
- DeepSpeed shims exist in the trainer but are optional; torch DDP/FSDP paths are the focus.
