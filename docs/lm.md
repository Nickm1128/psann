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
    positional_encoding="rope",   # set to "alibi" or "sinusoidal" if desired
    sine_params=dict(amp_init=1.0, freq_init=1.0, damp_init=0.01, trainable=True),
)

model.fit(dp, epochs=1, batch_tokens=65_536, lr=2e-4, amp="bf16")
print(model.generate("Once upon a time", top_p=0.9, max_new_tokens=64))
print(model.generate_batch(["hello", "goodnight"], max_new_tokens=32))
```

Minimal End-to-End Example
--------------------------
`examples/lm/minimal_train.py` wires together data prep, training, and generation on CPU. It loads
`examples/lm/sample_texts.txt`, reuses the bundled SentencePiece model at
`examples/lm/tokenizer/sample_texts.model`, repeats the 10-line corpus 64x (~6.5k tokens), trains a
4-layer WaveResNet for 12 epochs (`batch_tokens=512`, fp32), and prints a few completions.

Latest run (`reports/examples/20251107_1750_minimal_train/`) produced:

| prompt            | sample output                                                         |
|-------------------|------------------------------------------------------------------------|
| `hello world`     | `Nrom PSAN-LM mininimal example.`                                     |
| `wave networks`   | `Transformers formers with sine activations can canext toing.`         |
| `sine activations`| `En WaveResNet paths whisper through residual lines.`                 |

Reproduce with:

```
python examples/lm/minimal_train.py \
  --epochs 12 \
  --repeat 64 \
  --out reports/examples/<timestamp>_minimal_train
```

Lower `--repeat` for faster-but-noisier runs; increase it (or epochs) for smoother generations.

Public API Reference
--------------------

The Python surface intentionally mirrors the minimal example above. Everything lives under
`psann.lm` so imports stay stable across releases.

**`psannLMDataPrep`**
- Inputs: iterable of texts or file paths plus tokenizer options (`tokenizer="auto"` resolves in the documented order with `tokenizer_model_path` opt-in).
- Key knobs: `max_length`, `pack_sequences`, `val_split`, `seed`.
- Properties:
  - `vocab_size`: integer vocab derived from the fitted tokenizer.
  - `dataset` / `val_dataset`: lazily constructed :class:`LMDataset` instances.
  - `tokenizer`: the resolved tokenizer object (SentencePiece/HF/simple).
  - `tokenizer_backend`: string literal describing which backend is active.
- Implements `__len__` so you can log dataset sizes in scripts.

**`psannLM`**
- Constructor mirrors the model config: `base`, `d_model`, `n_layers`, `n_heads`, optional `d_mlp`, `vocab_size`, `positional_encoding`, and `sine_params`.
- `.fit(train_data, val_data=None, epochs=..., batch_tokens=..., lr=..., amp=..., ddp=..., **overrides)` wires up :class:`Trainer` under the hood and returns `self` for chaining.
- `.generate(prompt, max_new_tokens=128, top_k=None, top_p=0.9, temperature=1.0, repetition_penalty=None)` runs single-prompt sampling.
- `.generate_batch(prompts, ...)` buckets prompts by length and reuses KV-cache tensors automatically.
- `.save(path)` / `.load(path)` persist configs plus learned weights; the saved positional-encoding + sine settings are restored on load.

See `examples/lm/minimal_train.py` (CPU) and `examples/lm/generate.py` for full runnable snippets that exercise the same surface used in the tests.

Tokenizer Backend Policy
------------------------

`tokenizer="auto"` now resolves to **SentencePiece → Hugging Face tokenizers → simple char**.
The detectors only fall back when the prerequisite package cannot be imported, and the
selected backend is exposed via `psannLMDataPrep.tokenizer_backend`.

Latest comparison (corpus: `examples/lm/sample_texts.txt`, repeat=256, vocab=4096) was captured
with `python scripts/compare_tokenizers.py --repeat 256` and lives at
`reports/tokenizers/20251107_161551/metrics.json`.

| backend       | fit (s) | encode (s) | tokens/sec | tokens/char | model size |
|---------------|--------:|-----------:|-----------:|------------:|-----------:|
| sentencepiece | 0.172   | 0.065      | 1.10M      | 0.619       | 236 KiB    |
| tokenizers    | 0.029   | 0.072      | 0.39M      | 0.239       | 5.7 KiB    |

SentencePiece yields denser tokenisation (≈1.6 chars/token) with higher encode throughput.
The Hugging Face backend trains faster and produces a smaller JSON artifact, making it a good
fallback on CPU-only or minimal environments. To refresh the comparison after changing corpora or
hyperparameters, rerun:

```
python scripts/compare_tokenizers.py --corpus <path> --repeat 256 --vocab-size 4096
```

Artifacts are timestamped under `reports/tokenizers/<UTC_TIMESTAMP>/metrics.json` for auditability.

Positional Encoding Policy
--------------------------

RoPE remains the default positional encoding across both bases. Set `positional_encoding`
to `"rope"`, `"alibi"`, or `"sinusoidal"` via `psannLM(..., positional_encoding="alibi")`,
the CLI YAML (`model.positional_encoding: alibi`), or the low-level transformer configs.

- `"rope"`: rotary embeddings applied inside attention heads (requires even `d_model / n_heads`).
- `"alibi"`: RoPE is disabled and linear attention biases are injected instead; works with any
  head dimension and is preferable for very long contexts or KV-cache heavy decoding.
- `"sinusoidal"`: adds absolute sinusoidal embeddings to token inputs for compatibility with older
  checkpoints; attention uses standard dot-product logits.

ALiBi biases are exercised in `tests/lm/test_transformer_forward.py` to guard against regressions.

KV-cache Fast Path
------------------

`psannLM.generate_batch()` drives generation through PyTorch's KV-cache tensors (`use_cache=True`
and `past_kvs` feeds). On CPU this path already yields a ~13.5x speed-up over the naive loop
that replays the whole prompt per sample:

```
python scripts/benchmark_kv_cache.py --batch-size 8 --prompt-length 96 --max-new-tokens 64
# Artifact: reports/kv_cache/20251107_164826/metrics.json
```

The run above produced 177 tokens/s on the PyTorch fast path versus 13 tokens/s for the naive
loop (512 generated tokens total). Because this keeps latency acceptable for the current release,
we are deferring a fused C++/CUDA kernel until GPU throughput sweeps (GPU-03/TEST-03) show the
PyTorch path is the bottleneck (tracking ticket `KVFAST-01`). Re-run the script with the command
above to refresh the artifact after changing model dimensions or hardware.

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
- `positional_encoding`: `rope` (default) | `alibi` | `sinusoidal`
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

Latest GPU Validation Snapshot
------------------------------
- **Hardware / software:** dual NVIDIA H200 (139.7 GB each), Python 3.12.3, PyTorch 2.8.0+cu128
  (see `reports/tests/20251107_172133/system.json`).
- **Validation bundle:** `reports/gpu/20251107_172205/` covers GPU-01..08 and confirms AMP parity
  (`rel_diff=1.57e-4`), DDP/FSDP parity (`rel_diff=0.0` at world_size=2), KV-cache generation,
  and checkpoint save/load parity.
- **Throughput sweep:** `reports/benchmarks/20251107_015028_gpu_bundle/throughput.csv` records the
  GPU-03 runs. Best results: ResPSANN 614.7k tok/s and WaveResNet 612.6k tok/s at `batch_tokens`
  131k on the H200 pair (see the README in the same directory for the complete table).
- **Memory snapshot:** `reports/benchmarks/20251107_015028_gpu_bundle/memory.json` adds the GPU-04
  gradient-checkpoint measurement (71.8 MiB allocated / 296 MiB reserved, bf16).

Use `scripts/run_cuda_suite.sh` (or `scripts/run_gpu_validation.py --out reports/gpu`) to reproduce
the same battery; the benchmark README under `reports/benchmarks/20251107_015028_gpu_bundle/`
summarizes the key numbers for release/acceptance notes.

Trainable Sine Parameter Ablations
----------------------------------
WaveResNet benefits the most when the sine activation can adapt its amplitude **and** frequency.
We ran an 8-way grid over the `sine_params.learnable` list (frozen vs. every subset of
`["amplitude", "frequency", "decay"]`) using `examples/lm/configs/waveresnet_small.yaml`
(`epochs=2`, `batch_tokens=131072`, bf16) on the shuffled `datasets/lm/tiny_books.txt` shard.
Artifacts (metrics table, JSON summary, bar plot, run notes) live at
`reports/ablations/20251107_1730_sine_params/`.

| label         | learnable set                 | val ppl | tokens/s | delta vs frozen |
|---------------|-------------------------------|--------:|---------:|------------:|
| fixed_all     | `[]`                          | 28.41   | 284k     | +0.00       |
| amp_only      | `["amplitude"]`               | 25.08   | 283k     | -3.33       |
| freq_only     | `["frequency"]`               | 24.77   | 283k     | -3.64       |
| damp_only     | `["decay"]`                   | 27.93   | 284k     | -0.48       |
| amp_freq      | `["amplitude","frequency"]`   | 23.54   | 283k     | -4.87       |
| amp_damp      | `["amplitude","decay"]`       | 24.21   | 283k     | -4.20       |
| freq_damp     | `["frequency","decay"]`       | 22.83   | 282k     | -5.58       |
| all_trainable | `["amplitude","frequency","decay"]` | 22.11 | 282k     | -6.30       |

Key takeaways:
- Allowing both amplitude and frequency to move recovers ~17% lower perplexity versus keeping all
  parameters frozen, with \<1% throughput impact.
- Damping alone does not help, but pairing it with another learnable knob closes the remaining gap.
- Full trainability gives the best validation perplexity (22.11) and is the new documented default.

To reproduce any row, set `model.sine_params.trainable: false` and pass the desired subset via
`model.sine_params.learnable`. When using the CLI you can override inline, e.g.:

```
python -m psann.lm.train.cli \
  --config examples/lm/configs/waveresnet_small.yaml \
  --train.epochs 2 \
  --train.batch_tokens 131072 \
  --model.sine_params.trainable=false \
  --model.sine_params.learnable='[amplitude,frequency]'
```

The bar chart `reports/ablations/20251107_1730_sine_params/sine_param_tradeoffs.png` provides a
quick visual of the validation perplexity deltas for slide decks or release notes.

Test Artifacts
--------------
- Install dev extras (`pip install .[dev]`) to pull in `pytest-json-report`.
- `scripts/run_cuda_suite.sh` wraps `run_cuda_tests.py`, `run_gpu_tests.py`, and `run_gpu_validation.py` so one command exercises the entire CUDA + GPU validation battery.
- `scripts/run_cuda_tests.py` and `scripts/run_gpu_tests.py` automatically emit
  `pytest_report.json` alongside `junit.xml` whenever the plugin is available.
- Artifacts land under `reports/tests/<timestamp>/` with `system.json`, `summary.json`,
  `stdout.log`, and GPU test outputs (if applicable).

Caveats
-------
- The provided examples run on CPU by default; GPU performance requires CUDA-capable hardware.
- SentencePiece → Hugging Face tokenizers → simple char is the exact auto-detection order;
  inspect `psannLMDataPrep.tokenizer_backend` to confirm which backend is live, and expect quality
  drops if the char-level fallback is in use.
- DeepSpeed shims exist in the trainer but are optional; torch DDP/FSDP paths are the focus.
