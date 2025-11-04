PSANN‑LM
========

Production‑ready language modeling for PSANN with a clean public API and pluggable bases (ResPSANN, WaveResNet).

Quickstart
----------

```
from psann.lm import psannLM, psannLMDataPrep

texts = ["hello world", "goodnight moon"]
dp = psannLMDataPrep(
    texts,
    tokenizer="auto",              # or "sentencepiece" / "tokenizers" / "simple"
    tokenizer_model_path=None,      # optional: load a prebuilt tokenizer model
    max_length=256,
    pack_sequences=True,
)

model = psannLM(base="waveresnet", d_model=512, n_layers=8, n_heads=8,
                vocab_size=dp.vocab_size, rope=True,
                sine_params=dict(amp_init=1.0, freq_init=1.0, damp_init=0.01, trainable=True))

model.fit(dp, epochs=1, batch_tokens=65536, lr=2e-4, amp="bf16")
print(model.generate("Once upon a time", top_p=0.9, max_new_tokens=64))

# Mixed-length batch generation (length bucketing, no masks required)
outs = model.generate_batch(["hello", "goodnight"], max_new_tokens=32, top_p=0.9)
print(outs)
```

Configuration
-------------

Model (``psann.lm.config.ModelConfig``)
- base: ``waveresnet`` | ``respsann``
- d_model: hidden size (int)
- n_layers: transformer layers (int)
- n_heads: attention heads (int)
- d_mlp: MLP hidden size (default ``4*d_model``)
- vocab_size: override (default inferred from data)
- rope: use rotary embeddings (bool)
- sine_*: trainable sine parameters (amplitude/frequency/damping/trainable)

Data (``psann.lm.config.DataConfig``)
- tokenizer: ``auto`` | ``simple`` | ``sentencepiece`` | ``tokenizers``
- tokenizer_model_path: optional prebuilt tokenizer (SentencePiece ``.model`` or HF ``.json``)
- max_length: sequence length for training chunks
- pack_sequences: pack documents into a contiguous stream
- val_split: optional validation fraction
- seed: RNG seed for splits/shuffle

Train (``psann.lm.config.TrainConfig``)
- epochs: number of epochs
- batch_tokens: approximate tokens per micro‑batch
- lr: base learning rate
- warmup_steps: LR warmup steps (cosine schedule)
- weight_decay: AdamW weight decay
- label_smoothing: cross‑entropy label smoothing in [0,1)
- grad_clip: max global grad‑norm (0 disables)
- grad_accum_steps: gradient accumulation steps
- amp: ``bf16`` | ``fp16`` | ``fp32`` | ``none`` (placeholder)
- ddp: ``auto`` | ``on`` | ``off`` (placeholder)
- checkpoint_dir: save directory
- log_interval_steps: logging cadence (optimizer steps)
- save_interval_steps: checkpoint cadence (optimizer steps)

Scaling Tips
------------
- Prefer ``bf16`` for stability and speed; keep ``fp32`` for debugging.
- Set ``batch_tokens`` so that ``batch_tokens * grad_accum_steps`` fits memory.
- Use ``pack_sequences=True`` for better throughput on small corpora.
- Enable validation (``val_split>0``) to track best checkpoint automatically.

Caveats
-------
- Current trainer is CPU‑only; AMP/DDP hooks are placeholders.
- Tokenizer ``auto`` requires ``sentencepiece`` installed, else falls back to a simple char tokenizer.
- KV‑cache and batched generation are planned; greedy/top‑k/top‑p are available.
