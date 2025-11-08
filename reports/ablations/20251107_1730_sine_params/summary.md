# Trainable Sine Parameter Ablation

- Timestamp: 2025-11-07T17:30:00Z
- Corpus: datasets/lm/tiny_books.txt (50 MB)
- Config: waveresnet small (d_model=768, n_layers=12, bf16, batch_tokens=131072, epochs=2)
- Command: python -m psann.lm.train.cli --config examples/lm/configs/waveresnet_small.yaml --train.epochs 2 --train.batch_tokens 131072

Key takeaways:
1. Letting amplitude **and** frequency learn cuts validation perplexity by 17% versus frozen parameters.
2. Damping must pair with another trainable knob to avoid regressions; by itself it hovers near the baseline.
3. Fully trainable sine parameters produce the best perplexity (22.11) with <1% throughput cost.

Refer to `metrics.csv`/`metrics.json` in this directory for the raw grid.
