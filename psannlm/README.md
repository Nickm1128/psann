# PSANN-LM

Language-model tasks for PSANN, packaged separately as `psannlm`. Version **2.0.1** requires `psann>=2.0.1` and uses the typed task and architecture configuration API as its authoritative public surface.

Install core first, then this directory with `python -m pip install ./psannlm` from the repository root. PyTorch, NumPy, SentencePiece, tokenizers, datasets, Hugging Face Hub, and PyYAML are declared runtime dependencies.

Use `PSANNLM`, `PSANNLMDataPrep`, `LMConfig`, and `LMArchitectureConfig` from `psannlm`. Select one of four architecture kinds: transformer, residual, wave, or geometric-sparse. Spectral residual is a nested residual policy. Train with `model.fit(data, train=TrainConfig(...))`, generate with `model.generate(...)`, and persist with `save`/`load`.

```sh
python -m psannlm --help
python -m psannlm train --help
python -m psannlm generate --help
```

The [LM guide](https://github.com/Nickm1128/psann/blob/main/docs/lm.md) covers a complete small training example, YAML, tokenizer identity, and schema-v1 model/trainer checkpoints. The repository [migration guide](https://github.com/Nickm1128/psann/blob/main/docs/migration.md) documents compatibility routes for older applications and artifacts. Core-only installations do not contain this package; LM artifacts contain no core package files.
