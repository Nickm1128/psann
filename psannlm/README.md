# PSANN-LM

Language-model tasks for PSANN, packaged separately as `psannlm`. This checkout prepares version **0.13.0**, requiring `psann>=0.13.0`; publication is a separate operation. The existing `v1.0.0` repository tag is historical and is unchanged.

Install core first, then this directory with `python -m pip install ./psannlm` from the repository root. PyTorch, NumPy, SentencePiece, tokenizers, datasets, Hugging Face Hub, and PyYAML are declared runtime dependencies.

Use `PSANNLM`, `PSANNLMDataPrep`, `LMConfig`, and `LMArchitectureConfig` from `psannlm`. Select one of four architecture kinds: transformer, residual, wave, or geometric-sparse. Spectral residual is a nested residual policy. Train with `model.fit(data, train=TrainConfig(...))`, generate with `model.generate(...)`, and persist with `save`/`load`.

```sh
python -m psannlm --help
python -m psannlm train --help
python -m psannlm generate --help
```

The [LM guide](https://github.com/psann-project/psann/blob/main/docs/lm.md) covers a complete small training example, YAML, tokenizer identity, and schema-v1 model/trainer checkpoints. The repository [migration guide](https://github.com/psann-project/psann/blob/main/docs/migration.md) documents 0.x compatibility. Core-only installations do not contain this package; LM artifacts contain no core package files.
