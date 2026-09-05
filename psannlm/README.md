# psannlm — PSANN Language Modeling

`psannlm` is a separate distribution using the shared policies and numerical components
in `psann`. Install both with `pip install psann psannlm`; core `psann` does not require LM tooling.

```python
from psannlm import LMArchitectureConfig, LMConfig, PSANNLM, PSANNLMDataPrep, TrainConfig

texts = ["hello world", "goodnight moon", "the quick brown fox jumps over the lazy dog"] * 8
data = PSANNLMDataPrep(texts, tokenizer="simple", max_length=32)
model = PSANNLM(
    config=LMConfig(
        architecture=LMArchitectureConfig.wave(),
        d_model=128, n_layers=2, n_heads=4, vocab_size=data.vocab_size,
    ),
    device="cpu",
)
model.fit(data, train=TrainConfig(epochs=1, batch_tokens=256, lr=1e-3, amp="fp32"))
print(model.generate("hello", max_new_tokens=32, top_p=0.9))
```

`python -m psannlm` is the canonical CLI for `train`, `resume`, `eval`, and `generate`.
For a local YAML run: `python -m psannlm train --config examples/lm/configs/waveresnet_cpu.yaml`.
Streaming training accepts `--architecture` or a canonical JSON/YAML `--model-config`.

Model artifacts use `psannlm.model` schema version 1 and embed canonical configuration,
weights, device and fitted tokenizer state. Trainer artifacts use `psannlm.trainer` and
retain optimizer, scaler, scheduler, counters and RNG state. `PSANNLM.load` accepts model
artifacts; `psannlm.persistence.load_lm_checkpoint` reconstructs either artifact kind.
Old unversioned model and trainer files remain readable; old trainer/raw-weight files
require the model options that were not stored in those files.

Lowercase `psannLM`/`psannLMDataPrep`, `Trainer`, legacy base names, flat fitting, and the
`psannlm.train` and `psannlm.lm.train.cli` commands remain warning compatibility adapters
through 0.x. New code uses immutable `LMConfig` and `TrainConfig`.
See [the LM guide](../docs/lm.md) for policies, migration, and runtime details.
