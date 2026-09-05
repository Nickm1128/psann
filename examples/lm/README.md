PSANN-LM Examples
=================

Minimal usage
-------------

```
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
