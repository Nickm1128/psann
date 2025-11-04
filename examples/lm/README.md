PSANN-LM Examples
=================

Minimal usage
-------------

```
from psann.lm import psannLM, psannLMDataPrep

texts = ["hello world", "goodnight moon"]
data = psannLMDataPrep(texts, tokenizer="auto", max_length=256)
model = psannLM(base="waveresnet", d_model=256, n_layers=4, n_heads=4, vocab_size=data.vocab_size)

model.fit(data, epochs=1, batch_tokens=4096, lr=1e-3)
print(model.generate("Once upon a time", max_new_tokens=32, top_p=0.9))
```
