# Language modeling

`psannlm` is a separate distribution. Install core and then the LM package from this checkout with `python -m pip install -e .` and `python -m pip install ./psannlm`. The candidate metadata is 0.13.0 for both; LM requires `psann>=0.13.0`. NumPy, PyTorch, SentencePiece, tokenizers, datasets, Hugging Face Hub, and PyYAML are direct LM runtime dependencies. Optional evaluation integrations have their own dependencies.

## Train, generate, and save

```python
import torch
from psannlm import PSANNLM, PSANNLMDataPrep, LMConfig, LMArchitectureConfig, TrainConfig

data = PSANNLMDataPrep(
    ["waves learn useful patterns in small sequences " * 16],
    tokenizer="simple", max_length=8,
)
model = PSANNLM(
    config=LMConfig(
        architecture=LMArchitectureConfig.wave(),
        d_model=16, n_layers=1, n_heads=2, d_mlp=32,
        vocab_size=data.vocab_size,
    ),
    device="cpu",
)
model.fit(data, train=TrainConfig(
    epochs=1, steps_per_epoch=3, batch_tokens=16, lr=0.003,
    warmup_steps=0, amp="fp32", ddp="off", checkpoint_dir="outputs/lm-quickstart",
))
text = model.generate("waves learn", max_new_tokens=4, temperature=0)
model.save("lm.pt")
restored = PSANNLM.load("lm.pt", map_location="cpu")
assert restored.generate("waves learn", max_new_tokens=4, temperature=0) == text
```

`TrainConfig` controls training, including batching, epochs/steps, optimizer, accumulation, mixed precision, distributed execution, and checkpoints. `PSANNLMDataPrep` accepts local text sources or iterable text. Its `simple` tokenizer is suitable for a tiny local example. Production corpora can use `tokenizers`, SentencePiece, or a supported pretrained tokenizer. Preserve tokenizer vocabulary and special-token identity with checkpoints; vocabulary size alone does not establish token identity.

## Four architectures

Choose `LMArchitectureConfig.transformer()`, `.residual()`, `.wave()`, or `.geometric_sparse()`. Spectral residual adds `spectral=SpectralConfig(...)` to `.residual()`. It is a residual configuration, not a fifth kind. Typed policies are primary; canonical preset strings are `transformer`, `residual`, `wave`, and `geometric-sparse`. Tagged mappings specify all required policies; use `to_mapping(config)` from `psannlm.architectures` to create a complete portable mapping.

Transformer supports GELU/ReLU. Residual and wave support PSANN/GELU. Geometric-sparse supports PSANN/GELU/ReLU/tanh and valid mixtures. Wave temporal modes are `disabled`, `interleave`, `replace`, and `attention-only`; the default is disabled. See the [capability contract](architecture_contract.md) for incompatible combinations and early validation.

## Command line

```sh
python -m psannlm --help
python -m psannlm train --help
python -m psannlm train --config examples/lm/configs/waveresnet_cpu.yaml
python -m psannlm resume --config examples/lm/configs/waveresnet_cpu.yaml --resume-ckpt runs/lm/wrn_cpu_local/final.pt
python -m psannlm generate --ckpt lm.pt --prompt "waves learn" --temperature 0 --max-new-tokens 16 --device cpu
python -m psannlm eval --help
```

The YAML `model` section is an LMConfig mapping; `train` is TrainConfig and `data` selects sources/tokenization. Paths and output locations come from the selected file, so use the checkpoint path actually created by your run when resuming. Long options accept both `--config path.yaml` and `--config=path.yaml`. Streaming CLI mode also supports `--architecture` or `--model-config`; conflicting architecture sources and unknown keys reject early. Train/resume/eval/generate use the same canonical construction and persistence path.

## Checkpoints and runtime limits

Model and trainer artifacts use schema version 1. Trainer checkpoints include optimizer, step, scaler/scheduler, and resume state where applicable; a model export is intended for inference. `map_location="cpu"` reconstructs CUDA artifacts on CPU, and CUDA map location requires a working CUDA runtime. State/configuration and greedy generation are preserved through repeated new-format saves. See [migration](migration.md) for historical artifacts and explicit configuration requirements for raw weights.

Use [executable quickstarts](../examples/quickstarts.py) for small local workflows, [LM examples](../examples/lm/README.md) for training examples, and [benchmarks](benchmarks/lm_base_sweeps.md) for reproducible experiments. A single GPU cannot establish physical multi-GPU behavior; distributed examples require the hardware they declare.
