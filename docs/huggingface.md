# Hugging Face exports

PSANN-LM can export a checkpoint as a self-contained Transformers model. The
directory includes SafeTensors weights, configuration, fitted tokenizer state,
and the PSANN inference source. Loading it does not require installing `psann`
or `psannlm`. Custom code requires `trust_remote_code=True`.

## Export a checkpoint

Install the optional integration in the environment used for conversion:

```sh
python -m pip install 'psannlm[huggingface]'
python -m psannlm export-hf \
  --ckpt model.pt --output-dir exports/my-model --max-length 2048
```

From a source checkout, install the local packages with
`python -m pip install -e . -e './psannlm[huggingface]'` before running the command.

The output directory must be absent or empty. `--max-length` declares the intended
supported context length, including prompt and generated tokens. Inference
checkpoints do not reliably record the training context length; supply the value
used for your model. Exporting does not extend its trained context capability.

Model checkpoints include their fitted tokenizer. Trainer checkpoints usually
need the original tokenizer assets:

```sh
python -m psannlm export-hf \
  --ckpt runs/my-run/final.pt --output-dir exports/my-model --max-length 2048 \
  --tokenizer-backend tokenizers \
  --tokenizer-model-path runs/my-run/tokenizer/tokenizer.json \
  --tokenizer-special-map-path runs/my-run/tokenizer/special_tokens_map.json
```

Add `--tokenizer-remap-ids` only if the model was trained using PSANN's remapped
IDs. SentencePiece assets use `--tokenizer-backend sentencepiece` with their
original `.model` file. For a fitted simple tokenizer, use the Python interface,
or `--tokenizer-state` pointing to `psann_tokenizer.json` from an earlier export.
No tokenizer is trained or refitted during conversion. An explicit tokenizer
must exactly match any tokenizer already embedded in the checkpoint.

Raw legacy weights additionally require `--model-config` with their original
canonical YAML configuration. Architecture and attention-head counts are never
guessed from weights. Canonical model/trainer artifacts already record them.
Unversioned artifacts saved through DDP or `torch.compile` can carry uniform
`module.` or `_orig_mod.` prefixes; the legacy loader removes these wrappers,
including nested wrappers. Mixed prefixes reject instead of overwriting names.

The equivalent Python API can be wrapped for your checkpoint and output paths:

```python
from psannlm.huggingface import export_huggingface

def export_checkpoint(checkpoint, output_dir):
    return export_huggingface(
        checkpoint, output_dir,
        model_max_length=2048,
        # tokenizer=original_fitted_tokenizer,  # if absent from the checkpoint
        # legacy_config=original_lm_config,     # for raw legacy weights
        max_shard_size="5GB",
    )

# directory = export_checkpoint("model.pt", "exports/my-model")
```

## Load and generate

Install the export's `requirements.txt` in the destination environment. The
adapter supports Transformers 4.57 and 5.x. PyTorch, NumPy, and the tokenizer's
backend remain third-party dependencies; no PSANN packages are necessary.

```python
def generate_from_export(directory, prompts):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(directory, trust_remote_code=True)
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False, use_cache=False)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Use a local directory or a Hub repository ID:
# print(generate_from_export("exports/my-model", ["Hello", "Once upon a time"]))
```

Generation defaults to `use_cache=False` and recomputes the full context. Greedy
decoding and sampling are supported. Cached generation is explicitly rejected.
The tokenizer uses left padding for generation. Forward passes also accept right
padding; masks must describe contiguous nonempty sequences. Each row executes
the original model on its unpadded tokens to preserve positional, temporal, and
spectral behavior. Inputs beyond the declared context limit are rejected.

The custom tokenizer exposes opaque `<psann_id_N>` vocabulary labels. Encoding
and decoding delegate to the original fitted tokenizer, preserving IDs, Unicode,
special-token insertion, and any historical ID remapping. Do not replace it with
a tokenizer chosen only because its vocabulary has the same size.

Model loading constructs the native runtime on CPU before device placement;
allow enough host RAM for the model. This first adapter does not implement
quantization, KV caches, or optimized serving-engine integration. Only the four
built-in architectures are exportable; custom registry replacements reject.
Existing native checkpoints and training/export commands retain their formats.

## Publish

Inspect the generated model card and fill in training data, evaluations,
limitations, and your model weights' license. `LICENSE-PSANN` covers the bundled
source code; it does not assign a license to the trained weights.

```sh
hf auth login
hf upload YOUR_ACCOUNT/YOUR_MODEL exports/my-model --repo-type model
```

Exporting itself never uploads. The generated `runtime_manifest.json` records
package versions and SHA-256 hashes of the source and bundled modules. The runtime
copies existing inference implementations and rewrites internal imports to
sibling modules without registering global PSANN package aliases.

Standard `model.save_pretrained(...)` and `tokenizer.save_pretrained(...)` retain
the Python runtime and fitted tokenizer for another local round trip. Preserve
the original model card, license notice, requirements, and provenance manifest
separately when publishing a re-saved model.
