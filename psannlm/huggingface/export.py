"""Convert native checkpoints to standalone Hugging Face model directories."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from psannlm.architectures import LMConfig
    from psannlm.lm.data.tokenizer import Tokenizer


def _require_dependencies() -> None:
    try:
        import safetensors  # noqa: F401
        import transformers
    except ImportError as exc:
        raise ImportError(
            "Hugging Face export requires pip install 'psannlm[huggingface]'."
        ) from exc
    major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
    if not ((major == 4 and minor >= 57) or major == 5):
        raise ImportError("Hugging Face export supports transformers>=4.57,<6.")


def _version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        # Both source distributions have this release version.
        from psannlm.persistence import _PACKAGE_VERSION

        return _PACKAGE_VERSION


def export_huggingface(
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    model_max_length: int,
    tokenizer: Tokenizer | None = None,
    legacy_config: LMConfig | Mapping[str, Any] | None = None,
    max_shard_size: str | int = "5GB",
) -> Path:
    """Export a checkpoint locally; never upload or change the native artifact.

    ``tokenizer`` is an optional fitted ``psannlm.lm.data.tokenizer.Tokenizer``.
    It supplies missing trainer/legacy tokenizer state, or must exactly match an
    embedded tokenizer. ``model_max_length`` is the intended supported context
    length, which inference checkpoints do not otherwise record reliably.

    Returns the output ``Path``. The destination must be absent or empty.
    """
    _require_dependencies()
    from .bundle import write_runtime
    from .configuration_psann import PSANNConfig
    from .modeling_psann import PSANNForCausalLM
    from .tokenization_psann import PSANNTokenizer
    from psannlm.architectures import normalize_lm_config, to_mapping
    from psannlm.architectures import registry
    from psannlm.lm.data.tokenizer import Tokenizer
    from psannlm.persistence import load_lm_checkpoint

    destination = Path(output_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("output_dir must be absent or an empty directory.")
    # Reject replacement factories before checkpoint reconstruction calls one.
    builtin = {
        "transformer": registry._transformer,
        "residual": registry._residual,
        "wave": registry._wave,
        "geometric-sparse": registry._geometric,
    }
    if any(registry._BUILDERS.get(kind) is not factory for kind, factory in builtin.items()):
        raise ValueError("Hugging Face export supports built-in LM builders only.")
    if legacy_config is not None:
        legacy_config = normalize_lm_config(legacy_config, for_build=True)
    loaded = load_lm_checkpoint(checkpoint, map_location="cpu", legacy_config=legacy_config)
    if tokenizer is not None and not isinstance(tokenizer, Tokenizer):
        raise TypeError("tokenizer must be a fitted PSANN Tokenizer.")
    if tokenizer is not None and loaded.tokenizer is not None:
        if tokenizer.to_state() != loaded.tokenizer.to_state():
            raise ValueError("Explicit tokenizer conflicts with the checkpoint tokenizer.")
    tokenizer = tokenizer if tokenizer is not None else loaded.tokenizer
    if tokenizer is None:
        raise ValueError("The checkpoint has no tokenizer; supply its original fitted tokenizer.")
    if tokenizer.vocab_size != loaded.config.vocab_size:
        raise ValueError("Tokenizer vocabulary size does not match the model.")
    hf_tokenizer = PSANNTokenizer(
        tokenizer_state=tokenizer.to_state(), model_max_length=model_max_length
    )
    config = PSANNConfig(
        lm_config=to_mapping(loaded.config),
        model_max_length=model_max_length,
        bos_token_id=hf_tokenizer.bos_token_id,
        eos_token_id=hf_tokenizer.eos_token_id,
        pad_token_id=hf_tokenizer.pad_token_id,
    )
    model = PSANNForCausalLM(config, native_model=loaded.model)
    model.eval()
    # Stage outside the destination, so failures cannot leave a mixed/partial export.
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".psann-hf-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "model"
        stage.mkdir()
        model.save_pretrained(stage, safe_serialization=True, max_shard_size=max_shard_size)
        hf_tokenizer.save_pretrained(stage)
        manifest = write_runtime(stage)
        provenance = {
            "schema": "psannlm.huggingface",
            "schema_version": 1,
            "source_versions": {p: _version(p) for p in ("psann", "psannlm")},
            "export_versions": {p: version(p) for p in ("transformers", "safetensors")},
            "runtime_sources": manifest,
        }
        (stage / "runtime_manifest.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        license_path = Path(__file__).parents[1] / "LICENSE"
        (stage / "LICENSE-PSANN").write_text(
            license_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        runtime_requirements = (
            "torch>=2.4\nnumpy>=1.23\ntransformers>=4.57,<6\nsafetensors>=0.4.5\n"
        )
        backend = tokenizer.backend_name
        if backend == "sentencepiece":
            runtime_requirements += "sentencepiece>=0.1.99\n"
        elif backend == "tokenizers":
            runtime_requirements += "tokenizers>=0.22\n"
        (stage / "requirements.txt").write_text(runtime_requirements, encoding="utf-8")
        (stage / "README.md").write_text(
            "---\nlibrary_name: transformers\npipeline_tag: text-generation\n---\n\n"
            "# PSANN language model\n\n"
            "Self-contained PSANN inference runtime; installing psann or psannlm is unnecessary.\n"
            "Install the dependencies in requirements.txt. Review the bundled code before enabling "
            "`trust_remote_code=True`.\n\n"
            "```python\nfrom transformers import AutoTokenizer, AutoModelForCausalLM\n\n"
            'repo = "YOUR_ACCOUNT/YOUR_MODEL"\n'
            "tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)\n"
            "model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)\n"
            'inputs = tokenizer("Hello", return_tensors="pt")\n'
            "output = model.generate(**inputs, max_new_tokens=32, do_sample=False, use_cache=False)\n"
            "print(tokenizer.decode(output[0], skip_special_tokens=True))\n```\n\n"
            f"Architecture: `{loaded.config.architecture.kind}`. Context limit: {model_max_length} tokens.\n\n"
            "Generation recomputes the full context; KV caching is unsupported. Use left padding "
            "for batched generation. Opaque tokenizer labels preserve the original model-facing IDs; "
            "encoding and decoding use the original fitted tokenizer.\n\n"
            "Source code is covered by LICENSE-PSANN. The model publisher must document training "
            "data, evaluations, intended uses, limitations, and the weights' license before release.\n",
            encoding="utf-8",
        )
        if destination.exists():
            # Check again immediately before publication; never remove user files.
            destination.rmdir()
        stage.rename(destination)
    return destination
