"""Import direction and maintained construction routes are executable boundaries."""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "document",
    [
        "docs/lm.md",
        "docs/how_to_add_model_benchmark_dataset.md",
        "psannlm/README.md",
    ],
)
def test_maintained_lm_documents_teach_canonical_builder_registration(document):
    text = (ROOT / document).read_text(encoding="utf-8")
    assert "psannlm/lm/models/registry.py" not in text
    for snippet in re.findall(r"```python\n(.*?)```", text, re.S):
        tree = ast.parse(snippet)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else (node.func.attr if isinstance(node.func, ast.Attribute) else "")
                )
                assert name not in {"register_base", "get_base", "register_lm_builder"}, (
                    document,
                    name,
                )


def test_lm_imports_only_documented_shared_core_modules_and_exports():
    allowed = {"psann.architectures", "psann.architectures.components", "psann.utils"}
    for path in (ROOT / "psannlm").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("psann."):
                assert node.module in allowed, (path, node.lineno, node.module)
                if node.module == "psann.utils":
                    assert all(alias.name == "cleanup_hf_cache" for alias in node.names)
            if isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith("psann.") or alias.name in allowed
                    for alias in node.names
                ), path


def test_core_has_no_lm_import_or_optional_import_probe():
    for path in (ROOT / "src/psann").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("psannlm"), path
            elif isinstance(node, ast.Import):
                assert all(not alias.name.startswith("psannlm") for alias in node.names), path
            elif isinstance(node, ast.Call):
                name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                )
                if (
                    name in {"__import__", "import_module", "find_spec"}
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                ):
                    assert not str(node.args[0].value).startswith("psannlm"), path


def test_maintained_consumers_do_not_construct_legacy_models_or_call_legacy_registry():
    exceptions = {"psannlm/lm/api.py", "psannlm/lm/models/registry.py"}
    legacy = {
        "get_base",
        "psannLM",
        "psannLMDataPrep",
        "Trainer",
        "VanillaTransformerConfig",
        "ResPSANNTransformerConfig",
        "WaveResNetTransformerConfig",
        "GeoSparseTransformerConfig",
    }
    for directory in ("psannlm", "scripts", "examples"):
        for path in (ROOT / directory).rglob("*.py"):
            relative = path.relative_to(ROOT).as_posix()
            if relative in exceptions or relative.startswith("psannlm/lm/models/"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            aliases = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and "psannlm" in (node.module or ""):
                    aliases.update({alias.asname or alias.name: alias.name for alias in node.names})
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert aliases.get(node.func.id, node.func.id) not in legacy, (
                        relative,
                        node.lineno,
                        node.func.id,
                    )
