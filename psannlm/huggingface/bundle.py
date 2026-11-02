"""Build auditable sibling modules for Transformers' remote-code loader.

The manifest is intentionally explicit: adding a dependency must be reviewed.
Only import paths and the listed lightweight facades change; numerical code is
copied from installed package sources, never reimplemented in a second runtime.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path

MODULES = (
    "psann._aliases",
    "psann.activations",
    "psann.architectures.config",
    "psann.architectures.components",
    "psann.layers.film",
    "psann.layers.geo_sparse",
    "psann.layers.sine_residual",
    "psann.layers.spectral",
    "psann.nn",
    "psann.nn_geo_sparse",
    "psann.state",
    "psann.types",
    "psann.utils",
    "psannlm.architectures.config",
    "psannlm.architectures.activation",
    "psannlm.architectures.compat",
    "psannlm.architectures.registry",
    "psannlm.lm.config",
    "psannlm.lm.data.tokenizer",
    "psannlm.lm.models.sine",
    "psannlm.lm.models.transformer_respsann",
    "psannlm.lm.models.transformer_vanilla",
    "psannlm.lm.models.transformer_waveresnet",
    "psannlm.lm.models.transformer_geosparse",
    "psannlm.huggingface.configuration_psann",
    "psannlm.huggingface.modeling_psann",
    "psannlm.huggingface.tokenization_psann",
)


def module_name(name: str) -> str:
    if name.startswith("psannlm.huggingface."):
        return name.rsplit(".", 1)[-1]
    return "runtime_" + name.replace(".", "_")


class _Imports(ast.NodeTransformer):
    def __init__(self, package: str):
        self.package = package

    def visit_ImportFrom(self, node):
        name = node.module or ""
        if node.level:
            name = importlib.util.resolve_name("." * node.level + name, self.package)
        # Package facades reexport configuration only, avoiding unrelated builders.
        if name == "psann.architectures":
            name = "psann.architectures.config"
        if name.startswith(("psann.", "psannlm.")):
            if name not in MODULES:
                raise ValueError(f"Unlisted inference dependency: {name}")
            return ast.copy_location(
                ast.ImportFrom(module=module_name(name), names=node.names, level=1), node
            )
        return node

    def visit_Import(self, node):
        if any(a.name.split(".")[0] in {"psann", "psannlm"} for a in node.names):
            raise ValueError("Internal runtime imports must use from-import syntax.")
        return node


def write_runtime(directory: Path) -> dict[str, dict[str, str]]:
    manifest = {}
    for name in MODULES:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise ValueError(f"Inference source is unavailable: {name}")
        path = Path(spec.origin)
        source_bytes = path.read_bytes()
        source = source_bytes.decode("utf-8-sig")
        tree = ast.parse(source)
        if name == "psann.utils":
            # This helper's only dependencies are math and torch.
            helper = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "init_siren_linear_"
            )
            tree = ast.parse("from __future__ import annotations\nimport math\nimport torch\n")
            tree.body.append(helper)
        elif name == "psannlm.lm.config":
            # Models consume only positional-encoding normalization, not trainer policies.
            helper = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "normalize_positional_encoding"
            )
            tree = ast.parse(
                'from __future__ import annotations\nfrom typing import Optional\nPOS_ENCODING_CHOICES = ("rope", "alibi", "sinusoidal")\n'
            )
            tree.body.append(helper)
        package = name if path.name == "__init__.py" else name.rsplit(".", 1)[0]
        tree = _Imports(package).visit(tree)
        output = "# Generated from " + name + "; see runtime_manifest.json and LICENSE-PSANN.\n"
        output += ast.unparse(ast.fix_missing_locations(tree)) + "\n"
        if name.startswith("psannlm.huggingface."):
            # Local-directory loading copies immediate dependencies only. List the
            # complete runtime closure for HF's static source copier without eagerly
            # importing modules (which would break the builders' lazy import cycles).
            output += "\n# Source dependencies for the Hugging Face code copier.\nif False:\n"
            for dependency in MODULES:
                if not dependency.startswith("psannlm.huggingface."):
                    output += f"    from .{module_name(dependency)} import __name__\n"
        filename = module_name(name) + ".py"
        (directory / filename).write_text(output, encoding="utf-8")
        manifest[filename] = {
            "source_module": name,
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "bundled_sha256": hashlib.sha256(output.encode()).hexdigest(),
        }
    return manifest
