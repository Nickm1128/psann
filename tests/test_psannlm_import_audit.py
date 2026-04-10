from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PSANNLM_ROOT = REPO_ROOT / "psannlm"


def _module_name_for_path(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT).with_suffix("")
    parts = rel.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _package_name_for_module(module_name: str, path: Path) -> str:
    if path.name == "__init__.py":
        return module_name
    return module_name.rsplit(".", 1)[0]


def _collect_psannlm_modules() -> set[str]:
    modules = {"psannlm"}
    for path in PSANNLM_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        modules.add(_module_name_for_path(path))
    return modules


def test_psannlm_relative_imports_resolve_to_real_modules():
    known_modules = _collect_psannlm_modules()
    failures: list[str] = []

    for path in PSANNLM_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        module_name = _module_name_for_path(path)
        package_name = _package_name_for_module(module_name, path)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    target = "." * node.level + (node.module or "")
                    resolved = importlib.util.resolve_name(target, package_name)
                elif node.module and node.module.startswith("psannlm."):
                    resolved = node.module
                else:
                    continue

                if resolved.startswith("psannlm.") and resolved not in known_modules:
                    failures.append(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno} imports missing module {resolved}"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("psannlm.") and alias.name not in known_modules:
                        failures.append(
                            f"{path.relative_to(REPO_ROOT)}:{node.lineno} imports missing module {alias.name}"
                        )

    assert failures == []
