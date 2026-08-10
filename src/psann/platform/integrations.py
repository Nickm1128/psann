"""Explicit optional adapters for internal artifact registries."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import unquote, urlparse

from .inference import InferenceRuntime, load_runtime
from .registry import IdentifierRegistry
from .specs import InferenceConfig

ArtifactResolver = Callable[[str], str | os.PathLike[str]]
ARTIFACT_RESOLVERS: IdentifierRegistry[ArtifactResolver] = IdentifierRegistry("artifact resolver")


def _file_resolver(reference: str) -> Path:
    if "://" not in reference:
        return Path(reference)
    parsed = urlparse(reference)
    if parsed.scheme not in {"", "file"}:
        raise ValueError(f"The file resolver cannot handle scheme {parsed.scheme!r}.")
    if parsed.scheme == "":
        return Path(reference)
    path = unquote(parsed.path)
    if parsed.netloc:
        path = f"//{parsed.netloc}{path}"
    if os.name == "nt" and path.startswith("/") and len(path) > 2 and path[2] == ":":
        path = path[1:]
    return Path(path)


ARTIFACT_RESOLVERS.register("file", _file_resolver)


def register_artifact_resolver(
    identifier: str,
    resolver: ArtifactResolver,
    *,
    replace: bool = False,
) -> str:
    """Register an explicit URI-to-local-artifact adapter."""

    if not callable(resolver):
        raise TypeError("artifact resolver must be callable.")
    return ARTIFACT_RESOLVERS.register(identifier, resolver, replace=replace)


def resolve_artifact(
    reference: str | os.PathLike[str],
    *,
    resolver: str | None = None,
) -> Path:
    """Resolve a registry reference to a local file without implicit discovery."""

    value = os.fspath(reference)
    selected = resolver
    if selected is None:
        selected = value.split("://", 1)[0] if "://" in value else "file"
    resolved = Path(ARTIFACT_RESOLVERS.resolve(selected)(value)).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Artifact resolver {selected!r} returned no file: {resolved}")
    return resolved


def load_registry_runtime(
    reference: str | os.PathLike[str],
    *,
    resolver: str | None = None,
    config: InferenceConfig | Mapping[str, Any] | None = None,
) -> InferenceRuntime:
    """Resolve an internal-registry reference and restricted-load its artifact."""

    return load_runtime(resolve_artifact(reference, resolver=resolver), config=config)


__all__ = [
    "ARTIFACT_RESOLVERS",
    "ArtifactResolver",
    "load_registry_runtime",
    "register_artifact_resolver",
    "resolve_artifact",
]
