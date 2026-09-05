"""Release writers and compatibility reconstruction are executable contracts."""

import ast
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib
import warnings

import numpy as np
import pytest
import torch

import psann
from tools.text_encoding import find_mojibake

ROOT = Path(__file__).resolve().parents[1]
LEGACY = [
    "ResPSANNRegressor",
    "ResConvPSANNRegressor",
    "SGRPSANNRegressor",
    "WaveResNetRegressor",
    "GeoSparseRegressor",
]


def test_current_core_checkpoint_version_and_two_generation_prediction_closure(tmp_path):
    x = np.random.default_rng(71).normal(size=(16, 4)).astype(np.float32)
    y = x[:, :1] * 0.3
    model = psann.PSANNRegressor(
        architecture="residual",
        hidden_layers=1,
        hidden_units=8,
        epochs=2,
        random_state=17,
        device="cpu",
    ).fit(x, y)
    expected = model.predict(x)
    state = {key: value.clone() for key, value in model.model_.state_dict().items()}
    for generation in (1, 2):
        path = tmp_path / f"core-{generation}.pt"
        model.save(path)
        payload = torch.load(path, weights_only=False)
        assert payload["package_version"] == psann.__version__ == "2.0.1"
        assert (payload["schema"], payload["schema_version"]) == ("psann.regressor", 3)
        model = psann.PSANNRegressor.load(path)
        np.testing.assert_array_equal(model.predict(x), expected)
        for key, value in model.model_.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)


@pytest.mark.parametrize("metadata_present", [False, True])
def test_lm_runtime_version_with_and_without_distribution_metadata(metadata_present, monkeypatch):
    from psannlm import persistence

    def metadata(name):
        assert name == "psannlm"
        if not metadata_present:
            raise importlib.metadata.PackageNotFoundError(name)
        return "2.0.1"

    monkeypatch.setattr(persistence, "version", metadata)
    assert persistence.package_version() == psann.__version__ == "2.0.1"


@pytest.mark.parametrize("part,expected", [("patch", "2.0.2"), ("minor", "2.1.0")])
def test_release_helper_advances_all_version_writers_on_temporary_copies(
    part,
    expected,
    tmp_path,
    monkeypatch,
):
    from scripts import release

    files = {
        "PYPROJECT_PATH": "pyproject.toml",
        "INIT_PATH": "src/psann/__init__.py",
        "PSANNLM_PYPROJECT_PATH": "psannlm/pyproject.toml",
        "PSANNLM_PERSISTENCE_PATH": "psannlm/persistence.py",
    }
    for attribute, name in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        monkeypatch.setattr(release, attribute, target, raising=False)
    monkeypatch.setattr(release, "ROOT", tmp_path)
    assert release.main(["--part", part, "--skip-build", "--skip-upload"]) == 0
    core = tomllib.loads((tmp_path / files["PYPROJECT_PATH"]).read_text())["project"]
    lm = tomllib.loads((tmp_path / files["PSANNLM_PYPROJECT_PATH"]).read_text())["project"]
    assert core["version"] == lm["version"] == expected
    assert [dep for dep in lm["dependencies"] if dep.startswith("psann>=")] == [
        f"psann>={expected}"
    ]
    tree = ast.parse((tmp_path / files["INIT_PATH"]).read_text())
    versions = [
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
    ]
    assert versions == [expected]
    assert (tmp_path / files["INIT_PATH"]).read_text().endswith("\n")
    # Execute the copied fallback function without importing either checkout package.
    tree = ast.parse((tmp_path / files["PSANNLM_PERSISTENCE_PATH"]).read_text())
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        or isinstance(node, ast.FunctionDef)
        and node.name == "package_version"
    ]

    def missing(_):
        raise importlib.metadata.PackageNotFoundError("psannlm")

    namespace = {
        "version": missing,
        "PackageNotFoundError": importlib.metadata.PackageNotFoundError,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), "copied-persistence", "exec"), namespace)
    assert namespace["package_version"]() == expected
    assert (
        f'_PACKAGE_VERSION = "{expected}"\n\n\ndef package_version'
        in (tmp_path / files["PSANNLM_PERSISTENCE_PATH"]).read_text()
    )
    assert not (tmp_path / "dist").exists()


@pytest.mark.parametrize("name", LEGACY)
@pytest.mark.parametrize("clone_hook", [True, False], ids=["clone-hook", "constructor-clone"])
def test_legacy_lifecycle_warns_only_at_direct_construction_and_preserves_parity(
    name, clone_hook, tmp_path, monkeypatch
):
    from sklearn.base import clone

    facade = getattr(psann, name)
    if not clone_hook:
        # sklearn 1.2 uses constructor reconstruction; newer versions retain
        # that same fallback when the estimator has no custom cloning hook.
        monkeypatch.delattr(facade.__mro__[1], "__sklearn_clone__")
    options = dict(
        hidden_layers=2,
        hidden_width=8,
        epochs=2,
        random_state=71,
        device="cpu",
        batch_size=8,
        lsm=None,
        loss_params={},
    )
    if name == "GeoSparseRegressor":
        options.update(shape=(2, 2), k=3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        constructor_line = sys._getframe().f_lineno + 1
        model = facade(**options)
        assert model.get_params()["loss_params"] is options["loss_params"]
        assert model.set_params(lr=0.003) is model
        assert model.get_params()["loss_params"] is options["loss_params"]
        assert type(model) is facade
        cloned = clone(model)
        assert type(cloned) is facade
        assert cloned.get_params() == model.get_params()
        assert cloned.get_params()["loss_params"] is not options["loss_params"]
        assert not hasattr(cloned, "model_")
        shape = (16, 1, 4, 4) if name == "ResConvPSANNRegressor" else (16, 4)
        x = np.random.default_rng(13).normal(size=shape).astype(np.float32)
        y = x.reshape(16, -1).sum(axis=1, keepdims=True) * 0.1
        canonical = psann.PSANNRegressor(
            architecture=model.architecture,
            preprocessor=model.preprocessor,
            hidden_layers=2,
            hidden_units=8,
            epochs=2,
            random_state=71,
            device="cpu",
            batch_size=8,
            lr=0.003,
        ).fit(x, y)
        expected = canonical.predict(x)
        for candidate in (model, cloned):
            candidate.fit(x, y)
            np.testing.assert_array_equal(candidate.predict(x), expected)
        for generation in (1, 2):
            path = tmp_path / f"{name}-{generation}.pt"
            model.save(path)
            model = facade.load(path)
            assert type(model) is facade
            np.testing.assert_array_equal(model.predict(x), expected)
            np.testing.assert_array_equal(psann.PSANNRegressor.load(path).predict(x), expected)
        with pytest.raises(ValueError, match="Invalid parameter"):
            model.set_params(unknown_parameter=True)
        # A failed internal operation must not suppress the next external warning.
        second_line = sys._getframe().f_lineno + 1
        facade(**options)
    deprecations = [w for w in caught if w.category is DeprecationWarning]
    assert len(deprecations) == 2
    assert [w.lineno for w in deprecations] == [constructor_line, second_line]
    assert all(Path(w.filename).resolve() == Path(__file__).resolve() for w in deprecations)


def test_maintained_public_text_has_no_encoding_or_citation_artifacts():
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    suffixes = {
        "",
        ".py",
        ".md",
        ".rst",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".ipynb",
        ".sh",
        ".ps1",
        ".txt",
        ".ini",
        ".cfg",
        ".csv",
        ".vocab",
    }
    failures = []
    for name in tracked:
        path = ROOT / name
        if path.suffix.lower() in suffixes and path.is_file():
            text = path.read_text(encoding="utf-8")
            failures.extend((name, finding) for finding in find_mojibake(text))
            if path.suffix.lower() in {".json", ".ipynb"}:
                # JSON escaping must not hide corrupt cell text or configuration.
                decoded = json.dumps(json.loads(text), ensure_ascii=False)
                failures.extend((name, finding) for finding in find_mojibake(decoded))
            unresolved_citations = (":content" + "Reference[", "oai" + "cite:")
            failures.extend(
                (name, sequence) for sequence in unresolved_citations if sequence in text
            )
    assert failures == []
    assert find_mojibake("café ≈ 2 — 中文") == []


@pytest.mark.parametrize("description", [ROOT / "README.md", ROOT / "psannlm/README.md"])
def test_package_description_links_are_absolute_and_repository_targets_exist(description):
    links = re.findall(r"(?<!!)\[[^]]+\]\(([^)]+)\)", description.read_text(encoding="utf-8"))
    assert links
    assert all(link.startswith("https://") for link in links)
    repository_prefix = "https://github.com/Nickm1128/psann/blob/main/"
    repository_targets = [
        link.removeprefix(repository_prefix) for link in links if link.startswith(repository_prefix)
    ]
    assert repository_targets
    assert all((ROOT / target).is_file() for target in repository_targets)


@pytest.mark.parametrize("name", LEGACY)
def test_unversioned_checkpoint_facade_reconstruction_is_silent_and_closes_twice(name, tmp_path):
    import importlib

    module = {
        "ResPSANNRegressor": "residual",
        "ResConvPSANNRegressor": "residual",
        "SGRPSANNRegressor": "sgr",
        "WaveResNetRegressor": "wave",
        "GeoSparseRegressor": "geosparse",
    }[name]
    old_class = getattr(importlib.import_module(f"psann._sklearn.{module}"), name)
    options = dict(hidden_layers=2, hidden_units=8, epochs=2, random_state=71, device="cpu")
    if name == "GeoSparseRegressor":
        options.update(shape=(2, 2), k=3)
    shape = (16, 1, 4, 4) if name == "ResConvPSANNRegressor" else (16, 4)
    x = np.random.default_rng(17).normal(size=shape).astype(np.float32)
    y = x.reshape(16, -1).sum(axis=1, keepdims=True) * 0.1
    old = old_class(**options).fit(x, y)
    path = tmp_path / "unversioned.pt"
    old.save(path)
    expected = old.predict(x)
    facade = getattr(psann, name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        model = facade.load(path)
        for generation in (1, 2):
            assert type(model) is facade
            np.testing.assert_allclose(model.predict(x), expected, rtol=0, atol=0)
            path = tmp_path / f"migrated-{generation}.pt"
            model.save(path)
            model = facade.load(path)
            np.testing.assert_allclose(model.predict(x), expected, rtol=0, atol=0)
    assert not [w for w in caught if w.category is DeprecationWarning]


@pytest.mark.parametrize("defect", ["missing-floor", "duplicate-floor", "missing-fallback"])
def test_lm_release_writer_rejects_missing_or_ambiguous_version_fields_without_writing(
    defect,
    tmp_path,
    monkeypatch,
):
    from scripts import release

    project = (ROOT / "psannlm/pyproject.toml").read_text()
    fallback = (ROOT / "psannlm/persistence.py").read_text()
    if defect == "missing-floor":
        project = project.replace('"psann>=2.0.1",', "")
    elif defect == "duplicate-floor":
        project = project.replace('"psann>=2.0.1",', '"psann>=2.0.1", "psann>=0.12.4",')
    else:
        fallback = fallback.replace('_PACKAGE_VERSION = "2.0.1"', "")
    project_path, fallback_path = tmp_path / "pyproject.toml", tmp_path / "persistence.py"
    project_path.write_text(project)
    fallback_path.write_text(fallback)
    monkeypatch.setattr(release, "PSANNLM_PYPROJECT_PATH", project_path)
    monkeypatch.setattr(release, "PSANNLM_PERSISTENCE_PATH", fallback_path)
    with pytest.raises(RuntimeError, match="Expected one"):
        release.write_psannlm_version("2.0.2")
    assert project_path.read_text() == project
    assert fallback_path.read_text() == fallback
