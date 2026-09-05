"""Manifest coverage for source consumers, docs, configuration, and complete tasks."""

import ast
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml
from IPython.core.inputtransformer2 import TransformerManager

from psann import PSANNRegressor
from psannlm.architectures import build_lm_model, normalize_lm_config

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "docs/consumer_manifest.json").read_text())
EXAMPLES = {row["path"]: row for row in MANIFEST["examples"]}
LEGACY_NAMES = re.compile(
    r"\b(?:ResPSANNRegressor|ResConvPSANNRegressor|SGRPSANNRegressor|"
    r"WaveResNetRegressor|GeoSparseRegressor|psannLM|psannLMDataPrep|HISSOOptions)\b"
)


@pytest.mark.parametrize("row", MANIFEST["cli"], ids=lambda r: r["module"])
def test_every_public_cli_help_executes_and_teaches_canonical_options(row):
    result = subprocess.run(
        [sys.executable, "-m", row["module"], "--help"],
        cwd=ROOT,
        env=dict(os.environ, PYTHONUTF8="1"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower(), result.stdout
    if row["classification"] == "canonical":
        assert not LEGACY_NAMES.search(result.stdout)
        assert "--base " not in result.stdout and "--sine-" not in result.stdout


def code_cells(path):
    source = (ROOT / path).read_text(encoding="utf-8")
    if path.endswith(".py"):
        return [source]
    notebook = json.loads(source)
    return [
        TransformerManager().transform_cell("".join(cell["source"]))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def test_manifest_covers_every_maintained_example_configuration_and_document():
    examples = {
        p.relative_to(ROOT).as_posix()
        for directory in (ROOT / "examples", ROOT / "notebooks")
        for p in directory.rglob("*")
        if p.suffix in {".py", ".ipynb"} and ".ipynb_checkpoints" not in p.parts
    }
    assert examples == set(EXAMPLES)
    configs = {
        p.relative_to(ROOT).as_posix()
        for directory in (ROOT / "examples", ROOT / "configs", ROOT / "benchmarks")
        for p in directory.rglob("*")
        if p.suffix in {".yaml", ".yml"}
    }
    assert configs == {row["path"] for row in MANIFEST["configurations"]}
    tracked = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True
    ).splitlines()
    docs = {p for p in tracked if p.endswith((".md", ".rst")) and (ROOT / p).exists()}
    assert docs == {row["path"] for row in MANIFEST["documents"]}


@pytest.mark.parametrize("path", EXAMPLES)
def test_every_maintained_python_and_notebook_cell_compiles_and_uses_canonical_names(path):
    for index, code in enumerate(code_cells(path)):
        compile(code, f"{path}:cell{index}", "exec")
        assert not LEGACY_NAMES.search(code), path


@pytest.mark.parametrize("row", MANIFEST["documents"], ids=lambda r: r["path"])
def test_document_classification_local_links_and_canonical_python_blocks(
    row, tmp_path, monkeypatch
):
    path = ROOT / row["path"]
    text = path.read_text(encoding="utf-8")
    if row["classification"] == "historical result":
        assert text.startswith("> Historical result record.")
        return
    if row["classification"] == "canonical":
        assert not LEGACY_NAMES.search(text), row["path"]
        assert not re.search(r"python -m psannlm\.(?:train|cli|lm\.train)", text)
    for target in re.findall(r"\]\(([^)]+)\)", text):
        if re.match(r"[a-zA-Z]+:", target) or target.startswith("#"):
            continue
        target = target.split("#")[0].split("?")[0]
        assert (path.parent / target).resolve().exists(), (row["path"], target)
    # Complete fenced snippets are actual documented consumers, including fit/save/load.
    monkeypatch.chdir(tmp_path)
    namespace = {"__name__": "documented_consumer"}
    for index, code in enumerate(re.findall(r"```python\n(.*?)```", text, re.S)):
        compiled = compile(code, f"{row['path']}:block{index}", "exec")
        if row["classification"] == "canonical":
            exec(compiled, namespace)


@pytest.mark.parametrize("item", MANIFEST["inputs"], ids=lambda r: r.get("path", r.get("provider")))
def test_declared_input_is_present_or_has_an_executable_generator(item, tmp_path):
    if item["kind"] == "tracked":
        assert (ROOT / item["path"]).is_file()
    elif item["kind"] == "generated":
        arguments = list(item["arguments"])
        output = tmp_path / Path(item["path"]).name
        arguments[arguments.index("--out") + 1] = str(output)
        subprocess.run([sys.executable, str(ROOT / item["generator"]), *arguments], check=True)
        assert output.stat().st_size >= 1024 * 1024
        assert "\n" in output.read_text()
    else:
        assert item["kind"] == "external" and item["description"] and item["provider"]


@pytest.mark.parametrize("row", MANIFEST["configurations"], ids=lambda r: r["path"])
def test_every_configuration_normalizes_and_executes_its_model_build_boundary(row, monkeypatch):
    config = yaml.safe_load((ROOT / row["path"]).read_text())
    if row["runner"] == "hisso":
        from psann.episodic import EpisodicTrainer

        estimator = PSANNRegressor(**config["estimator"]["params"])
        trainer = EpisodicTrainer(estimator=estimator, strategy=config["episodic"]["strategy"])
        assert trainer.estimator is estimator
        return  # Actual fit/persistence for each file is in test_public_consumers.
    if row["runner"] == "lm-benchmark":
        from scripts._bench_lm_bases.models import benchmark_model_config

        models = [benchmark_model_config(config, name, 32) for name in config["models"]]
    else:
        from psannlm import normalize_train_config

        normalize_train_config(config["train"])
        models = [normalize_lm_config(config["model"])]
    for model in models:
        # Execute the full declared graph while placing large matrices on meta.
        # Policy validation and activation initialization still execute on CPU.
        with monkeypatch.context() as allocation:
            for layer in (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv1d):
                original_init = layer.__init__

                def meta_init(self, *args, _init=original_init, **kwargs):
                    kwargs["device"] = "meta"
                    _init(self, *args, **kwargs)

                allocation.setattr(layer, "__init__", meta_init)
            built = build_lm_model(replace(model, vocab_size=32)).model
        assert built.lm_config.d_model == model.d_model
        assert built.lm_config.architecture == model.architecture
        assert sum(p.numel() for p in built.parameters()) > 0
        del built
        # Real logits/gradients use an explicitly reduced shared size/depth;
        # geometry and all other architecture policies remain the original values.
        geometry = model.architecture.geometry
        width = model.d_model if geometry is not None and geometry.shape is not None else 24
        bounded = replace(
            model,
            d_model=width,
            n_layers=1,
            n_heads=1,
            d_mlp=model.d_mlp if width != 24 else 36,
            vocab_size=32,
        )
        actual = build_lm_model(bounded).model
        ids = torch.arange(14).reshape(2, 7)
        before = actual.lm_head.weight.detach().clone()
        optimizer = torch.optim.AdamW(actual.parameters(), lr=0.003)
        loss = torch.nn.functional.cross_entropy(
            actual(ids).flatten(0, 1), ids.roll(-1, 1).flatten()
        )
        loss.backward()
        optimizer.step()
        assert not torch.equal(before, actual.lm_head.weight)


def evaluate_expression(expression, namespace, tree, line):
    """Resolve the example's own configuration assignments, never replacement models."""
    compiled = compile(ast.Expression(expression), "consumer configuration", "eval")
    for _ in range(30):
        try:
            return eval(compiled, namespace)
        except NameError as error:
            name = error.name
            choices = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and node.lineno < line
                and any(
                    isinstance(target, ast.Name) and target.id == name for target in node.targets
                )
            ]
            assert choices, (name, ast.unparse(expression))
            assignment = max(choices, key=lambda node: node.lineno)
            namespace[name] = evaluate_expression(
                assignment.value, namespace, tree, assignment.lineno
            )
    raise AssertionError("Cyclic consumer configuration")


@pytest.mark.parametrize(
    "path",
    [
        p
        for p, r in EXAMPLES.items()
        if r["format"] == "python" and r["boundary"] == "core-estimator"
    ],
)
def test_each_numbered_estimator_configuration_builds_and_updates_parameters(path, monkeypatch):
    namespace = runpy.run_path(str(ROOT / path))
    tree = ast.parse((ROOT / path).read_text())
    namespace.update(
        M=2,
        seed=7,
        epochs=1,
        hidden_layers=2,
        hidden_width=16,
        activation_type="psann",
        lsm_cfg=None,
        train=np.ones((16, 2), dtype=np.float32),
    )
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "PSANNRegressor"
    ]
    assert calls, path
    changes = []
    step = torch.optim.Adam.step

    def record_step(optimizer, *args, **kwargs):
        parameters = [
            p for group in optimizer.param_groups for p in group["params"] if p.grad is not None
        ]
        before = [p.detach().clone() for p in parameters]
        result = step(optimizer, *args, **kwargs)
        changes.append(any(not torch.equal(a, b) for a, b in zip(before, parameters)))
        return result

    monkeypatch.setattr(torch.optim.Adam, "step", record_step)
    for call in calls:
        estimator = evaluate_expression(call, dict(namespace), tree, call.lineno)
        estimator.set_params(epochs=1, batch_size=8, early_stopping=False, device="cpu")
        if estimator.preprocessor is not None:
            estimator.set_params(preprocessor__component__pretraining__epochs=0)
        spatial = estimator.architecture.convolution is not None
        features = (
            int(np.prod(estimator.architecture.geometry.shape))
            if estimator.architecture.geometry and estimator.architecture.geometry.shape
            else 4
        )
        x = (
            np.random.default_rng(7)
            .normal(size=(16, 1, 8, 8) if spatial else (16, features))
            .astype(np.float32)
        )
        outputs = estimator.output_shape[0] if estimator.output_shape else 1
        per_element = spatial and estimator.architecture.convolution.per_element
        y = np.ones((16, outputs, 8, 8) if per_element else (16, outputs), dtype=np.float32)
        count = len(changes)
        estimator.fit(x, y)
        assert len(changes) > count and any(changes[count:]), path
        assert estimator.predict(x).shape == y.shape


@pytest.mark.parametrize("workflow", MANIFEST["workflows"], ids=lambda r: r["name"])
def test_documented_complete_workflow_updates_infers_and_closes_two_generations(
    workflow, tmp_path, monkeypatch
):
    quickstart = importlib.import_module("examples.quickstarts")
    if workflow["name"] in {"core", "preprocessing"}:
        estimator, x, y, prediction = quickstart.core(
            tmp_path, preprocessing=workflow["preprocessing"]
        )
        assert np.mean((prediction - y) ** 2) < np.mean(y**2)
        assert (tmp_path / "core-1.pt").is_file() and (tmp_path / "core-2.pt").is_file()
    elif workflow["name"] == "episodic":
        trainer, prices, actions, reward = quickstart.episodic(tmp_path)
        np.testing.assert_allclose(actions.sum(axis=1), 1, atol=1e-6)
        assert reward == trainer.evaluate(prices)
        assert np.mean(actions[:, 0]) > np.mean(actions[:, 1])
    else:
        from psannlm.lm import api

        snapshots = []
        original = api.build_lm_model

        def capture(config):
            result = original(config)
            snapshots.append((result.model, result.model.lm_head.weight.detach().clone()))
            return result

        monkeypatch.setattr(api, "build_lm_model", capture)
        model, data, generated = quickstart.lm(tmp_path)
        assert not torch.equal(snapshots[0][0].lm_head.weight, snapshots[0][1])
        checkpoint = torch.load(tmp_path / "trainer/final.pt", weights_only=True)
        assert checkpoint["state"]["step"] == 3
        assert all(state["step"].item() == 3 for state in checkpoint["optim"]["state"].values())
        assert model.generate("waves learn", max_new_tokens=4, temperature=0) == generated
