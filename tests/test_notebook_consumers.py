"""Execute notebook construction code without downloading research datasets."""

import ast
import __future__
from pathlib import Path
import math
import runpy

import numpy as np
import pytest
import torch

from psann import PSANNRegressor
from psann import architectures, preprocessing
from test_consumer_manifest import EXAMPLES, ROOT, code_cells, evaluate_expression


def notebook_definitions(path):
    namespace = {
        "__name__": __name__,
        "np": np,
        "torch": torch,
        "nn": torch.nn,
        "math": math,
        "Path": Path,
        "PSANNRegressor": PSANNRegressor,
        "DEVICE": "cpu",
        "SEED": 7,
        "seed": 7,
        "input_dim": 16,
        "shape": (4, 4),
        "activation_type": "psann",
        "hidden_layers": 2,
        "hidden_units": 16,
    }
    namespace.update({name: getattr(architectures, name) for name in architectures.__all__})
    namespace.update({name: getattr(preprocessing, name) for name in preprocessing.__all__})
    from examples.torch_backbone import build_backbone

    namespace["build_backbone"] = build_backbone
    cells = code_cells(path)
    tree = ast.parse("\n".join(cells))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "resolve_geo_shape",
            "build_wave_backbone",
            "make_wave_resnet",
            "count_geosparse_params",
            "count_dense_params",
        }:
            exec(
                compile(
                    ast.Module(body=[node], type_ignores=[]),
                    path,
                    "exec",
                    flags=__future__.annotations.compiler_flag,
                ),
                namespace,
            )
    return namespace, tree


@pytest.mark.parametrize(
    "path",
    [
        p
        for p, r in EXAMPLES.items()
        if r["format"] == "notebook" and r["boundary"] == "core-estimator"
    ],
)
def test_notebook_estimator_policies_fit_predict_and_preserve_configuration(path, tmp_path):
    namespace, tree = notebook_definitions(path)
    if "geosparse_vs_relu_benchmarks" in path:
        assignment = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "DEFAULT_TRAIN_CFG" for t in node.targets)
        )
        namespace["cfg"] = evaluate_expression(assignment.value, namespace, tree, assignment.lineno)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "PSANNRegressor"
    ]
    assert calls, path
    for index, call in enumerate(calls):
        estimator = evaluate_expression(call, dict(namespace), tree, call.lineno)
        estimator.set_params(
            epochs=1, batch_size=8, device="cpu", early_stopping=False, amp=False, compile=False,
            random_state=19,
        )
        if estimator.preprocessor is not None:
            estimator.set_params(preprocessor__component__pretraining__epochs=1)
        geometry = estimator.architecture.geometry
        features = (
            math.prod(geometry.shape)
            if geometry is not None and geometry.shape and estimator.preprocessor is None
            else 4
        )
        x = np.random.default_rng(7).normal(size=(16, features)).astype(np.float32)
        y = np.sin(x[:, :1])
        context = estimator.architecture.context
        fit = (
            {"context": np.ones((16, context.dim), dtype=np.float32)}
            if context and context.dim and context.builder is None
            else {}
        )
        if estimator.architecture.kind == "wave":
            # Compatibility reference keeps the original wrapper's implicit W0
            # warmup. The notebook must spell that policy out after migration.
            from psann import WaveResNetRegressor

            legacy = WaveResNetRegressor(
                hidden_layers=estimator.hidden_layers, hidden_units=estimator.hidden_units,
                epochs=1, batch_size=8, lr=estimator.lr, random_state=19, device="cpu",
                activation=estimator.architecture.activation,
                context_dim=context.dim if context else None,
                context_builder=context.builder if context else None,
                context_builder_params=dict(context.builder_params) if context and context.builder_params else None,
            )
            assert estimator.architecture == legacy.architecture
            legacy.fit(x, y, **fit)
        estimator.fit(x, y, **fit)
        expected = estimator.predict(x, **fit)
        if estimator.architecture.kind == "wave":
            np.testing.assert_array_equal(expected, legacy.predict(x, **fit))
        checkpoint = tmp_path / f"notebook-{index}.pt"
        estimator.save(checkpoint)
        restored = PSANNRegressor.load(checkpoint)
        assert restored.architecture == estimator.architecture
        np.testing.assert_array_equal(restored.predict(x, **fit), expected)


def test_notebook_sparse_and_dense_parameter_matching_uses_the_executed_backbones():
    path = "notebooks/geosparse_vs_relu_benchmarks.ipynb"
    namespace, _ = notebook_definitions(path)
    from psann import count_params
    from examples.torch_backbone import build_backbone

    namespace["count_params"] = count_params
    sparse = build_backbone(
        architectures.ArchitectureConfig.geometric_sparse(
            geometry=architectures.GeometryConfig(shape=(4, 4), k=8, seed=1337)
        ),
        (16,),
        1,
        depth=4,
    )
    dense = build_backbone(
        architectures.ArchitectureConfig.dense(
            activation=architectures.ActivationConfig(kind="relu")
        ),
        (16,),
        1,
        depth=2,
        width=24,
    )
    assert namespace["count_geosparse_params"](16, 1, shape=(4, 4)) == count_params(
        sparse, trainable_only=True
    )
    assert namespace["count_dense_params"](16, 1, hidden_units=24) == count_params(
        dense, trainable_only=True
    )


def test_parity_notebook_wave_factory_preserves_original_logits_and_gradients():
    namespace, _ = notebook_definitions("notebooks/PSANN_Parity_and_Probes.ipynb")
    from psann.models import WaveResNet

    options = dict(
        input_dim=4,
        hidden_dim=16,
        depth=3,
        output_dim=2,
        dropout=0.0,
        first_layer_w0=17.0,
        hidden_w0=0.7,
    )
    torch.manual_seed(71)
    original = WaveResNet(**options)
    torch.manual_seed(71)
    actual = namespace["build_wave_backbone"](**options)
    x = torch.randn(8, 4)
    expected = original(x)
    result = actual(x)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    expected.square().sum().backward()
    result.square().sum().backward()
    for before, after in zip(original.parameters(), actual.parameters()):
        torch.testing.assert_close(after.grad, before.grad, rtol=0, atol=0)


@pytest.mark.parametrize(
    "path", [p for p, r in EXAMPLES.items() if r["boundary"] == "torch-composition"]
)
def test_advanced_example_composition_executes_optimizer_and_prediction(path):
    namespace = runpy.run_path(str(ROOT / path))
    if "/10_" in path:
        model = namespace["PSANNWithAttention"](embed=8, depth=1, heads=2)
        x = torch.randn(4, 3, 1, 8, 8)
        target = torch.ones(4, 1)
        criterion = torch.nn.MSELoss()
    else:
        tree = ast.parse((ROOT / path).read_text())
        node = next(
            n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "PSANNCNN"
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
        model = namespace["PSANNCNN"]()
        x = torch.randn(4, 1, 16, 16)
        target = torch.tensor([0, 1, 1, 0])
        criterion = torch.nn.CrossEntropyLoss()
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion(model(x), target).backward()
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert model(x).shape[0] == 4


@pytest.mark.parametrize(
    "path", [p for p, r in EXAMPLES.items() if r["boundary"] == "logging-config"]
)
def test_logging_notebook_referenced_or_emitted_policies_execute(path, tmp_path):
    from psann.episodic import EpisodicTrainer
    import yaml

    if "GPU_Run" in path:
        namespace = dict(
            USE_AMP=False,
            DENSE_EPOCHS=1,
            WAVE_EPOCHS=1,
            TARGET_DEVICE="cpu",
            wave_npz_path=tmp_path / "wave.npz",
            Path=Path,
        )
        tree = ast.parse("\n".join(code_cells(path)))
        node = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "logging_config"
        )
        exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
        configs = [namespace["logging_config"](kind) for kind in ("dense", "wave")]
    else:
        configs = [
            yaml.safe_load((ROOT / "configs/hisso" / name).read_text())
            for name in ("dense_cpu_smoke.yaml", "wave_resnet_small.yaml")
        ]
    for raw in configs:
        params = dict(raw["estimator"]["params"], epochs=1, device="cpu", output_shape=(2,))
        strategy = dict(raw["episodic"]["strategy"], mixed_precision=False, warm_start=None)
        strategy["schedule"] = dict(episode_length=8, batch_episodes=2)
        trainer = EpisodicTrainer(estimator=PSANNRegressor(**params), strategy=strategy)
        prices = np.exp(np.linspace(0, 1, 32)[:, None] * np.array([[0.1, -0.1]])).astype(np.float32)
        trainer.fit(prices)
        np.testing.assert_allclose(trainer.predict(prices).sum(axis=1), 1, atol=1e-6)
        saved = tmp_path / "logging.pt"
        trainer.save(saved)
        restored = EpisodicTrainer.load(saved)
        assert restored.evaluate(prices) == trainer.evaluate(prices)
