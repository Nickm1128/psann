# Contributing

Install a suitable PyTorch build first, then core development dependencies and the separate LM distribution:

```sh
python -m pip install -e ".[dev,sklearn,viz]"
python -m pip install ./psannlm
```

Use typed canonical task/configuration APIs in new examples and documentation. Keep compatibility teaching in migration/deprecation material. Add new maintained consumers and prerequisites to [consumer_manifest.json](consumer_manifest.json). Preserve experimental intent explicitly when changing a benchmark configuration. Label historical results with their original scope and retain reproducible provenance such as dataset, seed, configuration, and hardware.

Run required checks sequentially:

```sh
python -m pytest -m "not slow and not gpu" -q
python -m pytest -m slow -q
python -m pytest -m gpu -q
python -m ruff check --select F,E9 .
python -m black --check .
python -m mypy src psannlm
git diff --check
python -m build
python -m build ./psannlm
```

GPU tests require CUDA; report the actual device count and runtime. On Windows, import torch before scikit-learn in combined numerical processes. A loader failure before tests start is not a passing test run.

Core and LM wheels must be disjoint. Test fresh core-only and combined installations, package versions, dependencies (`pip check`), import origins, CLI help, and real fit/train/inference/persistence paths. New runtime defects need regression coverage before repair. Keep mechanical formatting separate from semantic edits where practical; do not hide static errors by changing scope or adding broad ignores.

Generated checkpoints, logs, datasets, caches, build output, and private implementation process material do not belong in tracked package source. [Repository map](PROJECT_MAP.md) and [extension guide](how_to_add_model_benchmark_dataset.md) explain ownership boundaries.
