# Documentation

Start with the [regression quickstart](../README.md). Choose a task, then consult the API and architecture references for exact supported combinations.

| Task | Guide | Executable consumer |
| --- | --- | --- |
| Regression and model selection | [API](API.md) | [Core quickstart](../examples/quickstarts.py) |
| Preprocessing and LSM composition | [Preprocessing](preprocessing.md) | [Supervised comparison](../examples/14_psann_with_vs_without_lsm.py) |
| Episodic training and rewards | [Episodic training](episodic.md) | [Allocation example](../examples/26_hisso_unsupervised_allocation.py) |
| Language modeling | [LM guide](lm.md) | [LM examples](../examples/lm/README.md) |
| Architecture selection | [Capability contract](architecture_contract.md) | [Example index](examples/README.md) |
| Wave context | [Wave guide](wave_resnet.md) | [Context notebook](../notebooks/PSANN_WaveResNet_Context_Demo.ipynb) |
| Sparse connectivity | [Geometry guide](geo_sparse.md) | [Geometry example](../examples/28_geosparse_regression.py) |
| Upgrade an existing application | [Migration](migration.md) | Old-to-new executable snippets |

[Public imports](public_api.md), [architecture implementation](architecture.md), [shared components](architecture_components.md), [repository map](PROJECT_MAP.md), [contribution checks](CONTRIBUTING.md), [deprecation policy](deprecation_policy.md), and [changelog](../CHANGELOG.md) provide the reference path.

Historical empirical results live in the explicitly labeled benchmark reports. Their hardware, datasets, and package versions describe those runs only.
