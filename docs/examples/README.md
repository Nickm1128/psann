# Maintained examples

The executable inventory is [consumer_manifest.json](../consumer_manifest.json). Run from a source checkout with the declared optional dependencies and generated inputs. The test suite compiles every Python/notebook consumer, normalizes all YAML configurations, checks local references, and exercises build boundaries plus complete small workflows.

Start with [quickstarts](../../examples/quickstarts.py): core regression, LSM preprocessing, episodic reward training, and LM training/generation. Each saves and loads twice.

Numbered examples cover basic and custom-loss regression (01–04), spatial regression/segmentation (05–06), stateful forecasting (07–09), PyTorch temporal/classification composition (10–11), streaming updates (12–13), supervised LSM comparison (14), episodic comparisons (15–17), configuration benchmarking (21), allocation (26–27), and geometric-sparse regression (28). All estimator examples use `PSANNRegressor` with nested policies. The advanced PyTorch examples own their optimizer because they compose temporal networks or use cross-entropy; their backbones are built from canonical `ArchitectureConfig` through [torch_backbone.py](../../examples/torch_backbone.py).

[LM examples](../../examples/lm/README.md) and [notebooks](../../notebooks/README.md) describe their own prerequisites. Example numbering is retained for link stability; gaps are intentional. Unusable extras-head examples were removed because the required API did not exist; the supervised LSM comparison remains a different, supported experiment.
