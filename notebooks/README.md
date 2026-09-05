# Research notebooks

Run notebooks from the repository root with the candidate packages installed. Install the notebook-specific optional dependencies shown in the first cells (Jupyter, plotting/data tools, and any public data client). GPU use is optional unless explicitly selected; set the actual device and record its identity.

- Geometric-sparse versus dense ReLU comparisons preserve the connectivity, activation, and parameter-matching experiment.
- Mixed-activation comparisons preserve mixture ratios and seeds.
- The Bitcoin five-minute notebook uses public yfinance data, chronological splits, LSM preprocessing, and validation-selected thresholds. Data availability changes with provider retention; this is a research example, not a trading recommendation.
- The wave context notebook demonstrates explicit and generated context.
- The sine-parameter notebook compares trainable/frozen activation parameters with residual/wave policies and baseline models.
- The parity/probes notebook builds canonical backbones inside explicit PyTorch training experiments. Later synthetic probes use distinct helper names so earlier experiments are not silently redefined.
- HISSO logging notebooks use canonical estimator and episodic configuration mappings and local/generated inputs.

The private-database crypto notebook was removed because its external repository, schema, and data were not provided. Remaining public-data notebooks do not claim to reproduce that experiment. Preserved notebook outputs are historical observations; rerun after changing data/configuration and record the new environment.

[Consumer manifest](../docs/consumer_manifest.json) tracks parsing, prerequisites, and build coverage. [Quickstarts](../examples/quickstarts.py) provide bounded complete workflows suitable for a local smoke run.
