# Changelog

## 2.0.1 — 2026-09-05

This patch release replaces repository-relative links in the PyPI package description with absolute GitHub links. The 2.x typed configuration API remains authoritative, and there are no runtime behavior changes.

## 2.0.0 — 2026-09-05

This major release establishes the authoritative public interface around `PSANNRegressor` with immutable architecture and preprocessing policies, `EpisodicTrainer` with `HISSOConfig`, and the separate `PSANNLM`/`PSANNLMDataPrep` task API with `LMConfig` and `LMArchitectureConfig`. `python -m psannlm` is the canonical LM command.

Architecture validation rejects incompatible policies before numerical execution. Preprocessing composes frozen, jointly trained, and reconstruction-pretrained components with supervised and episodic tasks. LM construction has four typed kinds, including spectral behavior as a residual policy, and uses shared documented core primitives.

Core schema-v3 and LM schema-v1 persistence preserve accepted migration routes and repeated new-format reconstruction. Older imports, constructors, flat inputs, and CLI shims remain available only as migration compatibility routes during 2.x and emit the documented warnings. [Migration](docs/migration.md) details old-to-new mappings and limitations.

Documentation, examples, notebooks, and benchmark configurations teach the canonical surface. Maintained consumer coverage checks configuration semantics and executable workflows. Unusable extras-head examples and a notebook requiring an unprovided private database were removed with their unsupported scope made explicit. Numerical cleanup repairs recursive PyTorch module traversal for state controllers and tensor-valued dictionary handling in linear probes. Geometric-sparse preprocessing now uses the composed output width in construction and checkpoint reconstruction.

Bash helpers install the two distributions separately and use canonical LM commands and nested model policies. Their configured activation initialization, dimensions, datasets, and training budgets are preserved. `python -m psannlm sft` delegates to the existing prompt/response fine-tuning implementation.

Both distributions use version 2.0.0. The LM distribution directly requires `psann>=2.0.0` and declares its numerical, data, tokenizer, and YAML dependencies. Core-only installations do not contain LM code; distribution contents remain disjoint. Static checks cover the configured repository/source scope without a retained debt baseline.

New core and LM checkpoints record the current package version, including LM source use without installed distribution metadata. The release helper advances both package versions, runtime version sources, and the LM core dependency floor together. LM budget and stream-exhaustion output uses portable ASCII approximation markers.

Wave estimator cloning and parameter updates preserve an absent context policy. Compatibility constructors issue one caller warning when flat preprocessing is also supplied. Parameter counting routes residual and wave options correctly. Allocation examples declare one output per asset so their softmax represents a portfolio rather than a single constant weight.

The existing `v1.0.0` tag remains an unchanged historical repository marker. Version 2.0.0 starts the canonical release track.
