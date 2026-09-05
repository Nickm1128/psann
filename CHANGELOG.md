# Changelog

## 0.13.0 — prepared in source

This release candidate consolidates the public interface around `PSANNRegressor` with immutable architecture and preprocessing policies, `EpisodicTrainer` with `HISSOConfig`, and the separate `PSANNLM`/`PSANNLMDataPrep` task API with `LMConfig` and `LMArchitectureConfig`. `python -m psannlm` is the canonical LM command.

Architecture validation rejects incompatible policies before numerical execution. Preprocessing composes frozen, jointly trained, and reconstruction-pretrained components with supervised and episodic tasks. LM construction has four typed kinds, including spectral behavior as a residual policy, and uses shared documented core primitives.

Core schema-v3 and LM schema-v1 persistence preserve accepted migration routes and repeated new-format reconstruction. Compatibility imports, constructors, flat inputs, and CLI shims remain available through 0.x with the documented warnings. [Migration](docs/migration.md) details old-to-new mappings and limitations.

Documentation, examples, notebooks, and benchmark configurations teach the canonical surface. Maintained consumer coverage checks configuration semantics and executable workflows. Unusable extras-head examples and a notebook requiring an unprovided private database were removed with their unsupported scope made explicit. Numerical cleanup repairs recursive PyTorch module traversal for state controllers and tensor-valued dictionary handling in linear probes. Geometric-sparse preprocessing now uses the composed output width in construction and checkpoint reconstruction.

Both distributions use version 0.13.0. The LM distribution directly requires `psann>=0.13.0` and declares its numerical, data, tokenizer, and YAML dependencies. Core-only installations do not contain LM code; distribution contents remain disjoint. Static checks cover the configured repository/source scope without a retained debt baseline.

Wave estimator cloning and parameter updates preserve an absent context policy. Compatibility constructors issue one caller warning when flat preprocessing is also supplied. Parameter counting routes residual and wave options correctly. Allocation examples declare one output per asset so their softmax represents a portfolio rather than a single constant weight.

The existing `v1.0.0` tag is a historical repository marker. Package metadata resumes at 0.13.0; that tag has not been rewritten or moved. These notes describe source preparation, not a published upload or newly created tag.
