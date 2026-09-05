# LM architecture benchmarks

Run `python scripts/bench_lm_bases.py --config examples/lm/configs/base_compare_quick.yaml --dry-run` from a source checkout. The benchmark runner trains, evaluates next-token loss/accuracy, and saves a canonical model plus trainer artifacts. External datasets/tokenizers are explicit prerequisites; dry-run prints a plan without downloading or training.

Each entry under `models` is a tagged `LMConfig` mapping. Labels are experiment identifiers; `architecture.kind` establishes one of the four supported LM kinds. `spectral-residual` is a label for residual plus spectral policy. Shared `model_overrides` are deep-merged before strict normalization. `train` contains optimizer/runtime settings, and `data` defines corpus and sequence length.

A `sweep` is a Cartesian product of dotted paths and value lists, for example `train.lr: [0.001, 0.002]` or `model_overrides.architecture.activation.frequency_init: [1.0, 2.0]`. Complete initialization policies can be sweep values, including null for no extra initialization. Deep merges do not mutate source mappings. `--models` selects labels from the file; `--seeds` selects seeds. Both spaced and equals long-option forms work.

The maintained configuration contracts check 119 expanded variants across 14 benchmark YAML files against exact normalized parameter fingerprints. Renaming a policy must not change an experiment's model dimensions, activation initialization, optimizer settings, token budget, or data selection. The transformer comparison retains its originally executed activation even when an old inactive setting suggested otherwise.

Outputs include resolved configuration, system identity, metrics, model/trainer checkpoints, tokenizer metadata, and a leaderboard. Compare validation token counts and tokenizer identity before comparing perplexities; parameter matching and throughput alone do not establish scientific equivalence. Distillation runs preserve the configured teacher/student architectures and objective mixture.
