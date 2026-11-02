# LM architecture benchmarks

Run `python scripts/bench_lm_bases.py --config examples/lm/configs/base_compare_quick.yaml --dry-run` from a source checkout. The benchmark runner trains, evaluates next-token loss/accuracy, and saves a canonical model plus trainer artifacts. External datasets/tokenizers are explicit prerequisites; dry-run prints a plan without downloading or training.

Each entry under `models` is a tagged `LMConfig` mapping. Labels are experiment identifiers; `architecture.kind` establishes one of the four supported LM kinds. `spectral-residual` is a label for residual plus spectral policy. Shared `model_overrides` are deep-merged before strict normalization. `train` contains optimizer/runtime settings, and `data` defines corpus and sequence length.

A `sweep` is a Cartesian product of dotted paths and value lists, for example `train.lr: [0.001, 0.002]` or `model_overrides.architecture.activation.frequency_init: [1.0, 2.0]`. Complete initialization policies can be sweep values, including null for no extra initialization. Deep merges do not mutate source mappings. `--models` selects labels from the file; `--seeds` selects seeds. Both spaced and equals long-option forms work.

The maintained configuration contracts check 119 expanded variants across 14 benchmark YAML files against exact normalized parameter fingerprints. Renaming a policy must not change an experiment's model dimensions, activation initialization, optimizer settings, token budget, or data selection. The transformer comparison retains its originally executed activation even when an old inactive setting suggested otherwise.

Outputs include resolved configuration, system identity, metrics, model/trainer checkpoints, tokenizer metadata, and a leaderboard. Compare validation token counts and tokenizer identity before comparing perplexities; parameter matching and throughput alone do not establish scientific equivalence. Distillation runs preserve the configured teacher/student architectures and objective mixture.

## Bounded exploration

The [2026-11-02 GPU study snapshot](../../benchmarks/lm_exploration_20261102/README.md) records the tested candidates, full-size selection, frozen training recipe, and runtime source hashes. Its short pilots select a provisional candidate; final training quality remains to be established.

[explore_lm.py](../../scripts/explore_lm.py) adds cheap initialization probes and short training trials using the canonical model registry and native trainer. Run from the repository root with core PSANN, PSANN-LM, PyTorch, and PyYAML installed. It requires no Hugging Face export dependencies. The workflow is single process on CPU or one CUDA device.

Start with the synthetic preset to check the environment and output contracts:

```bash
python scripts/explore_lm.py init --preset smoke --out reports/exploration/smoke.yaml
python scripts/explore_lm.py prepare --config reports/exploration/smoke.yaml --out reports/exploration/smoke_tokens.pt
python scripts/explore_lm.py plan --config reports/exploration/smoke.yaml --cache reports/exploration/smoke_tokens.pt --out reports/exploration/smoke_plan.json
python scripts/explore_lm.py probe --config reports/exploration/smoke.yaml --cache reports/exploration/smoke_tokens.pt --out reports/exploration/smoke_probes.json
python scripts/explore_lm.py run --config reports/exploration/smoke.yaml --cache reports/exploration/smoke_tokens.pt --out reports/exploration/smoke_run --max-seconds 120
python scripts/explore_lm.py report --out reports/exploration/smoke_run
```

The smoke preset trains three tiny models for 12 steps: a GELU transformer control, default wave PSANN, and a wave PSANN activation candidate with frequency 2.25 and decay 0.001. These are candidate settings, not validated improvements. Synthetic losses test the tools and cannot choose a production architecture. Output files must be new; run directories must be absent or empty. Interrupted runs retain curves and metrics but do not resume. A run exits with status 2 if any trial fails, reaches its budget, or remains unstarted.

For real data, generate a separate configuration with `init --preset fineweb-scout`. Its defaults expect local FineWeb-Edu JSONL and a fitted tokenizer; inspect and edit the emitted paths before `prepare`. No data is downloaded. The preset compares four approximately 29M parameter proxies, including a frequency-spread initialization candidate, across three learning rates. It uses 256 steps per trial, 2,048 tokens per step, and a two-hour total wall-time budget. CPU runs force FP32; CUDA runs support FP32 or BF16 and require a CUDA-enabled PyTorch environment. Use `--device cuda` for GPU probes and runs. The wall-time limit is checked between optimizer steps and trials; a forward pass, validation, or checkpoint write can overrun it.

`data.files` accepts UTF-8 line-oriented text or JSONL using `data.text_key` (default `text`); `data.texts` accepts inline documents. Preparation bounds documents and token counts, removes exact duplicates after trimming whitespace, and assigns whole documents to deterministic hash-based train/validation splits. The splits are packed separately into fixed-length blocks with already shifted labels. A simple tokenizer fits only training documents. Other backends require saved tokenizer assets and explicit backend selection. The cache contains tokenizer state, document hashes, configuration identity, and a content hash. Changing data/tokenizer settings requires a new cache; architecture, optimizer, and seed sweeps reuse the same tokens. Repeated cache exposure is reported explicitly.

The FineWeb preset repurposes a local sample as a newly partitioned pool for fresh models. Its validation split is disjoint from its own training split; it does not establish independence from the old pretrained checkpoint or the wider original training corpus. Keep a separate untouched evaluation corpus for final model selection and distribution claims.

Use `plan --target` to count parameters with `exploration.target_dimensions` before allocating a full training run. The initial target is the old model's 304,565,264 parameters, with dimensions 1,024, 16 layers, 16 heads, and MLP width 4,096. The plan flags candidates outside a 5% parameter band and records a lower bound for FP32 AdamW parameter/gradient/state memory, excluding activation and workspace memory. A target plan constructs models on CPU but performs no optimization. `probe --target --limit 1` measures one full-size forward/backward pass without training a full model; choose a suitable batch/context length first.

Probe JSON records bounded samples of PSANN inputs, outputs, effective amplitudes/frequencies/decays, phases, analytic activation derivatives, heavily damped fractions, and tiny-derivative fractions. Gradient norms and gradient-to-weight ratios separate activation parameters, embeddings, readout, and backbone. A final-feature participation ratio provides a bounded covariance diagnostic. These measurements can expose vanishing derivatives, explosive gradients, or collapsed representations; they are diagnostics rather than predictions of final quality. Probe timing includes hooks and excludes optimizer updates. Probes preserve weights and random state and clear their diagnostic gradients.

Each trial first checks prefix logits for future-token dependence and context-length dependence, then repeats that check after training. A failed check excludes the trial from selection. This is an empirical guard, not a proof of causality. In particular, the current interleaved/replacement temporal convolution uses symmetric padding and fails the guard; its runtime behavior is preserved, but it is excluded from the default search.

Each trial saves its resolved configuration, initial/final causality checks and activation probes, step-by-step loss and token counts, fixed-prefix validation curves, trainer checkpoint, and a native model with tokenizer. Validation and training ordering use independent random generators. The optional `LMTrainer.train(..., on_step=callback)` observer receives `step`, accumulated `tokens`, `loss`, and the applied `learning_rate` on the main rank after each optimizer update and before gradients are cleared. Gradient snapshots at validation intervals therefore reflect any configured clipping. Observer exceptions propagate to stop a bounded trial. Omitting the observer preserves ordinary training behavior.

The report includes CSV and Markdown leaderboards plus JSON summaries and shortlists. It excludes failed, incomplete, and nonfinite trials. Pareto comparisons use equal training and validation token counts; throughput includes trainer setup, validation callbacks, and checkpoint writes. Seed aggregates report mean and sample standard deviation. Edit `bench.seeds` to replicate promising settings, and use the existing dotted-path `sweep` syntax to explore canonical activation, initialization, temporal, geometry, and optimizer policies. Compare loss at equal token budgets, inspect learning curves, and confirm candidates at increasing depth/context/token budgets before choosing a full-size run. A small-model winner is a provisional candidate rather than an established optimum.

Equal width and depth do not guarantee equal parameter counts across architectures. With the local 50,257-token vocabulary, the target audit counts 304,565,264 parameters for default/tuned wave and 321,473,552 for interleaved temporal wave. The latter exceeds the preset's 5% band and needs a narrower MLP or another explicit dimension adjustment before a same-size comparison.

After quantitative selection, use the local [instruction prompts](../../scripts/prompts_sft_oasst1.txt) for consistent qualitative checks, decoding their literal `\\n` escapes when constructing prompts. Keep pretraining and instruction tuning as distinct stages and assess both repetition and instruction following before distributing the replacement model.
