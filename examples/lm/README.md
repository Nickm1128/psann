# Language-model examples

Run these files from a source checkout with both distributions installed. The canonical API is `PSANNLM` with `LMConfig`/`LMArchitectureConfig` and `PSANNLMDataPrep`; the CLI is `python -m psannlm`.

- [Small complete workflows](../quickstarts.py): three training steps, generation, and two model checkpoint generations. Run `python examples/quickstarts.py --workflow lm`.
- [Minimal train](minimal_train.py): corpus repetition, prompts, logged generations, and trainer checkpoints. Run `python examples/lm/minimal_train.py --epochs 1 --repeat 1 --out outputs/lm-example`.
- [Generation](generate.py): train a small local model then sample text.
- [CPU configuration](configs/waveresnet_cpu.yaml), [larger configuration](configs/waveresnet_small.yaml), and [distributed configuration](configs/waveresnet_3b_fsdp.yaml): canonical YAML task inputs. Distributed training requires adequate accelerator memory and the requested process topology.
- [Tiny corpus configuration](configs/tiny_corpus_benchmark.yaml): generate its declared corpus with `python scripts/make_tiny_corpus.py --help` and the documented output path before training.

The other YAML files are benchmark configurations for `python scripts/bench_lm_bases.py --config PATH`. A `models` mapping names complete canonical LM configurations, and `model_overrides` applies shared canonical fields. A `sweep` maps dotted configuration paths to value lists; the runner expands their Cartesian product. These are source-checkout benchmark files, not input to `python -m psannlm train`. See [benchmark semantics](../../docs/benchmarks/lm_base_sweeps.md).

Configuration names retain historical experiment identifiers. Their current model mappings preserve the original dimensions, datasets, training budgets, and expanded sweep values; benchmark results should record the actual package version and resolved configuration.
