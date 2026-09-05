# Source-checkout utilities

Install core with `python -m pip install -e .`; add language modeling with `python -m pip install ./psannlm`. Run tools from the repository root and choose output paths outside tracked source. Each CLI provides `--help`; [consumer_manifest.json](../docs/consumer_manifest.json) records entrypoints and prerequisites.

## Training and benchmarking

- `python -m psannlm` owns LM train/resume/eval/generate/sft. [LM guide](../docs/lm.md) describes canonical configuration.
- `bench_lm_bases.py` compares named canonical model mappings; see [sweep semantics](../docs/benchmarks/lm_base_sweeps.md).
- `benchmark_hisso_variants.py` compares dense/convolutional episodic schedules and records measured reward and throughput.
- `benchmark_regressor_ablations.py`, `run_geosparse_vs_relu_benchmarks.py`, and `benchmark_geo_sparse_vs_dense.py` build core architecture policies for source-checkout experiments.
- `train_psannlm_chat.py` runs explicit pretraining/SFT stages; `gen_psannlm_chat.py` generates from their artifacts. Their dataset and tokenizer dependencies are optional source-tool prerequisites.
- `make_tiny_corpus.py` creates the declared local synthetic corpus. `fetch_benchmark_data.py` retrieves public price inputs for an explicitly configured experiment.

## Analysis and environment tools

`gpu_env_report.py` records hardware/runtime identity. `profile_psann.py` profiles bounded numerical workloads. `run_gpu_validation.py`, `run_cuda_tests.py`, and `run_gpu_tests.py` execute the requested CUDA tests; report actual hardware and counts.

`run_full_suite.py` coordinates source-checkout benchmarks. `postprocess_full_suite.py`, `aggregate_benchmarks.py`, `parse_trainer_log.py`, and `plot_loss_from_csv.py` summarize measured outputs. Use [result promotion guidance](../docs/benchmarks/promotion_guide.md) when retaining historical results. Do not write a new run over a tracked historical result file.

The manifest includes all Bash helpers. Consumer tests check their syntax and referenced files, execute embedded model configuration, and compare trained model behavior with the preserved experiment settings. Full RunPod workloads require their declared datasets and GPU resources; local validation does not run their full research budgets.

Compatibility helpers and the model-level HISSO profiler remain available for existing workflows; their scope and canonical replacements are described in [migration](../docs/migration.md). Release helper scripts are preparation tools; validation does not imply permission to upload, tag, or publish.
