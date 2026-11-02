# PSANN-LM exploration snapshot, 2026-11-02

This frozen research snapshot records the selection of a fresh 304,518,920-parameter model. It is an early-learning comparison, not evidence that the final model outperforms the previous pretrained model or is ready for distribution.

## Study

The study ran 33 trials with the same saved 50,257-token tokenizer. Each stage compared candidates at equal training and validation token counts within that stage:

| Stage | Trials | Training tokens per trial | Context | Validation tokens |
| --- | ---: | ---: | ---: | ---: |
| Approximately 29M-parameter scout: four candidates, three learning rates | 12 | 524,288 | 256 | 32,768 |
| Small-model confirmation: three candidates, three seeds | 9 | 4,194,304 | 512 | 65,536 |
| Full-size pilots: four candidates, two learning rates | 8 | 1,048,576 | 2,048 | 32,768 |
| Full-size finalists: two candidates, two additional seeds | 4 | 1,048,576 | 2,048 | 32,768 |

The full-size finalists combine the pilot seed 1337 with seeds 2027 and 3407. The wider wave model reached mean validation loss **6.64008 ± 0.00531** (sample standard deviation), compared with **6.74706 ± 0.04197** for the tuned reference. It has 0.0152% fewer parameters than the old 304,565,264-parameter model. See [finalist measurements](fullsize_confirmed_shortlist.json) and the archived YAML configurations.

The confirmation pool uses 50,000 documents from FineWeb-Edu shard 013, excluding the scout documents; documents are split by hash before token packing. Another 2,000 documents are reserved for final evaluation. Production training excludes all 61,999 scout, confirmation, and reserved-test document hashes. The pinned dataset revision and exclusion-manifest digest are in the [training recipe](replacement_training_recipe.json). Independence from the old model's historical training data is not established.

Exploration rejects empirically noncausal candidates. Native wave temporal interleave/replace modes read future tokens through symmetric convolutions and were excluded. Native behavior was preserved. See the [exploration workflow](../../docs/benchmarks/lm_base_sweeps.md).

## Selected training and runtime

The selected architecture uses wave PSANN activations, width 1,280, eight layers, 20 attention heads, MLP width 6,016, RoPE, SDPA, frequency initialization 2.25, decay initialization 0.001, and disabled temporal convolution. The fresh training schedule targets 5,999,951,872 tokens with BF16, peak learning rate 0.0006, 500 warmup steps, and 131,072 tokens per optimizer update. The initial stage has a 24-hour limit and saves resumable checkpoints; final evaluation and instruction tuning remain separate steps.

The [compilation benchmark](compile_benchmark.json) measured 7,665 eager versus 16,438 compiled tokens/s on NVIDIA GB10, a 2.14× steady optimizer-step speedup. It used identical starting weights and checked loss sequences within tolerance; it excludes data loading, validation, and checkpoint overhead. The [environment](cuda_environment.json) records the CUDA assembler override, precision-emulation settings, and local Python-header setup. These machine-specific paths require adjustment on another system.

The [archived training driver](train_selected.py) is an exact copy of the locally validated research driver, not a new installed package command. [Driver validation](selected_training_driver_validation.json) records CPU stop/resume parity and checkpoint retention. It imports source-checkout exploration helpers and requires the original tokenizer/token cache and exclusion list. Do not launch a second process against an active run directory.

## Provenance and limits

- [Snapshot manifest](snapshot_manifest.json) hashes the archived measurements, configurations, source manifest, and driver. The recipe and driver are byte-for-byte copies of the files used to launch training.
- [Runtime source manifest](source_manifest_20261102.json) records the pre-feature base commit and the hashes of the actual core/LM Python sources used. The package metadata still read 2.0.1 at launch; that version alone does not identify this uncommitted runtime.
- Paths inside archived JSON/YAML are historical local paths. Dataset files, token caches, document exclusion lists, checkpoints, logs, and environments are excluded from Git. This snapshot identifies those inputs but is not a standalone reproduction bundle.
- The running job continues to use its original local files. A later package release does not retroactively change its provenance. Native weights require held-out evaluation and instruction tuning before distribution.
- Feature commit `29f4381` captures the recorded runtime sources. Subsequent Python compatibility and CI repairs were prepared in a separate checkout; the active training checkout remains pinned to that feature commit until the job ends.
