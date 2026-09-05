# Migrating to the canonical 0.13.0 API

Version 0.13.0 makes the new task/configuration API authoritative. Existing direct imports and constructor routes remain available throughout the remaining 0.x line. They are compatibility facades, not additional recommended estimators or LM kinds. No 1.x compatibility removal occurs here.

The repository's `v1.0.0` tag is historical. Both package metadata versions resume at **0.13.0**, and the existing tag is not moved, deleted, or recreated. This source tree prepares artifacts; it does not claim an upload or a new release tag.

## Regression constructors

Import nested policies from `psann.architectures`. The direct legacy imports below are intentionally shown only for migration. Each legacy constructor emits one caller-located `DeprecationWarning`; warnings may be hidden by Python's default warning filter. Use `python -W default ...` to display them.

```python
from psann import (
    PSANNRegressor, ResPSANNRegressor, ResConvPSANNRegressor,
    SGRPSANNRegressor, WaveResNetRegressor, GeoSparseRegressor,
)
from psann.architectures import (
    ArchitectureConfig, ActivationConfig, ResidualConfig, ConvolutionConfig,
    SequenceConfig, SpectralConfig, GeometryConfig, WaveConfig, StateConfig, W0WarmupConfig,
)

# Each pair describes the same architecture intent; training options remain shared.
pairs = [
    (PSANNRegressor(activation_type="relu"),
     PSANNRegressor(architecture=ArchitectureConfig.dense(activation=ActivationConfig(kind="relu")))),
    (ResPSANNRegressor(residual_alpha_init=0.2),
     PSANNRegressor(architecture=ArchitectureConfig.dense(residual=ResidualConfig(alpha_init=0.2)))),
    (ResConvPSANNRegressor(conv_kernel_size=3),
     PSANNRegressor(architecture=ArchitectureConfig.convolutional(
         residual=ResidualConfig(), convolution=ConvolutionConfig(kernel_size=3)))),
    (SGRPSANNRegressor(k_fft=8),
     PSANNRegressor(architecture=ArchitectureConfig.for_sequence(spectral=SpectralConfig(k_fft=8)))),
    (WaveResNetRegressor(dropout=0.1),
     PSANNRegressor(architecture=ArchitectureConfig.for_wave(wave=WaveConfig(
         dropout=0.1, warmup=W0WarmupConfig(first_initial=10.0, hidden_initial=0.5, epochs=10))))),
    (GeoSparseRegressor(shape=(4, 4), k=8),
     PSANNRegressor(architecture=ArchitectureConfig.geometric_sparse(geometry=GeometryConfig(shape=(4, 4), k=8)))),
]
```

`hidden_width` becomes `hidden_units`. Flat `activation_type` and activation dictionaries become `ActivationConfig`; dictionary aliases such as `layout` become `mix_layout`. Flat attention/state/context values become their nested policies. Stateful dense behavior uses `ArchitectureConfig.dense(state=StateConfig(...))`; residual plus state is unsupported. Spatial settings become `ConvolutionConfig`. Wave warmup/progressive depth become `WaveConfig(warmup=W0WarmupConfig(...), progressive_depth=ProgressiveDepthConfig(...))`. Do not combine a canonical architecture with flat architecture keywords: conflicts reject early.

Canonical preset strings include dense, residual, convolutional, residual-convolutional, wave, sequence, and geometric-sparse. Historical preset spellings such as `respsann`, `waveresnet`, and `geosparse` remain warning adapters. The typed core has five kinds; residual is a dense/convolutional policy, not another kind. Direct imports from `psann`, `psann.sklearn`, and the compatibility estimator modules remain usable; legacy estimator names are absent from canonical `__all__`.

Top-level historical `ActivationConfig`, `AttentionConfig`, and `StateConfig` imports retain their previous meanings. Import their canonical immutable policy counterparts from `psann.architectures`; historical names are omitted from the canonical top-level wildcard surface.

## LSM composition

The old `lsm`, `lsm_train`, `lsm_pretrain_epochs`, and `lsm_lr` arguments normalize once into `preprocessor`. Retain explicit old values when migrating; a pretraining `epochs` value inside an old component mapping takes precedence over the flat default.

```python
from psann import PSANNRegressor
from psann.architectures import ArchitectureConfig
from psann.preprocessing import LSMConfig, LSMPretrainingConfig, PreprocessorConfig, PreprocessorTrainingConfig

old = PSANNRegressor(
    lsm={"type": "lsmexpander", "output_dim": 16, "hidden_layers": 2, "hidden_units": 32},
    lsm_train=True, lsm_pretrain_epochs=5, lsm_lr=0.0005,
)
new = PSANNRegressor(
    architecture=ArchitectureConfig.dense(),
    preprocessor=PreprocessorConfig(
        LSMConfig.dense(output_dim=16, hidden_layers=2, hidden_units=32,
                        pretraining=LSMPretrainingConfig(epochs=5)),
        training=PreprocessorTrainingConfig(trainable=True, lr=0.0005),
    ),
)
```

| Old component route | Canonical composition |
| --- | --- |
| `lsm={"type": "lsm", ...}` or `lsm={"type": "lsmexpander", ...}` | `PreprocessorConfig(LSMConfig.dense(...))` |
| `lsm={"type": "lsmconv2d", ...}` or `lsm={"type": "lsmconv2dexpander", ...}` | `PreprocessorConfig(LSMConfig.convolutional(...))` |
| Existing `LSM` module | `PreprocessorConfig(ModulePreprocessorConfig(module=component, input_topology="flat", output_topology="flat", output_dim=...))` |
| Fitted `LSMExpander` instance | Use its `component.model` module in the flat composition above |
| Existing `LSMConv2d` module | `PreprocessorConfig(ModulePreprocessorConfig(module=component, input_topology="spatial-2d", output_topology="spatial-2d", output_dim=...))` |
| Fitted `LSMConv2dExpander` instance | Use its `component.model` module in the spatial composition above |
| `lsm_train=False` | Default frozen `PreprocessorTrainingConfig` |
| `lsm_train=True`, `lsm_lr=...` | `PreprocessorTrainingConfig(trainable=True, lr=...)` |
| Reconstruction training fields | `LSMPretrainingConfig(...)` inside `LSMConfig` |

Import `ModulePreprocessorConfig` from `psann.preprocessing`. Explicit module topology and output width describe the module's actual output; they are not inferred from a class name. An unfitted expander has no graph yet: fit it first, or migrate its settings to `LSMConfig` for estimator-owned construction/pretraining. Direct old component imports and constructors remain usable. Conv preprocessing supports only its validated pretraining subset; do not silently copy unsupported dense-only options. Mixed old/new preprocessing representations either normalize equivalent values once or reject conflicts.

## Episodic training

```python
from psann import PSANNRegressor
from psann.architectures import ArchitectureConfig
from psann.episodic import EpisodicTrainer, HISSOConfig, EpisodeScheduleConfig

# Old fit route: estimator.fit(prices, hisso=True, hisso_window=8,
#     hisso_primary_transform="softmax", hisso_transition_penalty=0.001)
trainer = EpisodicTrainer(
    estimator=PSANNRegressor(architecture=ArchitectureConfig.dense(), output_shape=(2,)),
    strategy=HISSOConfig(
        schedule=EpisodeScheduleConfig(episode_length=8),
        reward="finance", primary_transform="softmax", transition_penalty=0.001,
    ),
)
# trainer.fit(prices); trainer.predict(prices); trainer.evaluate(prices)
```

Flat HISSO arguments and `HISSOOptions` remain compatibility inputs. The canonical trainer delegates to the accepted episodic numerical runtime; the estimator continues to own preprocessing and architecture. Schedule fields move into `EpisodeScheduleConfig`, reward/transform/penalty into `HISSOConfig`, and supervised warm start into `SupervisedWarmStartConfig`. Register portable reward names where appropriate. Custom reward callables and external context require the same semantics after load; unsupported callable descriptors reject rather than being silently dropped.

## LM APIs and base names

```python
from psannlm import PSANNLM, PSANNLMDataPrep, LMConfig, LMArchitectureConfig
from psann.architectures import SpectralConfig

architectures = {
    "transformer": LMArchitectureConfig.transformer(),
    "respsann": LMArchitectureConfig.residual(),
    "sgrpsann": LMArchitectureConfig.residual(spectral=SpectralConfig()),
    "waveresnet": LMArchitectureConfig.wave(),
    "geosparse": LMArchitectureConfig.geometric_sparse(),
}
model = PSANNLM(config=LMConfig(architecture=architectures["waveresnet"], d_model=32, n_layers=2, n_heads=4))
```

Lowercase `psannLM` and `psannLMDataPrep` remain direct warning facades. Replace them with `PSANNLM` and `PSANNLMDataPrep`. Old `base=` constructors remain compatibility adapters for all five base names above. The canonical high-level fit is `fit(data, train=TrainConfig(...))`; old flat fit arguments retain their historical behavior, including previously inactive fields, and warn with the affected names. Use canonical `TrainConfig` to activate those settings deliberately.

Preserve non-default architecture settings explicitly. Old sine amplitude/frequency/damping values map to `ActivationConfig` and any initialization spreads to `LMActivationInitializationConfig`. Spectral settings map to `SpectralConfig`; temporal convolution settings to `LMTemporalConfig`; geometry/depth/chunking to `GeometryConfig` and `LMGeometryExecutionConfig`. Raw backbone constructors, high-level APIs, and historical CLI defaults were not always identical; loading or normalizing an existing artifact is safer than assuming a preset reproduces all old defaults. Old benchmark configuration files remain readable; maintained YAML uses complete canonical model mappings.

Use `python -m psannlm train`, `resume`, `eval`, and `generate`. The legacy `python -m psannlm.train`, `python -m psannlm.cli`, and `python -m psannlm.lm.train.cli` modules remain warning shims. Legacy `--base` and `--sine-*` options remain accepted but are hidden from canonical help; choose `--architecture` or `--model-config`. Spaced and equals forms of long options are equivalent. Conflicting old/new fields and unknown policies reject before training.

## Checkpoint families

`python -m psannlm sft` delegates to the existing supervised fine-tuning runtime. The direct `python -m psannlm.sft` route remains usable. Source-checkout checkpoint diagnostics (`eval_ppl_sidecar.py`, `generate_from_trainer_ckpt.py`, `generate_from_ckpt_step6500.py`, `ppl_wikitext_psann.py`, and `finalize_bmrk01.py`) retain their historical evaluation or sampling settings; use the canonical CLI for new workflows. `profile_hisso.py` retains the old model-level trainer solely to reproduce its GELU-network timing workload; replacing its model would change that measurement.

| Input family | Supported route and result |
| --- | --- |
| Core unversioned/class-named, schema v1/v2 | `PSANNRegressor.load(path, map_location=...)`; normalizes architecture/preprocessing/episodic metadata; saving produces schema v3 |
| Core schema v3 | Same loader; repeated saves retain schema v3 and runtime behavior |
| Episodic core checkpoint | `EpisodicTrainer.load(path, map_location=...)`; reconstructs estimator and strategy |
| LM historical high-level checkpoint | `PSANNLM.load(path, map_location=...)`; normalizes the saved model description and tokenizer; saving produces model schema v1 |
| LM historical trainer checkpoint | `PSANNLM.load(...)` for inference where metadata is sufficient; trainer resume route preserves saved optimizer/step state |
| LM model schema v1 | `PSANNLM.load(...)`; canonical configuration, tokenizer, and model state |
| LM trainer schema v1 | `LMTrainer` resume or canonical CLI resume with matching configuration/data/tokenizer |
| LM raw weight dictionary | Supply explicit canonical model configuration and tokenizer identity to the supported CLI loading route; tensor shapes alone do not establish architecture semantics |

```python
import numpy as np
from psann import PSANNRegressor

def migrate_core_checkpoint(source, destinations, X):
    migrated = PSANNRegressor.load(source, map_location="cpu")
    expected = migrated.predict(X)
    for path in destinations:  # Supply two distinct new paths; preserve the source.
        migrated.save(path)
        migrated = PSANNRegressor.load(path, map_location="cpu")
        np.testing.assert_allclose(migrated.predict(X), expected, rtol=0, atol=0)
    return migrated
```

The [complete quickstarts](../examples/quickstarts.py) execute two successive generations for core, preprocessing, episodic, and LM tasks. CUDA artifacts can be loaded on CPU and CPU artifacts on a working CUDA runtime. Floating-point tolerances may be needed when comparing computations across devices; same-device persistence must preserve predictions/greedy tokens. Model-only exports do not invent missing optimizer resume state. Custom modules/callables must remain reconstructible under their supported descriptor/registration rules. Load only artifacts whose origin you trust.
