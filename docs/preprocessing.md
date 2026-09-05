# Preprocessing composition

Pass `preprocessor=PreprocessorConfig(...)` to `PSANNRegressor`. The policy separates the component's reconstruction pretraining from its participation in predictive training.

```python
from psann import PSANNRegressor
from psann.architectures import ArchitectureConfig
from psann.preprocessing import (
    LSMConfig, LSMPretrainingConfig, PreprocessorConfig, PreprocessorTrainingConfig,
)
preprocessor = PreprocessorConfig(
    LSMConfig.dense(
        output_dim=16, hidden_layers=2, hidden_units=32, sparsity=0.8,
        random_state=7, pretraining=LSMPretrainingConfig(epochs=5, lr=0.001),
    ),
    training=PreprocessorTrainingConfig(trainable=True, lr=0.0005),
)
model = PSANNRegressor(architecture=ArchitectureConfig.dense(), preprocessor=preprocessor)
```

The default training policy freezes the component during predictive training. `trainable=True` enables joint updates; `lr` selects the component optimizer-group rate. `LSMPretrainingConfig` controls reconstruction epochs, optimizer rate, ridge, batching, validation, and regularization. Zero pretraining epochs keeps initialization without reconstruction updates. A preset `LSMConfig.convolutional(...)` preserves spatial preprocessing; its output must match the selected core topology.

Custom torch modules use `ModulePreprocessorConfig`. Their output contract and persistence support are validated at the boundary. Use a registered/importable reconstruction route for portable custom artifacts; do not assume arbitrary local classes can be reconstructed from a fresh process.

Preprocessing is applied once through supervised fit, episodic fit, prediction, and checkpoint reconstruction. Fit and validation preserve their raw-input boundary. See [the supervised example](../examples/14_psann_with_vs_without_lsm.py), [episodic composition](../examples/27_hisso_lsm_allocation.py), and [executable round trips](../examples/quickstarts.py).
