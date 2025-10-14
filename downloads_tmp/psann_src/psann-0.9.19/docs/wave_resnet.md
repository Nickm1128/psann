**WaveResNet Backbone**
- WaveResNet stacks sine-activated residual blocks with SIREN-style initialisation to build compact, interference-friendly encoders. Each block applies `Linear → phase shift → sin → FiLM → Linear → norm → Dropout` and adds the result back to the skip connection. The sine nonlinearity is scaled by `w0` to control frequency content, while FiLM and phase shift layers let a context vector steer constructive/destructive interference.
- Residual updates follow `h_{l+1} = h_l + f_l(h_l, c)`, where `f_l` is the modulated sine block. Initialisation uses `weight ~ U(-√(6/f_in)/w0, √(6/f_in)/w0)` with zero bias to match SIREN recommendations (`w0=30` for the stem, 1–6 deeper).
- Norm options: `none` (plain residual), `weight` (WeightNorm on the block linears), `rms` (feature RMSNorm after the second linear). Dropout is applied on the residual branch before merging.
- The context pathway is optional. When provided, FiLM learns gamma/beta scaling and phase shift learns `φ(c)` added just before the sine activations; both can be toggled independently.
- Example:
  ```python
  import torch
  from psann.models import WaveResNet

  model = WaveResNet(
      input_dim=2,
      hidden_dim=128,
      depth=24,
      output_dim=1,
      context_dim=8,
      use_film=True,
      use_phase_shift=True,
      norm="rms",
      first_layer_w0=30.0,
      hidden_w0=2.0,
  )
  x = torch.randn(64, 2)
  c = torch.randn(64, 8)
  y = model(x, c)
  ```
- Guidance:
  - Increase `first_layer_w0` (e.g. 30) to admit high-frequency detail; keep `hidden_w0` within 1–6 for stability.
  - Phase-only steering: disable `use_film` but keep `use_phase_shift`. Amplitude-only: disable `use_phase_shift`.
  - `norm="rms"` stabilises deep stacks (24–64 layers) without erasing phase information; `weight` suits regimes needing implicit gating.
  - Apply branch dropout (0.0–0.1) when overfitting appears; the residual skip preserves low-frequency channels.
