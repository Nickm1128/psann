**Diagnostics Workflow**
- The diagnostics module provides quick checks on conditioning and representational richness without full training. Typical usage:
  ```python
  import torch
  from psann.models import WaveResNet
  from psann.utils import jacobian_spectrum, ntk_eigens, participation_ratio, mutual_info_proxy

  torch.manual_seed(42)
  model = WaveResNet(input_dim=2, hidden_dim=64, depth=12, output_dim=1, context_dim=4)
  x = torch.randn(32, 2)
  c = torch.randn(32, 4)

  jac = jacobian_spectrum(model, x, c, topk=8)
  ntk = ntk_eigens(model, x, c, topk=8)
  feats = model.forward_features(x, c).detach()
  pr = participation_ratio(feats)
  mi = mutual_info_proxy(feats, c)
  ```
- Interpretations:
  - `jacobian_spectrum`: reports leading eigenvalues of `JᵀJ`. A steep condition number hints at stiffness; adjust `w0`, RMSNorm, or dropout to smooth it.
  - `ntk_eigens`: eigen-spectrum of the empirical neural tangent kernel `JJᵀ`. A flat spectrum usually means poor diversity; FiLM/phase shift often widen it.
  - `participation_ratio`: `(∑λ)^2 / ∑λ²`. Higher is better—more effective dimensions in the feature covariance.
  - `mutual_info_proxy`: HSIC-based score between features and contexts. Larger values imply stronger context binding; near-zero indicates the encoder ignores `c`.
- Practical tips:
  - Always run in `torch.no_grad()` when only inspecting features (PR/MI) to save memory; keep gradients for Jacobian/NTK.
  - Use small batches (8–32) for Jacobian/NTK to keep autograd cost manageable.
  - Track spectra during depth/width sweeps; a sudden collapse in top eigenvalues usually precedes gradient issues.
