# Diagnostics Workflow

Use the diagnostics helpers to evaluate feature quality before committing to long training runs. The snippet below mirrors the recommended entry points.

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

- `jacobian_spectrum`: reports the leading eigenvalues of `J^T J`. A steep condition number hints at stiffness; adjust `w0`, RMSNorm, or dropout to smooth it.
- `ntk_eigens`: eigen-spectrum of the empirical neural tangent kernel `J J^T`. A flat spectrum usually signals poor diversity; FiLM or phase-shift blocks can widen it.
- `participation_ratio`: evaluates `(sum_i lambda_i)^2 / sum_i lambda_i^2`. Higher scores imply more effective feature dimensions.
- `mutual_info_proxy`: HSIC-style score between features and contexts. Near-zero values indicate the encoder is ignoring `c`.

Practical tips:
- Wrap feature-only probes (`participation_ratio`, `mutual_info_proxy`) in `torch.no_grad()` to save memory.
- Keep Jacobian and NTK batches small (8-32 samples) to bound autograd cost.
- Track spectra during depth or width sweeps; a sudden collapse in top eigenvalues typically precedes gradient issues.
