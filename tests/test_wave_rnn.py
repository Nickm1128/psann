from __future__ import annotations

import torch

from psann.models import WaveRNNCell, scan_regimes


def test_wave_rnn_scan_regimes_varies_attractors() -> None:
    torch.manual_seed(0)
    cell = WaveRNNCell(hidden_dim=6, context_dim=2, alpha=0.1, w0=1.5)
    contexts = torch.tensor([[0.0, 0.0], [1.0, -1.0], [2.5, 0.5]], dtype=torch.float32)
    results = scan_regimes(cell, contexts, steps=128, burn_in=32)

    assert len(results) == contexts.size(0)
    attractor_counts = {item["attractor_count"] for item in results}
    lyapunov_values = [item["lyapunov"] for item in results]

    assert any(count > 1 for count in attractor_counts)
    assert any(value > 0 for value in lyapunov_values)
    assert any(value < 0 for value in lyapunov_values)
    for item in results:
        traj = item["trajectory"]
        assert traj.ndim == 2
        assert traj.shape[-1] == cell.hidden_dim
