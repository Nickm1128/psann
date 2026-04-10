# ruff: noqa: F403,F405
from __future__ import annotations

from .shared import *


class PSANNConvSpine(nn.Module):
    def __init__(
        self,
        in_ch: int,
        hidden: int,
        depth: int,
        kernel_size: int,
        horizon: int,
        aggregator: str = "last",
    ):
        super().__init__()
        self.aggregator = aggregator
        self.core = PSANNConv1dNet(
            in_channels=in_ch,
            out_dim=hidden,
            hidden_layers=depth,
            conv_channels=hidden,
            hidden_channels=hidden,
            kernel_size=kernel_size,
            segmentation_head=True,
        )
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.core(x.transpose(1, 2))
        pooled = features[:, :, -1] if self.aggregator == "last" else features.mean(dim=-1)
        return self.head(pooled)


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: int, depth: int, out_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.size(0), -1)
        return self.net(flat)
