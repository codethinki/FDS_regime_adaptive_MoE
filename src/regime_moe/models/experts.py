"""Expert networks. Contract: forward(x [batch, features]) -> portfolio weights [batch, assets],
non-negative and summing to 1 (long-only)."""

from collections.abc import Callable

import torch
from torch import nn

from regime_moe.config import ModelConfig


class MLPExpert(nn.Module):
    def __init__(self, n_features: int, n_assets: int, hidden: list[int]):
        super().__init__()
        layers, width = [], n_features
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        layers.append(nn.Linear(width, n_assets))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.network(x), dim=-1)


EXPERTS: dict[str, Callable[[int, int, ModelConfig], nn.Module]] = {
    "mlp": lambda n_features, n_assets, cfg: MLPExpert(n_features, n_assets, cfg.mlp.hidden),
}


def build_experts(cfg: ModelConfig, n_features: int, n_assets: int) -> list[nn.Module]:
    if cfg.expert not in EXPERTS:
        raise ValueError(f"unknown expert '{cfg.expert}', choose from {sorted(EXPERTS)}")
    return [EXPERTS[cfg.expert](n_features, n_assets, cfg) for _ in range(cfg.n_experts)]
