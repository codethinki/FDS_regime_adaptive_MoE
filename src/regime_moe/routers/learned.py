import torch
from torch import nn

from regime_moe.config import LearnedGateConfig
from regime_moe.routers.base import Router


class LearnedRouter(Router):
    """Neural gate softmax(MLP(x)), trained jointly with the experts; no regime labels."""

    def __init__(self, n_experts: int, n_features: int, cfg: LearnedGateConfig):
        super().__init__(n_experts)
        self.gate = nn.Sequential(
            nn.Linear(n_features, cfg.hidden),
            nn.ReLU(),
            nn.Linear(cfg.hidden, n_experts),
        )

    def weights(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.gate(x), dim=-1)

    def parameters(self) -> list[nn.Parameter]:
        return list(self.gate.parameters())

    def to(self, device: str) -> "LearnedRouter":
        self.gate.to(device)
        return self
