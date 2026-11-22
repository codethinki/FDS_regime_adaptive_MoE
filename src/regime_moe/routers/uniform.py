import torch

from regime_moe.routers.base import Router


class UniformRouter(Router):
    """Every expert gets weight 1 / n_experts."""

    def weights(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return torch.full((len(x), self.n_experts), 1.0 / self.n_experts, device=x.device)
