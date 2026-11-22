import torch
from torch import nn


class BigMoE(nn.Module):
    """Big architecture (Willi et al. 2024, Fig. 1e): each expert is a complete network.

    Works with any experts that map x -> portfolio weights. Routing weights come from
    outside (a Router), so the MoE itself holds no gate.
    """

    def __init__(self, experts: list[nn.Module]):
        super().__init__()
        self.experts = nn.ModuleList(experts)

    def expert_outputs(self, x: torch.Tensor) -> torch.Tensor:
        """Shape [batch, experts, assets]."""
        return torch.stack([expert(x) for expert in self.experts], dim=1)

    def forward(self, x: torch.Tensor, route: torch.Tensor) -> torch.Tensor:
        """route: [batch, experts], rows sum to 1 -> portfolio weights [batch, assets]."""
        return (self.expert_outputs(x) * route.unsqueeze(-1)).sum(dim=1)
