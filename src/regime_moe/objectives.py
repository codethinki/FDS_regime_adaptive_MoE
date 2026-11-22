"""Training objectives: portfolio returns of a batch -> scalar loss (lower is better)."""

from collections.abc import Callable

import torch

from regime_moe.config import TrainConfig

Objective = Callable[[torch.Tensor], torch.Tensor]


def mean_variance(risk_aversion: float) -> Objective:
    """-mean + risk_aversion * variance over the batch (one-step, no costs; from the demo)."""

    def loss(portfolio_returns: torch.Tensor) -> torch.Tensor:
        return -portfolio_returns.mean() + risk_aversion * portfolio_returns.var(unbiased=False)

    return loss


OBJECTIVES: dict[str, Callable[[TrainConfig], Objective]] = {
    "mean_variance": lambda cfg: mean_variance(cfg.risk_aversion),
}


def build_objective(cfg: TrainConfig) -> Objective:
    if cfg.objective not in OBJECTIVES:
        raise ValueError(f"unknown objective '{cfg.objective}', choose from {sorted(OBJECTIVES)}")
    return OBJECTIVES[cfg.objective](cfg)
