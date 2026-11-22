"""Generic training loop; knows nothing about specific routers, experts or objectives."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from regime_moe.config import TrainConfig
from regime_moe.models import BigMoE
from regime_moe.objectives import Objective
from regime_moe.routers import Router


def train_moe(
    moe: BigMoE,
    router: Router,
    objective: Objective,
    x: torch.Tensor,
    y: torch.Tensor,
    idx: torch.Tensor,
    cfg: TrainConfig,
    seed: int,
    device: str = "cpu",
) -> list[float]:
    """x: features at t, y: asset returns at t+1, idx: positions of the rows in the full sequence.

    Returns the mean loss per epoch.
    """
    params = list(moe.parameters()) + router.parameters()
    optimizer = torch.optim.Adam(params, lr=cfg.learning_rate)
    loader = DataLoader(
        TensorDataset(idx, x, y),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    history = []
    moe.train()
    for _ in range(cfg.epochs):
        losses = []
        for idx_b, x_b, y_b in loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            weights = moe(x_b, router.weights(x_b, idx_b))
            loss = objective((weights * y_b).sum(dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        history.append(sum(losses) / len(losses))
    return history
