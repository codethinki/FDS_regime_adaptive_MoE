import numpy as np
import torch

from regime_moe.models import BigMoE
from regime_moe.routers import Router


@torch.no_grad()
def evaluate(
    moe: BigMoE,
    router: Router,
    x: torch.Tensor,
    y: torch.Tensor,
    idx: torch.Tensor,
    periods_per_year: int,
    device: str = "cpu",
) -> dict[str, float]:
    """Chronological evaluation; rows must be in time order. No transaction costs yet."""
    moe.eval()
    x, y = x.to(device), y.to(device)
    route = router.weights(x, idx)
    weights = moe(x, route)
    port = (weights * y).sum(dim=1).cpu().numpy()
    metrics = portfolio_metrics(port, weights.cpu().numpy(), periods_per_year)
    for k, w in enumerate(route.cpu().numpy().mean(axis=0)):
        metrics[f"route_expert_{k}"] = float(w)
    return metrics


def portfolio_metrics(port: np.ndarray, weights: np.ndarray, periods_per_year: int) -> dict[str, float]:
    """port: portfolio return per period, weights: [periods, assets]. No transaction costs."""
    mean, vol = float(port.mean()), float(port.std())
    equity = np.cumprod(1 + port)
    return {
        "ann_return": mean * periods_per_year,
        "ann_vol": vol * np.sqrt(periods_per_year),
        "sharpe": mean / (vol + 1e-12) * np.sqrt(periods_per_year),
        "max_drawdown": float((equity / np.maximum.accumulate(equity) - 1).min()),
        "turnover": float(np.abs(np.diff(weights, axis=0)).sum(axis=1).mean()),
    }


def equal_weight_metrics(y: np.ndarray, periods_per_year: int) -> dict[str, float]:
    """Benchmark: constant equal weights in every asset (rebalanced each period)."""
    weights = np.full_like(y, 1.0 / y.shape[1])
    return portfolio_metrics((weights * y).sum(axis=1), weights, periods_per_year)
