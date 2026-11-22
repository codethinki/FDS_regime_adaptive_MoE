from abc import ABC, abstractmethod

import numpy as np
import torch


class Router(ABC):
    """Maps a batch of periods to expert weights of shape [batch, n_experts] (rows sum to 1).

    Lifecycle: fit(train features) -> prepare(all features) -> weights(x, idx).
    ``idx`` are positions in the full feature array passed to ``prepare``; routers that
    precompute a sequence (HMM filtering) look up by position, others use ``x``.
    """

    def __init__(self, n_experts: int):
        self.n_experts = n_experts

    def fit(self, x_train: np.ndarray) -> None:
        """Fit on training features only. No-op for routers without a fitted model."""

    def prepare(self, x_all: np.ndarray) -> None:
        """Precompute anything that depends on the full (chronological) sequence."""

    @abstractmethod
    def weights(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor: ...

    def parameters(self) -> list[torch.nn.Parameter]:
        """Trainable parameters (only learned routers have any)."""
        return []

    # --- optional, for routers with an explicit regime model (used by router_analysis) ---

    regime_model: str | None = None  # routers sharing a regime model get analysed once

    def state_probabilities(self) -> np.ndarray | None:
        """Causal regime probabilities for the full sequence passed to prepare, [T, n_states]."""
        return None

    def insample_states(self, x_train: np.ndarray) -> np.ndarray | None:
        """Hindsight regime labels on the training data (may use future data; analysis only)."""
        return None

    def transition_matrix(self) -> np.ndarray | None:
        return None

    def to(self, device: str) -> "Router":
        return self
