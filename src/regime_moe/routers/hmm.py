"""Gaussian HMM router: HMM state k is routed to expert k.

Routing uses *filtered* probabilities P(s_t | x_1..x_t) from the forward algorithm.
hmmlearn's predict_proba returns smoothed probabilities P(s_t | x_1..x_T), which use
future data and must not be used for routing.
"""

import numpy as np
import torch
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from regime_moe.config import HMMConfig
from regime_moe.routers.base import Router


def emission_log_likelihood(hmm: GaussianHMM, x: np.ndarray) -> np.ndarray:
    """log p(x_t | s_t = k), shape [T, n_states]."""
    return np.column_stack(
        [multivariate_normal(mean=hmm.means_[k], cov=hmm.covars_[k]).logpdf(x) for k in range(hmm.n_components)]
    )


def filtered_probabilities(hmm: GaussianHMM, x: np.ndarray) -> np.ndarray:
    """Forward algorithm in log space: P(s_t | x_1..x_t), shape [T, n_states]."""
    log_b = emission_log_likelihood(hmm, x)
    log_a = np.log(hmm.transmat_ + 1e-12)
    log_alpha = np.empty_like(log_b)
    log_alpha[0] = np.log(hmm.startprob_ + 1e-12) + log_b[0]
    log_alpha[0] -= logsumexp(log_alpha[0])
    for t in range(1, len(x)):
        log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_a, axis=0) + log_b[t]
        log_alpha[t] -= logsumexp(log_alpha[t])
    return np.exp(log_alpha)


class HMMRouter(Router):
    regime_model = "hmm"

    def __init__(self, n_experts: int, cfg: HMMConfig, hard: bool):
        super().__init__(n_experts)
        if cfg.n_states != n_experts:
            raise ValueError(f"hmm.n_states ({cfg.n_states}) must equal model.n_experts ({n_experts})")
        self.cfg = cfg
        self.hard = hard
        self.hmm: GaussianHMM | None = None
        self.probs: torch.Tensor | None = None

    def fit(self, x_train: np.ndarray) -> None:
        self.hmm = GaussianHMM(
            n_components=self.cfg.n_states,
            covariance_type=self.cfg.covariance_type,
            n_iter=self.cfg.n_iter,
            random_state=self.cfg.seed,
        ).fit(x_train)

    def prepare(self, x_all: np.ndarray) -> None:
        self.probs = torch.as_tensor(filtered_probabilities(self.hmm, x_all), dtype=torch.float32)

    def weights(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        p = self.probs[idx.cpu()].to(x.device)
        if self.hard:
            p = torch.nn.functional.one_hot(p.argmax(dim=1), self.n_experts).float()
        return p

    def state_probabilities(self) -> np.ndarray:
        return self.probs.numpy()

    def insample_states(self, x_train: np.ndarray) -> np.ndarray:
        return self.hmm.predict(x_train)  # Viterbi path, uses the whole training window

    def transition_matrix(self) -> np.ndarray:
        return self.hmm.transmat_
