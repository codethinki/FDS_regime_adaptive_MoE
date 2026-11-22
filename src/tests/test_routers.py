import numpy as np
import pytest
import torch
from hmmlearn.hmm import GaussianHMM

from regime_moe.config import load_config
from regime_moe.routers import build_router
from regime_moe.routers.hmm import filtered_probabilities


@pytest.fixture
def x():
    rng = np.random.default_rng(0)
    calm, wild = rng.normal(0, 1, (200, 3)), rng.normal(0, 4, (100, 3))
    return np.vstack([calm, wild, calm]).astype(np.float32)


def test_filter_matches_hmmlearn_at_last_step(x):
    # at the last step, filtered and smoothed probabilities coincide (no future data left)
    hmm = GaussianHMM(2, covariance_type="diag", n_iter=50, random_state=0).fit(x)
    filtered = filtered_probabilities(hmm, x)
    assert np.allclose(filtered.sum(axis=1), 1.0)
    assert np.allclose(filtered[-1], hmm.predict_proba(x)[-1], atol=1e-6)


def test_filter_is_causal(x):
    # probabilities up to t must not change when later data changes
    hmm = GaussianHMM(2, covariance_type="diag", n_iter=50, random_state=0).fit(x)
    changed = x.copy()
    changed[250:] *= 10
    assert np.allclose(filtered_probabilities(hmm, x)[:250], filtered_probabilities(hmm, changed)[:250])


@pytest.mark.parametrize("name", ["uniform", "hmm_hard", "hmm_soft", "learned"])
def test_router_weights_are_distributions(x, name):
    cfg = load_config().router
    router = build_router(name, n_experts=cfg.hmm.n_states, n_features=x.shape[1], cfg=cfg)
    router.fit(x[:200])
    router.prepare(x)
    w = router.weights(torch.from_numpy(x[:50]), torch.arange(50))
    assert w.shape == (50, cfg.hmm.n_states)
    assert torch.all(w >= 0) and torch.allclose(w.sum(dim=1), torch.ones(50))
