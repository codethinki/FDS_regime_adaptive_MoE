from regime_moe.config import RouterConfig
from regime_moe.routers.base import Router
from regime_moe.routers.hmm import HMMRouter
from regime_moe.routers.learned import LearnedRouter
from regime_moe.routers.uniform import UniformRouter


def build_router(name: str, n_experts: int, n_features: int, cfg: RouterConfig) -> Router:
    builders = {
        "uniform": lambda: UniformRouter(n_experts),
        "hmm_hard": lambda: HMMRouter(n_experts, cfg.hmm, hard=True),
        "hmm_soft": lambda: HMMRouter(n_experts, cfg.hmm, hard=False),
        "learned": lambda: LearnedRouter(n_experts, n_features, cfg.learned),
    }
    if name not in builders:
        raise ValueError(f"unknown router '{name}', choose from {sorted(builders)}")
    return builders[name]()
