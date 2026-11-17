"""
Minimal Big-MoE portfolio-allocation prototype
----------------------------------------------
Purpose:
    Compare four routing mechanisms while keeping the expert architecture fixed:
      1. Uniform routing
      2. HMM hard routing
      3. HMM soft routing
      4. Learned neural gating

This is an educational prototype using SYNTHETIC market data.
It is not a trading strategy and its results are not evidence of profitability.

Install:
    pip install numpy pandas matplotlib scikit-learn hmmlearn torch

Run:
    python big_moe_routing_demo.py
"""

import copy
import random
import numpy as np
import pandas as pd
from scipy.special import logsumexp
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM


# --------------------------- Reproducibility ---------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_EXPERTS = 3
N_ASSETS = 4
N_FEATURES = 8
TRAIN_FRACTION = 0.70
EPOCHS = 35
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
RISK_AVERSION = 2.0


# ---------------------- 1. Synthetic market data -----------------------

def make_synthetic_market(n=5000):
    """
    Create a simple market with three hidden regimes.
    The regimes influence the statistical properties of asset returns.

    Features are observable. The regime labels are used ONLY to generate
    synthetic data and are not passed to the model.
    """
    transition = np.array([
        [0.96, 0.03, 0.01],
        [0.03, 0.94, 0.03],
        [0.02, 0.04, 0.94],
    ])

    regimes = np.zeros(n, dtype=int)
    for t in range(1, n):
        regimes[t] = np.random.choice(3, p=transition[regimes[t - 1]])

    # Each regime has different return tendencies and volatility.
    means = np.array([
        [0.0010, 0.0005, 0.0002, 0.0007],   # regime 0
        [-0.0010, -0.0003, 0.0002, -0.0007], # regime 1
        [0.0002, 0.0010, 0.0007, 0.0001],   # regime 2
    ])
    volatilities = np.array([0.008, 0.018, 0.011])

    asset_returns = np.zeros((n, N_ASSETS))
    for t in range(n):
        asset_returns[t] = (
            means[regimes[t]]
            + np.random.normal(0, volatilities[regimes[t]], N_ASSETS)
        )

    # Observable features: recent returns, volatility, momentum and volume.
    features = np.zeros((n, N_FEATURES))
    for t in range(n):
        recent = asset_returns[max(0, t - 5):t + 1]
        features[t, 0:4] = asset_returns[t]
        features[t, 4] = np.std(recent)
        features[t, 5] = np.mean(recent)
        features[t, 6] = np.mean(np.abs(recent))
        features[t, 7] = np.random.lognormal(mean=0.0, sigma=0.25)

    return features.astype(np.float32), asset_returns.astype(np.float32), regimes


# --------------------------- 2. Big architecture -----------------------

class Expert(nn.Module):
    """One complete network: market features -> asset allocation logits."""

    def __init__(self, input_dim, n_assets):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_assets),
        )

    def forward(self, x):
        # Softmax makes the output long-only portfolio weights summing to 1.
        return torch.softmax(self.network(x), dim=-1)


class BigMoE(nn.Module):
    """
    Big MoE: every expert is a complete network.
    Router weights combine the full portfolio outputs of the experts.
    """

    def __init__(self, input_dim, n_assets, n_experts=3):
        super().__init__()
        self.experts = nn.ModuleList([
            Expert(input_dim, n_assets) for _ in range(n_experts)
        ])
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_experts),
        )

    def expert_outputs(self, x):
        # Shape: [batch, experts, assets]
        return torch.stack([expert(x) for expert in self.experts], dim=1)

    def forward(self, x, routing_weights):
        """
        routing_weights shape: [batch, experts]
        Returns:
            portfolio_weights: [batch, assets]
        """
        outputs = self.expert_outputs(x)
        combined = torch.sum(
            outputs * routing_weights.unsqueeze(-1), dim=1
        )
        return combined


# --------------------------- 3. HMM router -----------------------------

def fit_hmm_router(x_train, x_all):
    """
    Fit the HMM only on training observations.
    Then infer regime probabilities for the full chronological sequence.

    HMM states are latent labels. State 0 does NOT inherently mean
    'bull market', state 1 'bear market', etc.
    """
    hmm = GaussianHMM(
        n_components=N_EXPERTS,
        covariance_type="diag",
        n_iter=200,
        random_state=SEED,
    )
    hmm.fit(x_train)

    # Use the forward-filtered probabilities P(state_t | observations_1:t),
    # not predict_proba(), which performs smoothing and can use future data.
    log_emissions = hmm._compute_log_likelihood(x_all)
    log_alpha = np.zeros((len(x_all), N_EXPERTS))
    log_alpha[0] = np.log(hmm.startprob_ + 1e-12) + log_emissions[0]
    log_alpha[0] -= logsumexp(log_alpha[0])

    log_transition = np.log(hmm.transmat_ + 1e-12)
    for t in range(1, len(x_all)):
        for state in range(N_EXPERTS):
            log_alpha[t, state] = (
                logsumexp(log_alpha[t - 1] + log_transition[:, state])
                + log_emissions[t, state]
            )
        log_alpha[t] -= logsumexp(log_alpha[t])

    probabilities = np.exp(log_alpha)
    hard_states = np.argmax(probabilities, axis=1)
    return probabilities.astype(np.float32), hard_states


# ---------------------------- 4. Training ------------------------------

def routing_weights_for_batch(mode, x, hmm_probs, hmm_states, indices, model):
    batch_size = len(indices)

    if mode == "uniform":
        return torch.full(
            (batch_size, N_EXPERTS),
            1.0 / N_EXPERTS,
            device=DEVICE,
        )

    if mode == "hmm_hard":
        states = torch.as_tensor(
            hmm_states[indices], dtype=torch.long, device=DEVICE
        )
        return torch.nn.functional.one_hot(
            states, num_classes=N_EXPERTS
        ).float()

    if mode == "hmm_soft":
        return torch.as_tensor(
            hmm_probs[indices], dtype=torch.float32, device=DEVICE
        )

    if mode == "learned":
        return torch.softmax(model.gate(x), dim=-1)

    raise ValueError(f"Unknown routing mode: {mode}")


def portfolio_loss(weights, next_returns):
    """
    Simple differentiable portfolio objective:
        loss = - mean(portfolio return) + risk_aversion * variance

    This is only a teaching objective. A real study should define and
    justify its investment objective, transaction costs and constraints.
    """
    portfolio_returns = torch.sum(weights * next_returns, dim=1)
    return (
        -portfolio_returns.mean()
        + RISK_AVERSION * portfolio_returns.var(unbiased=False)
    )


def train_one_model(mode, x_train, y_train, hmm_probs, hmm_states):
    model = BigMoE(N_FEATURES, N_ASSETS, N_EXPERTS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    dataset = TensorDataset(
        torch.arange(len(x_train)),
        torch.tensor(x_train),
        torch.tensor(y_train),
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        model.train()
        losses = []

        for indices, x_batch, returns_batch in loader:
            indices_np = indices.numpy()
            x_batch = x_batch.to(DEVICE)
            returns_batch = returns_batch.to(DEVICE)

            route = routing_weights_for_batch(
                mode, x_batch, hmm_probs, hmm_states, indices_np, model
            )
            weights = model(x_batch, route)
            loss = portfolio_loss(weights, returns_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(
                f"{mode:10s} | epoch {epoch + 1:02d}/{EPOCHS} "
                f"| loss {np.mean(losses):.6f}"
            )

    return model


# ----------------------------- 5. Testing ------------------------------

@torch.no_grad()
def evaluate(model, mode, x_test, y_test,
             hmm_probs_test, hmm_states_test):
    model.eval()
    x = torch.tensor(x_test, dtype=torch.float32, device=DEVICE)
    returns = torch.tensor(y_test, dtype=torch.float32, device=DEVICE)

    if mode == "uniform":
        route = torch.full(
            (len(x), N_EXPERTS), 1.0 / N_EXPERTS, device=DEVICE
        )
    elif mode == "hmm_hard":
        states = torch.tensor(
            hmm_states_test, dtype=torch.long, device=DEVICE
        )
        route = torch.nn.functional.one_hot(
            states, num_classes=N_EXPERTS
        ).float()
    elif mode == "hmm_soft":
        route = torch.tensor(
            hmm_probs_test, dtype=torch.float32, device=DEVICE
        )
    elif mode == "learned":
        route = torch.softmax(model.gate(x), dim=-1)
    else:
        raise ValueError(mode)

    weights = model(x, route)
    portfolio_returns = torch.sum(weights * returns, dim=1).cpu().numpy()
    routing = route.cpu().numpy()

    mean_return = float(np.mean(portfolio_returns))
    volatility = float(np.std(portfolio_returns))
    sharpe_like = mean_return / (volatility + 1e-8)

    return {
        "mean_return_per_period": mean_return,
        "volatility_per_period": volatility,
        "return_volatility_ratio": sharpe_like,
        "average_expert_1_weight": float(routing[:, 0].mean()),
        "average_expert_2_weight": float(routing[:, 1].mean()),
        "average_expert_3_weight": float(routing[:, 2].mean()),
    }


# -------------------------------- Main ----------------------------------

def main():
    print(f"Running on: {DEVICE}")
    print("Generating synthetic market data...")

    X, returns, hidden_regimes = make_synthetic_market()

    split = int(len(X) * TRAIN_FRACTION)

    # Scale using training data only.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[:split]).astype(np.float32)
    X_all = scaler.transform(X).astype(np.float32)
    X_test = X_all[split:]

    # The target is the next-period asset return.
    # In a real dataset, align features at time t with returns at t+1.
    Y = np.roll(returns, -1, axis=0)
    Y_train = Y[:split]
    Y_test = Y[split:]

    print("Fitting HMM using training data only...")
    hmm_probs, hmm_states = fit_hmm_router(X_train, X_all)
    hmm_probs_test = hmm_probs[split:]
    hmm_states_test = hmm_states[split:]

    modes = ["uniform", "hmm_hard", "hmm_soft", "learned"]
    results = []
    trained_models = {}

    for mode in modes:
        print(f"\nTraining model with {mode} routing")
        model = train_one_model(
            mode,
            X_train,
            Y_train,
            hmm_probs[:split],
            hmm_states[:split],
        )

        trained_models[mode] = model
        metrics = evaluate(
            model, mode, X_test, Y_test,
            hmm_probs_test, hmm_states_test
        )
        metrics["routing_method"] = mode
        results.append(metrics)

    results_df = pd.DataFrame(results).set_index("routing_method")
    print("\nTEST RESULTS (synthetic data; illustrative only)")
    print(results_df.round(6).to_string())

    results_df.to_csv("big_moe_demo_results.csv")
    print("\nSaved results to big_moe_demo_results.csv")

    # Inspect how the learned gate distributes inputs across experts.
    learned_model = trained_models["learned"]
    with torch.no_grad():
        x = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
        learned_routes = torch.softmax(
            learned_model.gate(x), dim=-1
        ).cpu().numpy()

    print("\nAverage learned routing probabilities:")
    for i, value in enumerate(learned_routes.mean(axis=0), start=1):
        print(f"Expert {i}: {value:.3f}")


if __name__ == "__main__":
    main()
