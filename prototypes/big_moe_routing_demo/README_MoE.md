# Big MoE Portfolio Allocation Prototype

## 1. Project purpose

This project is an educational **Mixture of Experts (MoE)** prototype
for portfolio allocation.

It compares four ways of combining three neural-network experts:

1.  **Uniform routing** -- every expert receives the same weight.
2.  **HMM Hard routing** -- the expert corresponding to the most likely
    Hidden Markov Model (HMM) state is selected.
3.  **HMM Soft routing** -- expert weights are the HMM state
    probabilities.
4.  **Learned Gate** -- a neural network learns the expert weights
    directly from market features and the portfolio objective.

The main research question is:

> Does a learned routing mechanism allocate a portfolio more effectively
> than routing based on HMM-inferred market states?

The prototype uses **synthetic market data**, not real historical
prices. Its results are useful for understanding and debugging the
architecture; they are not evidence of investment profitability.

------------------------------------------------------------------------

## 2. Big picture: the pipeline

The script follows this sequence:

``` text
Synthetic market data
        |
        +--> Market features (8 per observation)
        |
        +--> Next-period returns (4 assets)
        |
        v
Fit HMM on training features
        |
        +--> Filtered state probabilities
        +--> Most likely HMM state
        |
        v
Train four separate Big MoE models
        |
        +--> Uniform router
        +--> HMM Hard router
        +--> HMM Soft router
        +--> Learned neural gate
        |
        v
Evaluate on chronological test period
        |
        +--> Mean return
        +--> Volatility
        +--> Return / volatility ratio
        +--> Average expert routing weights
        |
        v
big_moe_demo_results.csv
```

The **Big MoE** design means that each expert is a complete neural
network. Each expert independently maps market features to a full
portfolio allocation. The router combines those allocations.

------------------------------------------------------------------------

## 3. Installation and execution

### Requirements

-   Python 3.10 or newer is recommended.
-   `pip`
-   The libraries below.

### Create and activate a virtual environment

On macOS or Linux:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install dependencies

``` bash
python -m pip install --upgrade pip
python -m pip install numpy pandas scipy scikit-learn hmmlearn torch
```

### Run the script

``` bash
python big_moe_routing_demo.py
```

The script prints training progress and test metrics. It also writes:

``` text
big_moe_demo_results.csv
```

The CSV is saved in the current working directory (normally the folder
from which the command is run).

------------------------------------------------------------------------

## 4. Configuration

At the beginning of the script, the experiment is controlled by
constants:

``` python
SEED = 42
N_EXPERTS = 3
N_ASSETS = 4
N_FEATURES = 8
TRAIN_FRACTION = 0.70
EPOCHS = 35
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
RISK_AVERSION = 2.0
```

  -----------------------------------------------------------------------
  Parameter                           Meaning
  ----------------------------------- -----------------------------------
  `SEED`                              Sets random seeds to make runs more
                                      reproducible.

  `N_EXPERTS`                         Number of expert networks and HMM
                                      states: 3.

  `N_ASSETS`                          Number of assets in the portfolio:
                                      4.

  `N_FEATURES`                        Number of observable market
                                      features: 8.

  `TRAIN_FRACTION`                    First 70% of observations are used
                                      for training; final 30% for
                                      testing.

  `EPOCHS`                            Number of training passes through
                                      the training dataset.

  `BATCH_SIZE`                        Number of observations processed in
                                      one mini-batch.

  `LEARNING_RATE`                     Step-size parameter used by Adam.

  `RISK_AVERSION`                     Multiplier on the portfolio-return
                                      variance penalty in the loss.
  -----------------------------------------------------------------------

An **epoch** is one complete pass through the training dataset. A
**batch** is a smaller group of observations processed before an
optimizer update.

------------------------------------------------------------------------

## 5. Synthetic market data

Function:

``` python
make_synthetic_market(n=5000)
```

This function creates 5,000 observations and simulates three hidden
market regimes. A regime is a recurring statistical condition of the
market, such as a period with different average returns or volatility.
The simulated regimes are not explicitly named bull, bear, or sideways.

### 5.1 Regime transitions

The transition matrix is:

``` python
transition = np.array([
    [0.96, 0.03, 0.01],
    [0.03, 0.94, 0.03],
    [0.02, 0.04, 0.94],
])
```

Each row describes the probabilities of moving from the current regime
to each possible next regime. For example, if the current regime is 0,
the probability of staying in regime 0 is 0.96.

The code samples a hidden regime at each time step using this matrix.

### 5.2 Asset returns

The arrays `means` and `volatilities` define different return behavior
for each simulated regime.

For each time step, each asset return is generated as:

``` text
return = regime-specific average return + random noise
```

This makes the four assets behave differently depending on the hidden
regime.

### 5.3 Observable features

The model receives eight features:

  Feature positions   Feature
  ------------------- --------------------------------------------
  1--4                Current returns of the four assets
  5                   Standard deviation of recent asset returns
  6                   Mean of recent asset returns
  7                   Mean absolute recent asset return
  8                   Simulated trading volume

The function returns:

-   `features`: observable inputs to the models.
-   `asset_returns`: returns for the four assets.
-   `regimes`: hidden labels used to generate the synthetic data.

**Important:** The hidden regime labels are not passed into the model or
used as supervised training targets. They are only used to generate the
artificial market. The HMM must infer latent states from the observable
features.

------------------------------------------------------------------------

## 6. The Big MoE architecture

The code defines two classes: `Expert` and `BigMoE`.

### 6.1 One complete expert

``` python
class Expert(nn.Module):
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
        return torch.softmax(self.network(x), dim=-1)
```

Each expert receives 8 features and has:

-   An input-to-hidden layer: 8 → 64.
-   A ReLU activation.
-   A hidden layer: 64 → 32.
-   Another ReLU activation.
-   An output layer: 32 → 4.

The final `softmax` turns the four output values into portfolio weights
that are non-negative and sum to 1. This means the example is
**long-only** and fully invested; it does not model short selling or
cash allocation.

For example, an expert might output:

``` text
Asset 1: 50%
Asset 2: 25%
Asset 3: 15%
Asset 4: 10%
Total: 100%
```

Each expert has its own trainable network parameters.

### 6.2 Combining experts

`BigMoE` creates three complete experts:

``` python
self.experts = nn.ModuleList([
    Expert(input_dim, n_assets)
    for _ in range(n_experts)
])
```

It also defines a gate network:

``` python
self.gate = nn.Sequential(
    nn.Linear(input_dim, 32),
    nn.ReLU(),
    nn.Linear(32, n_experts),
)
```

The gate maps the 8 market features to 3 output values, one for each
expert. In the learned-routing experiment, softmax turns these values
into expert weights.

The experts first produce their own portfolio allocations. The router
then calculates a weighted combination:

\[ w\_{`\text{final}`{=tex}}(t)=`\sum`{=tex}\_{i=1}\^{3}g_i(t)w_i(t) \]

where:

-   (w_i(t)) is the portfolio allocation produced by expert (i).
-   (g_i(t)) is the routing weight assigned to expert (i).
-   The three routing weights sum to 1.

For example, if the router outputs `[0.7, 0.2, 0.1]`, the final
portfolio is 70% of Expert 1's allocation, 20% of Expert 2's allocation,
and 10% of Expert 3's allocation.

------------------------------------------------------------------------

## 7. HMM: estimating latent market states

Function:

``` python
fit_hmm_router(x_train, x_all)
```

The Hidden Markov Model is fitted only on the training features:

``` python
hmm.fit(x_train)
```

It has three hidden states because `N_EXPERTS = 3`.

The HMM learns statistical patterns and transition probabilities from
the observable features. It does **not** know that a state is a bull
market, bear market, or sideways market. State numbering is arbitrary.

### 7.1 Filtered probabilities

At each time (t), the HMM estimates:

\[ P(S_t `\mid `{=tex}X_1,`\ldots`{=tex},X_t) \]

This is the probability of the current hidden state given observations
up to the current time.

The code uses a forward-filtering calculation. It does not use
`predict_proba()`, because that method performs smoothing and can use
future observations when estimating earlier states.

The function returns:

-   `probabilities`: three state probabilities for each time step.
-   `hard_states`: the index of the state with the highest probability.

For example:

``` text
Probabilities: [0.80, 0.15, 0.05]
Most likely state: 0
```

This means state 0 has the highest estimated probability at that time.

**Implementation note:** The code calls `hmm._compute_log_likelihood`,
an internal `hmmlearn` method. Internal methods can change between
library versions, so this line may need updating if a future `hmmlearn`
version changes its API.

------------------------------------------------------------------------

## 8. The four routing experiments

All four experiments use the same Big MoE structure, but use different
rules to calculate the routing weights.

### Experiment 1 --- Uniform routing

``` python
return torch.full(
    (batch_size, N_EXPERTS),
    1.0 / N_EXPERTS,
    device=DEVICE,
)
```

Every expert receives the same weight:

``` text
Expert 1: 33.33%
Expert 2: 33.33%
Expert 3: 33.33%
```

This is the baseline. It does not use HMM probabilities or the learned
gate.

### Experiment 2 --- HMM Hard routing

The code takes the most likely HMM state and converts it to a one-hot
vector.

For example:

``` text
HMM probabilities: [0.80, 0.15, 0.05]
Hard routing:      [1.00, 0.00, 0.00]
```

Only the expert with the matching state index contributes to the final
portfolio.

The association between HMM state 0 and Expert 1, state 1 and Expert 2,
etc. is an implementation convention. HMM state labels have no inherent
economic meaning.

### Experiment 3 --- HMM Soft routing

The HMM probabilities are used directly:

``` text
HMM probabilities: [0.80, 0.15, 0.05]
Soft routing:      [0.80, 0.15, 0.05]
```

All experts can contribute, with larger weights given to experts
corresponding to more probable states.

### Experiment 4 --- Learned Gate

``` python
if mode == "learned":
    return torch.softmax(model.gate(x), dim=-1)
```

This is the learned routing method.

1.  The market features `x` enter the gate neural network.
2.  The gate produces three logits.
3.  Softmax converts the logits into three positive weights summing to
    1.
4.  Those weights combine the three expert portfolio allocations.

Unlike the HMM methods, the learned gate does not explicitly estimate
market states. It learns a mapping from market features to expert
weights by optimizing the portfolio loss.

The gate is not told that a particular feature pattern means "bull
market" or "high volatility regime." It may learn patterns related to
market conditions, but it is not required to assign named regimes.

------------------------------------------------------------------------

## 9. Portfolio objective and loss

Function:

``` python
portfolio_loss(weights, next_returns)
```

First, the code calculates the portfolio return:

\[ R_p(t)=`\sum`{=tex}\_{i=1}\^{4}w_i(t)r_i(t+1) \]

where:

-   (w_i(t)) is the weight allocated to asset (i).
-   (r_i(t+1)) is the next-period return of asset (i).

The loss is:

\[ Loss=-E\[R_p\]+`\lambda `{=tex}Var(R_p) \]

In the code, (`\lambda`{=tex}) is `RISK_AVERSION`.

-   The negative mean-return term encourages higher average returns.
-   The variance term penalizes fluctuations in portfolio returns.

PyTorch minimizes the loss. Minimizing negative mean return encourages
the model to maximize mean return, while the variance term encourages
lower variability.

This is a simplified teaching objective, not a complete financial
utility function. It does not include transaction costs, turnover,
slippage, taxes, drawdown constraints, leverage limits, or a risk-free
rate.

------------------------------------------------------------------------

## 10. How training works

Function:

``` python
train_one_model(mode, x_train, y_train, hmm_probs, hmm_states)
```

A new Big MoE is created for each routing experiment. Therefore, the
four methods train four separate models from their own initial
parameters.

The optimizer is:

``` python
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
```

`model.parameters()` includes the parameters of all three experts and
the gate network.

For each batch, the script:

1.  Gets the market features and next-period returns.
2.  Calculates routing weights for the selected method.
3.  Runs all experts to produce their portfolio allocations.
4.  Combines expert allocations using the routing weights.
5.  Calculates portfolio loss.
6.  Calls `optimizer.zero_grad()` to clear old gradients.
7.  Calls `loss.backward()` to calculate gradients.
8.  Calls `optimizer.step()` to update trainable parameters.

The process repeats over all batches and all 35 epochs.

### What is trained in each experiment?

-   **Uniform:** Experts are trained through the equally weighted
    combined portfolio. The gate is not used.
-   **HMM Hard:** Experts are trained through the selected expert for
    each observation. The gate is not used.
-   **HMM Soft:** Experts are trained through the HMM-weighted
    combination. The gate is not used.
-   **Learned:** The experts and the gate are trained jointly. Gradients
    from the portfolio loss can update both the experts and the gate.

Because the optimizer is created from `model.parameters()`, the gate is
included in the optimizer even in the first three experiments, but it
receives no useful gradient when it is not used to calculate routing.

------------------------------------------------------------------------

## 11. Testing and evaluation

Function:

``` python
evaluate(...)
```

The model is evaluated on the final 30% of the chronological dataset,
which was not used to train the experts.

The script calculates:

  -----------------------------------------------------------------------
  Metric                              Meaning
  ----------------------------------- -----------------------------------
  `mean_return_per_period`            Average portfolio return per test
                                      observation

  `volatility_per_period`             Standard deviation of portfolio
                                      returns

  `return_volatility_ratio`           Mean return divided by volatility

  `average_expert_1_weight`           Average routing weight assigned to
                                      Expert 1

  `average_expert_2_weight`           Average routing weight assigned to
                                      Expert 2

  `average_expert_3_weight`           Average routing weight assigned to
                                      Expert 3
  -----------------------------------------------------------------------

The return/volatility ratio is a simple Sharpe-like measure. It is
**not** an annualized Sharpe ratio and does not subtract a risk-free
rate.

The average expert weights help reveal how much each expert contributes
on average. For the hard router, they correspond to the fraction of test
observations assigned to each expert. For soft and learned routing, they
are average probability weights.

------------------------------------------------------------------------

## 12. Main function: execution order

The `main()` function connects all components:

1.  Generate 5,000 synthetic observations.
2.  Split them chronologically into training and testing periods.
3.  Standardize features using training-period statistics only.
4.  Align features with next-period asset returns.
5.  Fit the HMM on training features and calculate filtered state
    probabilities.
6.  Train one Big MoE for each of the four routing methods.
7.  Evaluate each model on the test period.
8.  Print the comparison table.
9.  Save `big_moe_demo_results.csv`.
10. Print the average learned routing probabilities on the test period.

The script starts the process with:

``` python
if __name__ == "__main__":
    main()
```

------------------------------------------------------------------------

## 13. Understanding the terminal output

During training, you may see:

``` text
Training model with uniform routing
uniform    | epoch 10/35 | loss -0.000527
uniform    | epoch 20/35 | loss -0.000975
uniform    | epoch 30/35 | loss -0.001340
```

This means the uniform-routing model is being trained. The epoch count
is training progress. The loss is the objective value on training
batches.

A more negative loss can indicate improvement in this particular
objective, but it does **not** prove better test performance. Compare
the final test metrics to evaluate generalization.

At the end, the script prints a test-results table and writes the CSV
file.

------------------------------------------------------------------------

## 14. Important limitations and research checks

This prototype is deliberately simple. Before using it for a research
conclusion, address the following points.

### 14.1 Correct next-period target alignment

The current script contains:

``` python
Y = np.roll(returns, -1, axis=0)
```

`np.roll` wraps the final row around to the first row. This means the
final observation receives the first observation's return as its target.
That is not a valid next-period target.

A safer alignment is:

``` python
X = X[:-1]
Y = returns[1:]
```

Then split the aligned `X` and `Y` chronologically. This also ensures
that the last feature row has a real next-period return.

### 14.2 Synthetic data only

The market data is generated by the script. It does not represent actual
historical market conditions.

### 14.3 Expert specialization is not guaranteed

The three experts are separate networks, but this does not guarantee
that they learn different strategies. Inspect their allocations and
behavior, not only the router weights.

### 14.4 HMM state-to-expert mapping

HMM state labels are arbitrary. The code maps state index 0 to Expert 1,
state index 1 to Expert 2, and state index 2 to Expert 3. This mapping
is a design choice, not a discovered semantic relationship.

### 14.5 Repeated runs

One random seed is not enough for a strong comparison. Repeat
experiments with multiple seeds and report mean and dispersion of the
test metrics.

### 14.6 Financial realism

A more realistic study should consider transaction costs, turnover,
slippage, portfolio constraints, appropriate benchmarks, drawdowns, and
a clearly defined evaluation period.

### 14.7 Validation protocol

The test set should remain untouched during model selection. For a more
rigorous study, use a chronological validation period for hyperparameter
selection and a separate final test period.

------------------------------------------------------------------------

## 15. What to investigate in the research

The most important comparison is not simply which model has the highest
return in one run.

Investigate:

1.  Does Learned Gate improve out-of-sample performance compared with
    HMM Soft?
2.  Does HMM Hard suffer when the HMM is uncertain about the current
    state?
3.  Does the learned gate use all experts, or does it collapse onto one
    expert?
4.  Do routing weights change over time and across different market
    conditions?
5.  Are results stable across multiple random seeds and market periods?
6.  Do improvements remain after transaction costs and realistic
    constraints are included?

The core research objective is to isolate the effect of the **routing
mechanism** while keeping the Big MoE expert architecture consistent.

------------------------------------------------------------------------

## 16. Glossary

  -----------------------------------------------------------------------
  Term                                Simple explanation
  ----------------------------------- -----------------------------------
  Asset                               An instrument in which the
                                      portfolio can invest.

  Portfolio allocation                The percentage of capital assigned
                                      to each asset.

  Expert                              A complete neural network that
                                      outputs a portfolio allocation.

  Router / Gate                       Mechanism that assigns weights to
                                      experts.

  Mixture of Experts                  Architecture that combines outputs
                                      from multiple expert models.

  Big MoE                             MoE in which each expert is a
                                      complete network.

  Market regime                       A recurring statistical condition
                                      or behavior of the market.

  HMM                                 Probabilistic model used to infer
                                      hidden states from observations.

  Hard routing                        Select one expert.

  Soft routing                        Combine experts using probability
                                      weights.

  Learned Gate                        Neural network that learns expert
                                      weights from data and the training
                                      objective.

  Epoch                               One full pass through the training
                                      dataset.

  Batch                               A subset of observations processed
                                      in one training step.

  Loss                                Numerical objective that training
                                      attempts to minimize.

  Backpropagation                     Calculation of gradients used to
                                      update model parameters.

  Overfitting                         When a model learns training data
                                      patterns that do not generalize
                                      well.

  Look-ahead bias                     Using information from the future
                                      when making a past decision.
  -----------------------------------------------------------------------

**Final reminder:** This repository is a learning and
architecture-comparison prototype. It should be treated as a starting
point for the research, not as a deployable investment strategy.
