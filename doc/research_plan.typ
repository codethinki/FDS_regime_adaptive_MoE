#set document(title: "Research Plan")
#set page(paper: "a4", margin: 2.2cm, numbering: "1")
#set text(size: 11pt)
#set heading(numbering: "1.")

= Research Plan: Regime-Adaptive Mixture of Experts for Portfolio Allocation

_Draft for the next meeting._

== Research question

*Main question.* Does routing RL portfolio experts with a causally inferred market regime improve out-of-sample risk-adjusted performance, after transaction costs, compared with
(a) a single RL agent that sees the regime signal in its state, and
(b) simple rule-based de-risking?

*Sub-questions*
+ *Router.* How do the regime model (jump model vs. HMM) and the gating (hard vs. soft) affect detection latency, switching frequency and portfolio performance?
+ *Training data.* Do experts trained in the template simulator generalise better to real data than experts trained on real data only?
+ *Specialisation.* Do the experts learn economically different allocations per regime?

*Design principle.* Timing (which regime is it now?) is learned by the router from real data. The experts only learn how to allocate _given_ a regime. So a simulator is used for robustness at most, never for timing.

== Setup

- *Assets:* SPY, TLT, GLD, DBC + cash (BIL); daily data from 2013
- *Split:* train 2013-01 to 2024-08, test 2024-09 to today (to be fixed before any training run)
- *Costs:* 5 bps per unit of turnover; signals lagged by one day

== Method

=== Router
Statistical jump model (Shu, Yu & Mulvey 2024) on SPY excess returns $R_t$:
$ min_(Theta, s) sum_t ||x_t - theta_(s_t)||^2 + lambda sum_t bb(1){s_t != s_(t-1)} $
- Features $x_t$: $"DD"_10$, $"Sortino"_20$, $"Sortino"_60$ (EWM, halflife in days)
- Online inference: $hat(s)_t$ (or $p_t$ for the continuous JM) uses data up to $t$ only
- Comparison: Gaussian HMM with filtered posterior $P(X_t | Y_(0:t))$

=== Experts and mixture
- $K$ experts $pi_k (s_t)$ output portfolio weights $w in Delta^4$ (4 assets + cash)
- State $s_t$: return features plus previous weights $w_(t-1)$ (needed for turnover control)
- Mixture with a fixed (not learned) gate:
  $ w_t = sum_(k=1)^K p_t (k) dot pi_k (s_t) $
- Reward: differential Sharpe ratio (Moody & Saffell 2001) net of costs $c dot ||w_t - w_(t-1)||_1$
- Algorithm: PPO with small MLPs (to be confirmed)

=== Training data (sub-question 2)
+ Real data: random contiguous windows from the training period, with feature warm-up before each window
+ Template simulator (`external/portfolio-allocator`)

== Baselines

#table(
  columns: 2,
  [*Baseline*], [*Tests*],
  [SPY, 60/40, equal weight, inverse vol], [passive allocation],
  [200-day MA, vol targeting 10%], [rule-based de-risking],
  [Markowitz (template)], [the template's expert],
  [JM + fixed allocation per regime], [does RL add anything beyond the router?],
  [Single PPO agent with $p_t$ in the state (Raj 2025)], [does the MoE add anything beyond the regime signal?],
)

Reference values, test period 2024-09 to 2026-11: equal weight Sharpe 0.94, SPY 0.71, vol targeting 0.47.

== Evaluation
- Sharpe (vs. T-bills) with standard errors; paired test of Sharpe differences (Ledoit & Wolf 2008)
- Max drawdown, turnover, Sortino
- Router: switches per year, detection delay, agreement with in-sample regimes
- 5+ seeds per RL configuration, reported as mean $plus.minus$ std

== First prototype result (`notebooks/01_jump_model_router.ipynb`)
- JM with 2 states, $lambda = 50$: in-sample, Bull has SPY Sharpe 1.76 and Bear $-0.25$ (train).
- *Online* the separation shrinks: Bear has SPY Sharpe 0.71 (train) because detection lags by a median of 24 days, so the router often switches to Bear after the drop and stays there into the rebound.
- Implication: detection latency is the central problem of the router, which supports sub-question 1.
- GLD is the only asset with a positive Sharpe in Bear (0.84 in-sample), so the experts do have something regime-specific to learn.

== Paper structure (draft)
+ *Introduction:* non-stationary markets, regime-adaptive MoE, research question
+ *Related work:* regime models (Hamilton 1989; jump models, Shu et al. 2024); RL in finance (Moody & Saffell 2001; Raj 2025); MoE in RL (Willi et al. 2024)
+ *Data:* assets, period, features
+ *Method:* router, experts, mixture, reward, training data
+ *Experimental setup:* split, baselines, metrics, statistical tests
+ *Results:* router analysis, portfolio performance, ablations (router type, gating, training data)
+ *Discussion and limitations:* short test period, latency, simulator assumptions
+ *Conclusion*

== Open questions for the meeting
- Is the test period 2024-09 to today fixed, or may we choose it (e.g. 2022 to today, which includes a bear market)?
- Is the template simulator mandatory, or optional as a comparison?
- Is the RL algorithm free to choose (PPO), or is the template's IL + AWR expected?
- Which deliverable: paper format and length, code, presentation?
