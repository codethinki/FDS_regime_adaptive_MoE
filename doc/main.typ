= RegimeAdaptive Mixture of Portfolio Policies


== Experts definition
The experts are defined as a set of portfolio policies that are trained under different market regimes.

Expert variants:
- GRBT
- Random Forest
- Neural Network

Router variants:
- HMM
- Jump Model


== Basline
Compare against baseline RL model with router as input.

== Why soft routing
As in the paper "Mixture of Experts with Soft Routing for Multi-Task Reinforcement Learning" by Xue Bin Peng et al. (2021) says the SoftMoE and Hardcoded work best in MTRL and Hardcoded works best in CRL, though SoftMoe stil outperforms the baseline in CRL. Also in the case of CRL it has been tested that SoftMoE is effective in situations where tasks are frequently changed.


== Hyopthesis
A soft routed MoE Portfolio Policy will outperform a single policy trained on all regimes with a router as input.

Therfore our Research Question is: 
- Can an MoE Portfolio Policy with soft routing outperform a single policy trained on all regimes with a router as input?
- Does explicit market regime detection improve MoE routing performance compared with learned gating?



