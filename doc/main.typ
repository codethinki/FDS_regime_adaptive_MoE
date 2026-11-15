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

== why soft routing
- Multi task reinforcement learning
- effective in situations where tasks are frequently changed
- MT RL $->$ better to do soft routing

== Hyopthesis
A soft routed MoE Portfolio Policy will outperform a single policy trained on all regimes with a router as input.

Therfore our Research Question is: Can an MoE Portfolio Policy with soft routing outperform a single policy trained on all regimes with a router as input?



