#set document(title: "Supervisor Notes")
#set page(paper: "a4", margin: 2.2cm, numbering: "1")
#set text(size: 11pt)
#set heading(numbering: "1.")

= Supervisor Notes (Robin)

== Email reply to our questions (Nov 2026)

=== Router selection
- An HMM-based router is a reasonable choice.
- Compare it with a *learned gating mechanism* that does not rely on explicit regime detection.

=== MoE architecture
- Focus on the *Big* architecture (e), where each expert is a complete network, and vary only the routing.
- This isolates the effect of routing without adding architectural differences.
- Routing variants to compare:
  + *Uniform* weighting (baseline)
  + *Hard* routing on HMM regime predictions
  + *Soft* routing on HMM regime probabilities
  + *Learned* routing with a neural gating network
- If time permits: the *Final* architecture (c), to check whether the results generalise to another MoE structure.

=== Market regime detection
- Regimes are latent and must be inferred from observable market data.
- An HMM is a suitable starting point: it gives both regime classifications (hard) and regime probabilities (soft).
- Related to the master's thesis (modelling of market dynamics).
- *Primary research objective: the comparison of routing mechanisms*, not developing a new regime detection algorithm.

=== Master's thesis code
- Repository: #link("https://github.com/emregulben/portfolio-allocator") (submodule at `external/portfolio-allocator`)
- Simulated market data and portfolio allocation. We should familiarise ourselves with it; integrating the simulation environment into our experiments will be discussed later.

=== Is regime detection necessary to train the router?
- No. An HMM router *explicitly* uses inferred regimes to choose experts. A learned gating network learns these decisions *directly from market features*, without regime labels.
- Key question: does *explicitly* modelling regimes beat *implicit* regime identification through learned routing?

=== Priorities
+ Establish the Big architecture.
+ Define the routing mechanisms.
+ Only then expand the experimental scope.
+ Next deliverable: *exposé*.

== Architectures in Willi et al. (2024), Fig. 1
- (a) *Baseline:* the original network, no MoE
- (b) *Middle:* MoE module replaces the penultimate layer (Ceron et al. 2024b)
- (c) *Final:* MoE module replaces the final layer
- (d) *All:* every layer is replaced by an MoE module
- (e) *Big:* a single MoE module; each expert is the full original network

Their results: Big outperforms the other architectures with hardcoded routing (one expert per task). With Big, SoftMoE and Hardcoded routing work best in multi-task RL, and Hardcoded works best in continual RL.
