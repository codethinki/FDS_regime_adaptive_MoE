= Summary: Regime-Switching Factor Investing with HMMs

== Markov chain (formal definition)

+ *State space:* a finite set of possible states
  $ S = {1, 2, dots, N} $
  e.g. $S = {"bull", "sideways", "bear"}$, $N = 3$.

+ *Sequence of random variables:* one per time step, each taking values in $S$:
  $ X_0, X_1, X_2, dots $
  $X_t$ is the state at time $t$.

+ *Markov property:* the next state depends only on the current state, not on the history before it:
  $ P(X_(t+1) = j | X_t = i, X_(t-1), dots, X_0) = P(X_(t+1) = j | X_t = i) $

+ *Transition matrix $A$:*
  $ a_(i j) = P(X_(t+1) = j | X_t = i) $
  - Row $i$ lists the probabilities of moving from state $i$ to each state.
  - Each row sums to 1: $sum_j a_(i j) = 1$.
  - *Homogeneous:* $A$ does not depend on $t$.

+ *Initial distribution $pi$:*
  $ pi_i = P(X_0 = i) $

A Markov chain is fully described by $(S, A, pi)$.

== Hidden Markov model: what it learns

HMM = hidden Markov chain + emissions.

- *Hidden chain:* $X_t in S$ is a Markov chain with $(A, pi)$, but $X_t$ is never observed.
- *Emissions:* each state $i$ generates observations from its own distribution
  $ Y_t | (X_t = i) tilde cal(N)(mu_i, Sigma_i) $
  For us: $Y_t = ("return"_t, "vol"_t)$.
- *Conditional independence:* $P(Y_t | X_(0:T), Y_(-t)) = P(Y_t | X_t)$
- *Parameters to learn:*
  $ lambda = (A, mu, Sigma, pi) $
  $S$ (i.e. $N$) is chosen by us; the state names (bull, bear, ...) are assigned afterwards from $hat(mu)_i$.

#table(
  columns: 3,
  [*Object*], [*Unknown?*], [*What we do*],
  [$S$], [no (we pick $N$)], [--],
  [$A, pi$], [yes], [estimate $arrow hat(A), hat(pi)$],
  [$mu, Sigma$], [yes], [estimate $arrow hat(mu), hat(Sigma)$],
  [$X_0, dots, X_T$], [yes], [infer $arrow P(X_t = i | Y)$],
)

== Objective: likelihood $P(Y | lambda)$

Joint probability of one path $X$ and the data $Y$:
$ P(X, Y | lambda) = pi_(X_0) product_(t=1)^T a_(X_(t-1) X_t) product_(t=0)^T P(Y_t | X_t) $

Likelihood: sum over all $N^(T+1)$ hidden paths, computed in $O(T N^2)$ by the forward algorithm:
$ P(Y | lambda) = sum_X P(X, Y | lambda) $

Training objective:
$ hat(lambda) = op("argmax")_lambda log P(Y_(0:T) | lambda) $

== Training: Baum-Welch (EM)

+ Choose $N$; initialise $lambda$ randomly.
+ *E-step:* with the current $lambda$ (forward-backward algorithm) compute
  $ gamma_t (i) = P(X_t = i | Y_(0:T), lambda) $
  $ xi_t (i, j) = P(X_t = i, X_(t+1) = j | Y_(0:T), lambda) $
+ *M-step:* re-estimate $lambda$:
  $ mu_i = (sum_t gamma_t (i) Y_t) / (sum_t gamma_t (i)) quad
    Sigma_i = (sum_t gamma_t (i) (Y_t - mu_i)(Y_t - mu_i)^top) / (sum_t gamma_t (i)) $
  $ a_(i j) = (sum_t xi_t (i, j)) / (sum_t gamma_t (i)) quad
    pi_i = gamma_0 (i) $
+ Repeat 2--3 until $log P(Y | lambda)$ stops increasing (paper: max. 75 iterations).
+ EM finds only a local optimum: run several random inits, keep the best $log P(Y | lambda)$.

== Their issue: the Gaussian emission assumption

- Training assumes $Y_t | (X_t = i) tilde cal(N)(mu_i, Sigma_i)$, but vol is positive and right-skewed, so a Gaussian fits badly.
- Their patch (§3.5.2): after training, per regime and per feature fit a distribution (normal, lognormal, Pareto, gamma, beta, exponential), chosen by the K-S statistic
  $ D = max_x |F_"emp" (x) - F_"fit" (x)| $
  then switch if $"pdf"_"vol" > 0.3$ and $"pdf"_"ret" > 0.5$.
- This does *not* fix the problem:
  - The labels $X_t$ were already produced by the Gaussian HMM; the K-S curves are fitted on these labels and cannot change them.
  - States are defined by Gaussians but detected with other distributions, which is inconsistent.
  - The rule drops $A$ and the return/vol correlation, and a PDF value is not a probability (the thresholds depend on units).
- The distribution assumption has to be fixed *during training*, not afterwards.

== Our plan

+ *Lognormal vol:* model $"vol"_t tilde "LogNormal"$ per state, i.e. train the Gaussian HMM on
  $ Y_t = (r_t, log "vol"_t) $
  ($log "vol"_t tilde cal(N) arrow.l.r.double "vol"_t tilde "LogNormal"$)
+ *Check the fit* per state (e.g. QQ-plots of $Y_t$ for days with $hat(X)_t = i$).
+ *If it still does not fit:* GMM-HMM, each state's emission is a mixture of $K$ Gaussians
  $ P(Y_t | X_t = i) = sum_(k=1)^K c_(i k) cal(N)(Y_t; mu_(i k), Sigma_(i k)), quad sum_k c_(i k) = 1 $
  - Small $K$ (2--3): the bear regime has only $approx 350$ training days.
  - `hmmlearn.hmm.GMMHMM(n_components=3, n_mix=K)`
