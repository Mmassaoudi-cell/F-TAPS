# F-TAPS

F-TAPS (Follower-side Threat-Aware Parameter-Shared actor-critic) is a multi-agent reinforcement learning model for incentive-driven honeypot deployment in smart-grid Advanced Metering Infrastructure (AMI). A retailer with a fixed budget selects among price and time bids from prosumer nodes under a Stackelberg leader-follower formulation, and F-TAPS learns the nodes' bidding policy.

The model has two components:

- **Threat-risk-weighted objective.** Defense-data quality is reweighted by the attack intensity each node observes, so data from heavily attacked nodes is worth more.
- **Parameter-shared, permutation-invariant backbone.** One actor and one critic, each with a mean-and-max pooled context over the other agents, are reused by all agents. The parameter count stays nearly constant as the number of nodes grows, and a trained policy can act for a different number of nodes.

The whole model is in one file, `ftaps.py`.

## Install

Python 3.10 or later with `numpy` and `torch`:

```bash
pip install numpy torch
```

Then copy `ftaps.py` into your project or run from a clone of this repository. A CPU is sufficient.

## Usage

```python
from ftaps import AMIStackelbergEnv, EnvConfig, FTAPS, steady_state_scores

# Market with 4 nodes, 3 honeypot services, and a per-service budget of 100.
env = AMIStackelbergEnv(EnvConfig(K=4, N=3, budget=100.0, seed=1000))

model = FTAPS.for_env(env, seed=1000)
history = model.fit(env, total_steps=15_000, seed=1000)

# Steady-state scores over the final 10% of training steps.
print(steady_state_scores(history))
# {'welfare': ..., 'sps_utility': ..., 'tpr_utility': ..., 'jain_fairness': ...}

# Scores of the trained policy without exploration.
print(model.score(env, steps=1_000))

print(model.n_parameters())  # 94087
model.save("ftaps.pt")
```

To drive the attack-rate process with an empirical attack-intensity series (a one-dimensional array of values in [0, 1]) instead of the synthetic random walk, pass it as `EnvConfig(attack_trace=trace)`. Setting `EnvConfig(rho=0.0)` disables threat weighting.

## Headline result

With 4 nodes, 3 services, and a budget of 100, F-TAPS reaches a steady-state social welfare of 398.2 ± 37.7 (mean ± standard deviation over ten test seeds), the highest of twelve compared methods and 39.9% above the MADDPG baseline (284.6 ± 33.5), with 94,087 trainable parameters against 382,492 for MADDPG. The welfare margin over MADDPG widens as the number of nodes grows from 4 to 15.

## Citation

```bibtex
@article{massaoudi2026ftaps,
  title  = {Threat-Weighted, Parameter-Shared Multi-Agent Reinforcement Learning for Scalable Incentive Mechanisms in Smart-Grid Edge Networks},
  author = {Massaoudi, Mohamed and Ez Eddin, Maymouna},
  year   = {2026}
}
```

## License

MIT License. See `LICENSE`.

## Contact

Mohamed Massaoudi, mohamed.massaoudi@tamu.edu
