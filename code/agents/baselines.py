"""Random, Fixed, Heuristic, Greedy baseline policies (Section VI-A)."""
from __future__ import annotations
import numpy as np


class RandomPolicy:
    """Selects prices and times uniformly at random within allowed ranges."""
    def __init__(self, N, p_range, tau_range, rng=None):
        self.N, self.p_range, self.tau_range = N, p_range, tau_range
        self.rng = rng or np.random.default_rng()

    def act(self, obs):
        p = self.rng.uniform(*self.p_range, size=self.N)
        t = self.rng.uniform(*self.tau_range, size=self.N)
        return np.concatenate([p, t]).astype(np.float32)


class FixedPolicy:
    """Constant bid at the midpoint of allowed ranges."""
    def __init__(self, N, p_range, tau_range):
        self.N = N
        self.p_mid = (p_range[0] + p_range[1]) / 2
        self.t_mid = (tau_range[0] + tau_range[1]) / 2

    def act(self, obs):
        return np.concatenate([
            np.full(self.N, self.p_mid), np.full(self.N, self.t_mid)]).astype(np.float32)


class HeuristicPolicy:
    """Bids proportional to own observed data quality and resources.

    obs = [quality_1..quality_N (normalized), attack_rate (normalized), resources (normalized)]
    Higher quality -> higher price (more confident in value provided);
    higher resources -> more time offered.
    """
    def __init__(self, N, p_range, tau_range):
        self.N, self.p_range, self.tau_range = N, p_range, tau_range

    def act(self, obs):
        quality = obs[: self.N]
        resources = obs[-1]
        p = self.p_range[0] + quality * (self.p_range[1] - self.p_range[0])
        t = np.full(self.N, self.tau_range[0] + resources * (self.tau_range[1] - self.tau_range[0]))
        return np.concatenate([p, t]).astype(np.float32)


class GreedyPolicy:
    """Samples M candidate actions and picks the one maximizing the immediate
    reward under a one-step lookahead, holding other agents' most recent
    actions fixed (myopic best response)."""
    def __init__(self, agent_idx, env, N, p_range, tau_range, n_candidates=20, rng=None):
        self.k, self.env = agent_idx, env
        self.N, self.p_range, self.tau_range = N, p_range, tau_range
        self.n_candidates = n_candidates
        self.rng = rng or np.random.default_rng()

    def act(self, obs, other_actions):
        """other_actions: list of K arrays, current placeholder actions for
        all agents (agent k's own entry will be overwritten by candidates)."""
        best_a, best_r = None, -np.inf
        for _ in range(self.n_candidates):
            p = self.rng.uniform(*self.p_range, size=self.N)
            t = self.rng.uniform(*self.tau_range, size=self.N)
            cand = np.concatenate([p, t]).astype(np.float32)
            trial = list(other_actions)
            trial[self.k] = cand
            rewards, _ = self.env.compute_rewards(trial)
            if rewards[self.k] > best_r:
                best_r, best_a = rewards[self.k], cand
        return best_a
