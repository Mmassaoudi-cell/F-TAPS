"""
AMI Stackelberg honeypot-incentive environment.

Reproduces the system model of:
  Albaseer et al., "Incentive-Driven Honeypot Defense: A Multi-Agent DRL
  Framework for Securing Smart Grid Networks," IEEE TNSM, vol. 23, 2026.

Implementation assumptions (see SOURCE_PAPER_AUDIT.md, "Partially specified"):
  - Action per SPS is a 2N-dim vector (price, time) PER SERVICE, matching the
    Table II actor output width "N x 2", generalizing Eq. (1)'s single p_k.
  - theta_n(x) = sqrt(x) (canonical concave utility).
  - Reward weights alpha=beta=gamma=delta=1 applied to min-max normalized
    (quality, price, time) components.
  - TPR scoring weights eta=xi=zeta=1 applied to normalized (quality, price, time).
  - Observation history window T=1 (current step only).
  - Per-unit-time operating cost c_k ~ U[0.1, 1.0], fixed per SPS per episode.
  - Exogenous quality/attack-rate/resource processes follow a bounded random
    walk (documented as "highly dynamic" in the paper, exact dynamics unspecified).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    K: int = 4
    N: int = 3
    budget: float = 100.0
    p_min: float = 1.0
    p_max: float = 20.0
    tau_min: float = 0.1
    tau_max: float = 20.0
    q_max: float = 20.0
    cr_max: float = 10.0
    theta_scale: float = 5.0  # calibration constant for theta_n(x) = theta_scale * sqrt(x);
    # paper specifies only "concave," no scale; chosen so TPR utility can be
    # positive at realistic (quality, time, budget) magnitudes (see REPRODUCTION_REPORT.md).
    eta: float = 1.0
    xi: float = 1.0
    zeta: float = 1.0
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    delta: float = 1.0
    max_steps: int = 500
    process_noise: float = 0.05  # std of bounded random-walk step, as a fraction of range
    seed: int | None = None
    rho: float = 0.0  # Candidate A: threat-risk weight on quality, tilde_q = q*(1+rho*attack_rate); rho=0 recovers source reproduction exactly
    leader_mode: str = "greedy"  # Candidate B: "greedy" (Algorithm 2, source) or "exact" (per-service knapsack DP)
    knapsack_grid: int = 200  # discretization steps for the "exact" leader's DP over the budget axis
    attack_trace: np.ndarray | None = None  # empirical attack-intensity trace (e.g., derived from CIC-IDS-2017);
    # when set, attack_rate follows this real, non-stationary trace (per-agent phase-shifted) instead of a
    # synthetic bounded random walk. See code/experiments/build_attack_trace.py.
    attack_trace_noise: float = 0.05  # per-agent idiosyncratic noise added on top of the shared real trace


class AMIStackelbergEnv:
    """Multi-agent environment: K SPS agents choose (price, time) per service.

    obs per agent k: concat over n of [quality(k,n)/q_max, attack_rate(k)/1.0,
                                        resources(k)/cr_max]  -> dim = N + 2
    action per agent k: concat over n of [price(k,n) in [p_min,p_max],
                                           time(k,n) in [tau_min,tau_max]] -> dim = 2N
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.K, self.N = cfg.K, cfg.N
        self.obs_dim = self.N + 2
        self.act_dim = 2 * self.N
        self.op_cost = self.rng.uniform(0.1, 1.0, size=self.K)
        self._t = 0
        self.quality = None
        self.attack_rate = None
        self.resources = None
        self.reset()

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._t = 0
        self.quality = self.rng.uniform(0, self.cfg.q_max, size=(self.K, self.N))
        self.resources = self.rng.uniform(0, self.cfg.cr_max, size=self.K)
        if self.cfg.attack_trace is not None:
            trace = self.cfg.attack_trace
            self._trace_phase = self.rng.integers(0, len(trace), size=self.K)
            self.attack_rate = self._sample_trace(self._trace_phase)
        else:
            self.attack_rate = self.rng.uniform(0, 1.0, size=self.K)
        return self._get_obs()

    def _sample_trace(self, phase):
        trace = self.cfg.attack_trace
        base = trace[phase % len(trace)]
        noise = self.rng.normal(0, self.cfg.attack_trace_noise, size=self.K)
        return np.clip(base + noise, 0.0, 1.0)

    def _get_obs(self):
        obs = np.concatenate(
            [self.quality / self.cfg.q_max,
             self.attack_rate[:, None] / 1.0,
             self.resources[:, None] / self.cfg.cr_max],
            axis=1,
        )  # (K, N+2)
        return [obs[k].astype(np.float32) for k in range(self.K)]

    def _step_exogenous(self):
        c = self.cfg
        def walk(x, lo, hi):
            step = self.rng.normal(0, c.process_noise * (hi - lo), size=x.shape)
            return np.clip(x + step, lo, hi)
        self.quality = walk(self.quality, 0, c.q_max)
        self.resources = walk(self.resources, 0, c.cr_max)
        if c.attack_trace is not None:
            self._trace_phase = self._trace_phase + 1
            self.attack_rate = self._sample_trace(self._trace_phase)
        else:
            self.attack_rate = walk(self.attack_rate, 0, 1.0)

    def effective_quality(self):
        """Candidate A: attack-risk-weighted quality. rho=0 -> identical to
        the source paper's raw quality (exact reproduction)."""
        return self.quality * (1.0 + self.cfg.rho * self.attack_rate[:, None])

    def tpr_selection(self, prices: np.ndarray, times: np.ndarray):
        """Dispatches to the source paper's Algorithm 2 (greedy scoring) or
        Candidate B's exact per-service knapsack solve, per cfg.leader_mode.
        prices, times: (K, N) arrays. Returns sel: (K, N) boolean mask."""
        if self.cfg.leader_mode == "exact":
            return self._tpr_selection_exact(prices, times)
        return self._tpr_selection_greedy(prices, times)

    def _tpr_selection_greedy(self, prices, times):
        c = self.cfg
        eff_q = self.effective_quality()
        sel = np.zeros((self.K, self.N), dtype=bool)
        for n in range(self.N):
            bids = []
            for k in range(self.K):
                p, tau = prices[k, n], times[k, n]
                if p > c.p_max or tau > c.tau_max:
                    continue
                q_norm = eff_q[k, n] / c.q_max
                p_norm = (p - c.p_min) / (c.p_max - c.p_min)
                tau_norm = (tau - c.tau_min) / (c.tau_max - c.tau_min)
                score = c.eta * q_norm - c.xi * p_norm + c.zeta * tau_norm
                bids.append((score, k, p, tau))
            bids.sort(key=lambda b: b[0], reverse=True)
            spent = 0.0
            for score, k, p, tau in bids:
                cost = p * tau
                if spent + cost <= c.budget:
                    sel[k, n] = True
                    spent += cost
        return sel

    def _tpr_selection_exact(self, prices, times):
        """Candidate B: exact 0/1-knapsack maximizing
        theta_n(sum_k eff_q*tau*sel) - sum_k p*tau*sel  s.t. sum_k p*tau*sel <= budget.
        theta_n is monotonically increasing, so for a *fixed* accepted set the
        objective is monotone in aggregate eff_q*tau; we solve the equivalent
        knapsack "maximize value = eff_q*tau subject to cost = p*tau <= budget"
        via discretized DP (item weight = cost, item value = eff_q*tau), which
        is exact up to the budget discretization grid, then evaluate the true
        theta_n-based utility on the resulting selection."""
        c = self.cfg
        eff_q = self.effective_quality()
        sel = np.zeros((self.K, self.N), dtype=bool)
        grid = c.knapsack_grid
        for n in range(self.N):
            costs = prices[:, n] * times[:, n]
            values = eff_q[:, n] * times[:, n]
            valid = (prices[:, n] <= c.p_max) & (times[:, n] <= c.tau_max) & (costs > 0)
            idxs = np.where(valid)[0]
            if len(idxs) == 0:
                continue
            scale = c.budget / grid
            w = np.clip(np.round(costs[idxs] / scale).astype(int), 0, grid)
            v = values[idxs]
            dp = np.zeros(grid + 1)
            choice = np.zeros((len(idxs), grid + 1), dtype=bool)
            for i in range(len(idxs)):
                wi, vi = w[i], v[i]
                new_dp = dp.copy()
                if wi <= grid:
                    cand = dp[: grid + 1 - wi] + vi
                    take = cand > new_dp[wi:]
                    new_dp[wi:][take] = cand[take]
                    choice[i, wi:] = take
                dp = new_dp
            b = grid
            for i in range(len(idxs) - 1, -1, -1):
                if choice[i, b]:
                    sel[idxs[i], n] = True
                    b -= w[i]
        return sel

    def compute_rewards(self, actions: list[np.ndarray]):
        """Pure computation of rewards/info for a joint action, without
        advancing environment state. Used by step() and by the Greedy
        baseline for one-step lookahead."""
        c = self.cfg
        A = np.stack(actions, axis=0)  # (K, 2N)
        prices = np.clip(A[:, : self.N], c.p_min, c.p_max)
        times = np.clip(A[:, self.N :], c.tau_min, c.tau_max)

        sel = self.tpr_selection(prices, times)
        eff_q = self.effective_quality()  # Candidate A: rho=0 -> eff_q == self.quality (exact reproduction)

        # TPR utility per service: theta_n(sum_k eff_quality*time*sel) - sum_k price*time*sel
        tpr_util_per_service = np.zeros(self.N)
        for n in range(self.N):
            agg_quality_time = np.sum(eff_q[:, n] * times[:, n] * sel[:, n])
            payment = np.sum(prices[:, n] * times[:, n] * sel[:, n])
            tpr_util_per_service[n] = c.theta_scale * np.sqrt(max(agg_quality_time, 0.0)) - payment
        tpr_total_util = float(np.sum(tpr_util_per_service))

        # SPS utility: sum_n (price*time - cost*time) * sel
        sps_util = np.zeros(self.K)
        for k in range(self.K):
            revenue = np.sum(prices[k] * times[k] * sel[k])
            cost = self.op_cost[k] * np.sum(times[k] * sel[k])
            sps_util[k] = revenue - cost

        welfare = tpr_total_util + float(np.sum(sps_util))

        # Shaped reward, Eq. (26), normalized components in [0,1]-ish scale.
        rewards = np.zeros(self.K)
        for k in range(self.K):
            q_norm = eff_q[k] / c.q_max
            p_norm = (prices[k] - c.p_min) / (c.p_max - c.p_min)
            tau_norm = (times[k] - c.tau_min) / (c.tau_max - c.tau_min)
            per_service = (c.alpha * q_norm - c.beta * p_norm + c.gamma * tau_norm) * sel[k]
            rewards[k] = np.sum(per_service) + c.delta * welfare / self.K

        info = {
            "sps_util": sps_util,
            "tpr_util": tpr_total_util,
            "tpr_util_per_service": tpr_util_per_service,
            "welfare": welfare,
            "sel": sel,
            "prices": prices,
            "times": times,
        }
        return rewards, info

    def step(self, actions: list[np.ndarray]):
        """actions: list of K arrays, each shape (2N,) = [p_1..p_N, tau_1..tau_N]."""
        rewards, info = self.compute_rewards(actions)
        self._step_exogenous()
        self._t += 1
        done = self._t >= self.cfg.max_steps
        obs = self._get_obs()
        return obs, rewards, done, info
