"""F-TAPS: Follower-side Threat-Aware Parameter-Shared multi-agent actor-critic.

F-TAPS learns the follower-side bidding policy of a Stackelberg incentive
mechanism for honeypot deployment in smart-grid Advanced Metering
Infrastructure (AMI). A retailer (TPR, the leader) with a per-service budget
selects among bids from K small-scale power suppliers (SPSs, the followers)
for N honeypot services; each SPS bids a price and a time allocation per
service.

The model has two components:

1. Threat-risk-weighted objective. The data quality of SPS k for service n is
   replaced by  q_kn * (1 + rho * attack_rate_k)  in the TPR utility, in the
   TPR scoring rule, and in the training reward, so defense data is worth more
   where attacks are more frequent.

2. Parameter-shared, permutation-invariant backbone. One actor
   mu(o_k, ctx_-k) and one critic Q(o_k, a_k, ctx_-k) are reused by all K
   agents, where ctx_-k is a mean-and-max pooling over the other agents'
   observations (actor) or observation-action pairs (critic). The parameter
   count is therefore O(1) in K, and a trained policy can act for a different
   number of agents.

Public API
----------
EnvConfig, AMIStackelbergEnv   threat-weighted Stackelberg market
FTAPS                          the agent: fit(), act(), score(), save(), load()
steady_state_scores            welfare / utility / fairness from a history

Dependencies: numpy, torch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["EnvConfig", "AMIStackelbergEnv", "FTAPS", "steady_state_scores",
           "RHO", "ACTOR_LR", "CRITIC_LR"]

# Tuned values of the threat weight and the two learning rates.
RHO = 0.9935931423871028
ACTOR_LR = 4.335184350704857e-05
CRITIC_LR = 0.004557048882646388


# --------------------------------------------------------------------------
# Threat-weighted Stackelberg market
# --------------------------------------------------------------------------

@dataclass
class EnvConfig:
    K: int = 4                    # number of SPS agents (followers)
    N: int = 3                    # number of honeypot services
    budget: float = 100.0         # TPR budget per service
    p_min: float = 1.0            # price bounds
    p_max: float = 20.0
    tau_min: float = 0.1          # time-allocation bounds
    tau_max: float = 20.0
    q_max: float = 20.0           # maximum data quality
    cr_max: float = 10.0          # maximum edge-device resources
    theta_scale: float = 5.0      # theta_n(x) = theta_scale * sqrt(x)
    eta: float = 1.0              # TPR scoring weights (quality, price, time)
    xi: float = 1.0
    zeta: float = 1.0
    alpha: float = 1.0            # reward weights (quality, price, time, welfare)
    beta: float = 1.0
    gamma: float = 1.0
    delta: float = 1.0
    rho: float = RHO              # threat weight; rho = 0 disables threat weighting
    process_noise: float = 0.05   # random-walk step std, as a fraction of each range
    max_steps: int = 10**9        # episode length (default: one continuing episode)
    seed: int | None = None
    attack_trace: np.ndarray | None = None  # optional empirical attack-intensity
    # series in [0, 1]; when given, each agent reads it from its own random
    # starting phase instead of following a synthetic random walk
    attack_trace_noise: float = 0.05        # per-agent noise added to the trace


class AMIStackelbergEnv:
    """K SPS agents bid (price, time) per service; the TPR accepts bids by a
    scoring rule until the per-service budget binds.

    Observation of agent k (dim N + 2):
        [quality(k, 1..N) / q_max, attack_rate(k), resources(k) / cr_max]
    Action of agent k (dim 2N):
        [price(k, 1..N) in [p_min, p_max], time(k, 1..N) in [tau_min, tau_max]]
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.K, self.N = cfg.K, cfg.N
        self.obs_dim = self.N + 2
        self.act_dim = 2 * self.N
        self.op_cost = self.rng.uniform(0.1, 1.0, size=self.K)  # per-unit-time cost c_k
        self._t = 0
        self.quality = None
        self.attack_rate = None
        self.resources = None
        self.reset()

    @property
    def p_range(self):
        return (self.cfg.p_min, self.cfg.p_max)

    @property
    def tau_range(self):
        return (self.cfg.tau_min, self.cfg.tau_max)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._t = 0
        self.quality = self.rng.uniform(0, self.cfg.q_max, size=(self.K, self.N))
        self.resources = self.rng.uniform(0, self.cfg.cr_max, size=self.K)
        if self.cfg.attack_trace is not None:
            self._trace_phase = self.rng.integers(0, len(self.cfg.attack_trace), size=self.K)
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
        )  # (K, N + 2)
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
        """Threat-risk-weighted quality: q * (1 + rho * attack_rate)."""
        return self.quality * (1.0 + self.cfg.rho * self.attack_rate[:, None])

    def tpr_selection(self, prices: np.ndarray, times: np.ndarray):
        """Leader-side acceptance. For each service, bids are ranked by
        eta * quality - xi * price + zeta * time (each normalized) and accepted
        in that order while the budget allows.

        prices, times: (K, N). Returns a (K, N) boolean acceptance mask."""
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

    def compute_rewards(self, actions: list[np.ndarray]):
        """Rewards and utilities of a joint action, without advancing the state."""
        c = self.cfg
        A = np.stack(actions, axis=0)  # (K, 2N)
        prices = np.clip(A[:, : self.N], c.p_min, c.p_max)
        times = np.clip(A[:, self.N:], c.tau_min, c.tau_max)

        sel = self.tpr_selection(prices, times)
        eff_q = self.effective_quality()

        # TPR utility per service: theta_n(sum_k q~ * tau * sel) - sum_k p * tau * sel
        tpr_util_per_service = np.zeros(self.N)
        for n in range(self.N):
            agg_quality_time = np.sum(eff_q[:, n] * times[:, n] * sel[:, n])
            payment = np.sum(prices[:, n] * times[:, n] * sel[:, n])
            tpr_util_per_service[n] = c.theta_scale * np.sqrt(max(agg_quality_time, 0.0)) - payment
        tpr_total_util = float(np.sum(tpr_util_per_service))

        # SPS utility: sum_n (p - c_k) * tau * sel
        sps_util = np.zeros(self.K)
        for k in range(self.K):
            revenue = np.sum(prices[k] * times[k] * sel[k])
            cost = self.op_cost[k] * np.sum(times[k] * sel[k])
            sps_util[k] = revenue - cost

        welfare = tpr_total_util + float(np.sum(sps_util))

        # Training reward: normalized (quality, price, time) terms of the
        # accepted bids plus an equal share of the social welfare.
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
        """actions: K arrays of shape (2N,) = [p_1..p_N, tau_1..tau_N]."""
        rewards, info = self.compute_rewards(actions)
        self._step_exogenous()
        self._t += 1
        done = self._t >= self.cfg.max_steps
        return self._get_obs(), rewards, done, info


# --------------------------------------------------------------------------
# Parameter-shared, permutation-invariant actor-critic
# --------------------------------------------------------------------------

def pool_context(feats, k):
    """Mean-and-max pooling over all agents except k.

    feats: (B, K, D). Returns (B, 2D)."""
    others = torch.cat([feats[:, :k], feats[:, k + 1:]], dim=1)
    if others.shape[1] == 0:
        return torch.zeros(feats.shape[0], feats.shape[-1] * 2, device=feats.device)
    return torch.cat([others.mean(dim=1), others.max(dim=1).values], dim=-1)


def _scale(x01, lo, hi):
    return lo + x01 * (hi - lo)


class SharedActor(nn.Module):
    def __init__(self, obs_dim, act_dim, p_range, tau_range):
        super().__init__()
        self.N = act_dim // 2
        self.p_range, self.tau_range = p_range, tau_range
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 2 * obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim), nn.Sigmoid(),
        )

    def forward(self, own_obs, ctx):
        x01 = self.net(torch.cat([own_obs, ctx], dim=-1))
        p01, tau01 = x01[..., : self.N], x01[..., self.N:]
        return torch.cat([_scale(p01, *self.p_range), _scale(tau01, *self.tau_range)], dim=-1)


class SharedCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim + 2 * (obs_dim + act_dim), 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, own_obs, own_act, ctx):
        return self.net(torch.cat([own_obs, own_act, ctx], dim=-1))


class ReplayBuffer:
    def __init__(self, capacity, K, obs_dim, act_dim, rng):
        self.capacity, self.rng = capacity, rng
        self.obs = np.zeros((capacity, K, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, K, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, K), dtype=np.float32)
        self.next_obs = np.zeros((capacity, K, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.ptr, self.size = 0, 0

    def add(self, obs, act, rew, next_obs, done):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i] = obs, act, rew
        self.next_obs[i], self.done[i] = next_obs, float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = self.rng.randint(0, self.size, size=batch_size)

        def to_t(x):
            return torch.as_tensor(x, dtype=torch.float32, device=device)

        return (to_t(self.obs[idx]), to_t(self.act[idx]), to_t(self.rew[idx]),
                to_t(self.next_obs[idx]), to_t(self.done[idx]))


def steady_state_scores(history, tail_frac=0.1):
    """Steady-state scores from a per-step history: means over the final
    `tail_frac` of steps, and Jain's fairness index of the per-agent mean
    SPS utilities over the same window."""
    total = len(history["welfare"])
    n = max(1, int(total * tail_frac))
    mean_pa = np.asarray(history["sps_util_per_agent"][-n:]).mean(axis=0)
    jain = float((mean_pa.sum() ** 2) / (len(mean_pa) * np.sum(mean_pa ** 2) + 1e-8))
    return {
        "welfare": float(np.mean(history["welfare"][-n:])),
        "sps_utility": float(np.mean(history["sps_utility"][-n:])),
        "tpr_utility": float(np.mean(history["tpr_utility"][-n:])),
        "jain_fairness": jain,
    }


class FTAPS:
    """One shared actor and one shared critic for all K agents, trained
    off-policy with deterministic policy gradients.

    Parameters
    ----------
    K, obs_dim, act_dim : market dimensions (see AMIStackelbergEnv).
    p_range, tau_range  : (low, high) bounds of the price and time actions.
    noise_decay_steps   : steps over which exploration noise decays linearly
                          from noise_start to noise_end; if None, fit() sets
                          it to its own total_steps.
    seed                : seeds network initialization, exploration noise,
                          and replay sampling.
    """

    def __init__(self, K, obs_dim, act_dim, p_range, tau_range,
                 actor_lr=ACTOR_LR, critic_lr=CRITIC_LR, gamma=0.99, tau=0.001,
                 buffer_size=100_000, batch_size=512, grad_clip=0.5,
                 noise_start=0.3, noise_end=0.01, noise_decay_steps=None,
                 device="cpu", seed=0):
        self.K, self.obs_dim, self.act_dim = K, obs_dim, act_dim
        self.p_range, self.tau_range = tuple(p_range), tuple(tau_range)
        self.gamma, self.tau, self.batch_size, self.grad_clip = gamma, tau, batch_size, grad_clip
        self.noise_start, self.noise_end, self.noise_decay_steps = noise_start, noise_end, noise_decay_steps
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.rng = np.random.RandomState(seed)

        self.actor = SharedActor(obs_dim, act_dim, self.p_range, self.tau_range).to(self.device)
        self.target_actor = SharedActor(obs_dim, act_dim, self.p_range, self.tau_range).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.critic = SharedCritic(obs_dim, act_dim).to(self.device)
        self.target_critic = SharedCritic(obs_dim, act_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.buffer = ReplayBuffer(buffer_size, K, obs_dim, act_dim, self.rng)
        self._step = 0

    @classmethod
    def for_env(cls, env, **kwargs):
        """Build an agent sized for `env`."""
        return cls(env.K, env.obs_dim, env.act_dim, env.p_range, env.tau_range, **kwargs)

    def n_parameters(self):
        """Trainable parameters of the shared actor and critic."""
        return sum(p.numel() for m in (self.actor, self.critic) for p in m.parameters())

    def _noise_scale(self):
        decay = self.noise_decay_steps if self.noise_decay_steps is not None else 50_000
        frac = min(1.0, self._step / max(1, decay))
        return self.noise_start + frac * (self.noise_end - self.noise_start)

    @torch.no_grad()
    def act(self, obs_list, explore=False):
        """Joint action for any number of agents: one (2N,) array per agent."""
        feats = torch.as_tensor(np.stack(obs_list), dtype=torch.float32,
                                device=self.device).unsqueeze(0)  # (1, K, obs_dim)
        sigma = self._noise_scale()
        avg_range = ((self.p_range[1] - self.p_range[0]) + (self.tau_range[1] - self.tau_range[0])) / 2
        N = self.act_dim // 2
        actions = []
        for k in range(feats.shape[1]):
            a = self.actor(feats[:, k], pool_context(feats, k)).squeeze(0).cpu().numpy()
            if explore:
                a = a + self.rng.normal(0, sigma * avg_range, size=a.shape)
                a[:N] = np.clip(a[:N], *self.p_range)
                a[N:] = np.clip(a[N:], *self.tau_range)
            actions.append(a.astype(np.float32))
        return actions

    def store(self, obs, act, rew, next_obs, done):
        self.buffer.add(np.stack(obs), np.stack(act), np.array(rew), np.stack(next_obs), float(done))
        self._step += 1

    def update(self):
        """One gradient step on the shared critic and the shared actor, with
        losses summed over the K agents. Returns None until the buffer holds
        one batch."""
        if self.buffer.size < self.batch_size:
            return None
        obs, act, rew, next_obs, done = self.buffer.sample(self.batch_size, self.device)

        # Critic: one-step TD target from the target networks.
        with torch.no_grad():
            next_actions = torch.zeros_like(act)
            for k in range(self.K):
                next_actions[:, k] = self.target_actor(next_obs[:, k], pool_context(next_obs, k))
            next_feats = torch.cat([next_obs, next_actions], dim=-1)
        feats = torch.cat([obs, act], dim=-1)

        critic_loss = 0.0
        for k in range(self.K):
            with torch.no_grad():
                q_next = self.target_critic(next_obs[:, k], next_actions[:, k],
                                            pool_context(next_feats, k)).squeeze(-1)
                y = rew[:, k] + self.gamma * (1 - done) * q_next
            q = self.critic(obs[:, k], act[:, k], pool_context(feats, k)).squeeze(-1)
            critic_loss = critic_loss + F.mse_loss(q, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_opt.step()

        # Actor: deterministic policy gradient through the shared critic.
        actor_loss = 0.0
        for k in range(self.K):
            acted = act.clone()
            acted[:, k] = self.actor(obs[:, k], pool_context(obs, k))
            ctx = pool_context(torch.cat([obs, acted], dim=-1), k)
            actor_loss = actor_loss - self.critic(obs[:, k], acted[:, k], ctx).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_opt.step()

        # Soft target updates.
        for target, online in ((self.target_actor, self.actor), (self.target_critic, self.critic)):
            for tp, p in zip(target.parameters(), online.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return {"actor_loss": actor_loss.item() / self.K,
                "critic_loss": critic_loss.item() / self.K}

    def fit(self, env, total_steps=15_000, seed=None):
        """Train online in `env` for `total_steps` steps.

        Returns the per-step history: 'welfare', 'sps_utility' (sum over
        agents), 'tpr_utility', and 'sps_util_per_agent'."""
        if self.noise_decay_steps is None:
            self.noise_decay_steps = total_steps
        obs = env.reset(seed=seed)
        history = {"welfare": [], "sps_utility": [], "tpr_utility": [], "sps_util_per_agent": []}
        for _ in range(total_steps):
            actions = self.act(obs, explore=True)
            next_obs, rewards, done, info = env.step(actions)
            self.store(obs, actions, rewards, next_obs, done)
            self.update()
            obs = next_obs if not done else env.reset()
            history["welfare"].append(float(info["welfare"]))
            history["sps_utility"].append(float(np.sum(info["sps_util"])))
            history["tpr_utility"].append(float(info["tpr_util"]))
            history["sps_util_per_agent"].append(info["sps_util"].copy())
        return history

    def score(self, env, steps=1_000, seed=None, tail_frac=1.0):
        """Roll out the trained policy without exploration or learning and
        return its welfare, utilities, and Jain fairness (see
        steady_state_scores)."""
        obs = env.reset(seed=seed)
        history = {"welfare": [], "sps_utility": [], "tpr_utility": [], "sps_util_per_agent": []}
        for _ in range(steps):
            next_obs, _, done, info = env.step(self.act(obs, explore=False))
            obs = next_obs if not done else env.reset()
            history["welfare"].append(float(info["welfare"]))
            history["sps_utility"].append(float(np.sum(info["sps_util"])))
            history["tpr_utility"].append(float(info["tpr_util"]))
            history["sps_util_per_agent"].append(info["sps_util"].copy())
        return steady_state_scores(history, tail_frac)

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path):
        state = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
