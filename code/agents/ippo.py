"""Independent PPO (I-PPO) baseline, Table III of the source paper."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def scale_to_range(x01, lo, hi):
    return lo + x01 * (hi - lo)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, p_range, tau_range):
        super().__init__()
        self.N = act_dim // 2
        self.p_range, self.tau_range = p_range, tau_range
        self.mean_net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim), nn.Sigmoid(),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)

    def scaled_mean(self, obs):
        x01 = self.mean_net(obs)
        p01, tau01 = x01[..., : self.N], x01[..., self.N :]
        p = scale_to_range(p01, *self.p_range)
        tau = scale_to_range(tau01, *self.tau_range)
        return torch.cat([p, tau], dim=-1)

    def dist(self, obs):
        mean = self.scaled_mean(obs)
        std = torch.exp(self.log_std).clamp(1e-3, 5.0)
        return Normal(mean, std)

    def act(self, obs):
        d = self.dist(obs)
        raw = d.rsample()
        N = self.N
        p_clamped = raw[..., :N].clamp(*self.p_range)
        t_clamped = raw[..., N:].clamp(*self.tau_range)
        raw_clamped = torch.cat([p_clamped, t_clamped], dim=-1)
        logp = d.log_prob(raw).sum(-1)
        return raw_clamped, logp


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class IPPOAgent:
    def __init__(self, obs_dim, act_dim, p_range, tau_range, actor_lr=3e-4,
                 critic_lr=1e-3, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
                 update_epochs=4, minibatch_size=64, entropy_coef=0.01,
                 value_coef=0.5, grad_clip=0.5, device="cpu"):
        self.device = torch.device(device)
        self.actor = GaussianActor(obs_dim, act_dim, p_range, tau_range).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)
        self.opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma, self.gae_lambda, self.clip_eps = gamma, gae_lambda, clip_eps
        self.update_epochs, self.minibatch_size = update_epochs, minibatch_size
        self.entropy_coef, self.value_coef, self.grad_clip = entropy_coef, value_coef, grad_clip
        self.rollout = []

    @torch.no_grad()
    def act(self, obs):
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, logp = self.actor.act(o)
        v = self.critic(o)
        return a.squeeze(0).cpu().numpy(), logp.item(), v.item()

    def store(self, obs, act, logp, val, rew, done):
        self.rollout.append((obs, act, logp, val, rew, done))

    def _compute_gae(self, rewards, values, dones, last_val):
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        values_ext = values + [last_val]
        for t in reversed(range(T)):
            delta = rewards[t] + self.gamma * values_ext[t + 1] * (1 - dones[t]) - values_ext[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            adv[t] = gae
        returns = adv + np.array(values, dtype=np.float32)
        return adv, returns

    def finish_rollout(self, last_val):
        obs = np.array([r[0] for r in self.rollout], dtype=np.float32)
        act = np.array([r[1] for r in self.rollout], dtype=np.float32)
        old_logp = np.array([r[2] for r in self.rollout], dtype=np.float32)
        val = [r[3] for r in self.rollout]
        rew = [r[4] for r in self.rollout]
        done = [r[5] for r in self.rollout]
        adv, ret = self._compute_gae(rew, val, done, last_val)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        to_t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        obs_t, act_t, old_logp_t, adv_t, ret_t = map(to_t, (obs, act, old_logp, adv, ret))

        T = len(self.rollout)
        idxs = np.arange(T)
        for _ in range(self.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, T, self.minibatch_size):
                mb = idxs[start:start + self.minibatch_size]
                d = self.actor.dist(obs_t[mb])
                logp = d.log_prob(act_t[mb]).sum(-1)
                ratio = torch.exp(logp - old_logp_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[mb]
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy = d.entropy().sum(-1).mean()
                v = self.critic(obs_t[mb])
                value_loss = ((v - ret_t[mb]) ** 2).mean()
                loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), self.grad_clip)
                self.opt.step()
        self.rollout = []
