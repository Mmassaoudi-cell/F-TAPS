"""MATD3 benchmark: MADDPG's independent-per-agent centralized-critic
architecture (reused from maddpg.py) + TD3-style twin critics and target
policy smoothing. Kept as a separate file so maddpg.py remains an untouched,
literal reproduction of the source paper."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from code.agents.maddpg import Actor, Critic, ReplayBuffer


class MATD3:
    def __init__(self, K, obs_dim, act_dim, p_range, tau_range,
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.001,
                 buffer_size=100_000, batch_size=512, grad_clip=0.5,
                 noise_start=0.3, noise_end=0.01, noise_decay_steps=50_000,
                 policy_noise=0.1, policy_noise_clip=0.2, device="cpu", seed=0):
        self.K, self.obs_dim, self.act_dim = K, obs_dim, act_dim
        self.gamma, self.tau, self.batch_size, self.grad_clip = gamma, tau, batch_size, grad_clip
        self.policy_noise, self.policy_noise_clip = policy_noise, policy_noise_clip
        self.device = torch.device(device)
        torch.manual_seed(seed)

        joint_obs_dim, joint_act_dim = K * obs_dim, K * act_dim
        self.actors, self.target_actors, self.actor_opts = [], [], []
        self.critics1, self.critics2 = [], []
        self.target_critics1, self.target_critics2 = [], []
        self.critic_opts = []
        for _ in range(K):
            a = Actor(obs_dim, act_dim, p_range, tau_range).to(self.device)
            ta = Actor(obs_dim, act_dim, p_range, tau_range).to(self.device)
            ta.load_state_dict(a.state_dict())
            c1 = Critic(joint_obs_dim, joint_act_dim).to(self.device)
            c2 = Critic(joint_obs_dim, joint_act_dim).to(self.device)
            tc1 = Critic(joint_obs_dim, joint_act_dim).to(self.device)
            tc2 = Critic(joint_obs_dim, joint_act_dim).to(self.device)
            tc1.load_state_dict(c1.state_dict())
            tc2.load_state_dict(c2.state_dict())
            self.actors.append(a); self.target_actors.append(ta)
            self.critics1.append(c1); self.critics2.append(c2)
            self.target_critics1.append(tc1); self.target_critics2.append(tc2)
            self.actor_opts.append(torch.optim.Adam(a.parameters(), lr=actor_lr))
            self.critic_opts.append(torch.optim.Adam(
                list(c1.parameters()) + list(c2.parameters()), lr=critic_lr))

        self.buffer = ReplayBuffer(buffer_size, K, obs_dim, act_dim)
        self.p_range, self.tau_range = p_range, tau_range
        self.noise_start, self.noise_end, self.noise_decay_steps = noise_start, noise_end, noise_decay_steps
        self._step = 0

    def _noise_scale(self):
        frac = min(1.0, self._step / max(1, self.noise_decay_steps))
        return self.noise_start + frac * (self.noise_end - self.noise_start)

    @torch.no_grad()
    def act(self, obs_list, explore=True):
        actions = []
        sigma = self._noise_scale()
        avg_range = ((self.p_range[1]-self.p_range[0]) + (self.tau_range[1]-self.tau_range[0]))/2
        for k in range(self.K):
            o = torch.as_tensor(obs_list[k], dtype=torch.float32, device=self.device).unsqueeze(0)
            a = self.actors[k](o).squeeze(0).cpu().numpy()
            if explore:
                a = a + np.random.normal(0, sigma * avg_range, size=a.shape)
                N = self.act_dim // 2
                a[:N] = np.clip(a[:N], *self.p_range)
                a[N:] = np.clip(a[N:], *self.tau_range)
            actions.append(a.astype(np.float32))
        return actions

    def store(self, obs, act, rew, next_obs, done):
        self.buffer.add(np.stack(obs), np.stack(act), np.array(rew), np.stack(next_obs), float(done))
        self._step += 1

    def update(self):
        if self.buffer.size < self.batch_size:
            return None
        obs, act, rew, next_obs, done = self.buffer.sample(self.batch_size, self.device)
        B = obs.shape[0]
        joint_obs, joint_act, joint_next_obs = obs.reshape(B,-1), act.reshape(B,-1), next_obs.reshape(B,-1)

        with torch.no_grad():
            next_actions = []
            for k in range(self.K):
                a = self.target_actors[k](next_obs[:, k])
                noise = (torch.randn_like(a) * self.policy_noise).clamp(
                    -self.policy_noise_clip, self.policy_noise_clip)
                N = self.act_dim // 2
                a = a.clone()
                a[:, :N] = (a[:, :N] + noise[:, :N]).clamp(*self.p_range)
                a[:, N:] = (a[:, N:] + noise[:, N:]).clamp(*self.tau_range)
                next_actions.append(a)
            joint_next_act = torch.cat(next_actions, dim=-1)

        for k in range(self.K):
            with torch.no_grad():
                q1n = self.target_critics1[k](joint_next_obs, joint_next_act).squeeze(-1)
                q2n = self.target_critics2[k](joint_next_obs, joint_next_act).squeeze(-1)
                y = rew[:, k] + self.gamma * (1 - done) * torch.min(q1n, q2n)
            q1 = self.critics1[k](joint_obs, joint_act).squeeze(-1)
            q2 = self.critics2[k](joint_obs, joint_act).squeeze(-1)
            critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            self.critic_opts[k].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(list(self.critics1[k].parameters()) + list(self.critics2[k].parameters()), self.grad_clip)
            self.critic_opts[k].step()

            acted = [self.actors[j](obs[:, j]) if j == k else act[:, j] for j in range(self.K)]
            joint_act_for_actor = torch.cat(acted, dim=-1)
            actor_loss = -self.critics1[k](joint_obs, joint_act_for_actor).mean()
            self.actor_opts[k].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[k].parameters(), self.grad_clip)
            self.actor_opts[k].step()

        for k in range(self.K):
            for tp, p in zip(self.target_actors[k].parameters(), self.actors[k].parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for tp, p in zip(self.target_critics1[k].parameters(), self.critics1[k].parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for tp, p in zip(self.target_critics2[k].parameters(), self.critics2[k].parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
        return {}
