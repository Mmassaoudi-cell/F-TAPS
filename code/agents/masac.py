"""MASAC benchmark: multi-agent soft actor-critic. Independent stochastic
Gaussian actors (reusing ippo.py's GaussianActor) + MADDPG-style centralized
twin critics with an entropy-regularized target. Entropy coefficient is a
fixed hyperparameter (no automatic temperature tuning) -- documented
simplification, consistent in spirit with the other benchmarks' fixed
exploration schedules."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from code.agents.ippo import GaussianActor
from code.agents.maddpg import Critic, ReplayBuffer


class MASAC:
    def __init__(self, K, obs_dim, act_dim, p_range, tau_range,
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005,
                 buffer_size=100_000, batch_size=512, grad_clip=0.5,
                 alpha=0.01, device="cpu", seed=0):
        self.K, self.obs_dim, self.act_dim = K, obs_dim, act_dim
        self.gamma, self.tau, self.batch_size, self.grad_clip, self.alpha = gamma, tau, batch_size, grad_clip, alpha
        self.device = torch.device(device)
        torch.manual_seed(seed)

        joint_obs_dim, joint_act_dim = K * obs_dim, K * act_dim
        self.actors, self.actor_opts = [], []
        self.critics1, self.critics2 = [], []
        self.target_critics1, self.target_critics2 = [], []
        self.critic_opts = []
        for _ in range(K):
            a = GaussianActor(obs_dim, act_dim, p_range, tau_range).to(self.device)
            c1 = Critic(joint_obs_dim, joint_act_dim).to(self.device)
            c2 = Critic(joint_obs_dim, joint_act_dim).to(self.device)
            tc1 = Critic(joint_obs_dim, joint_act_dim).to(self.device)
            tc2 = Critic(joint_obs_dim, joint_act_dim).to(self.device)
            tc1.load_state_dict(c1.state_dict()); tc2.load_state_dict(c2.state_dict())
            self.actors.append(a)
            self.critics1.append(c1); self.critics2.append(c2)
            self.target_critics1.append(tc1); self.target_critics2.append(tc2)
            self.actor_opts.append(torch.optim.Adam(a.parameters(), lr=actor_lr))
            self.critic_opts.append(torch.optim.Adam(
                list(c1.parameters()) + list(c2.parameters()), lr=critic_lr))

        self.buffer = ReplayBuffer(buffer_size, K, obs_dim, act_dim)
        self.p_range, self.tau_range = p_range, tau_range

    @torch.no_grad()
    def act(self, obs_list, explore=True):
        actions = []
        for k in range(self.K):
            o = torch.as_tensor(obs_list[k], dtype=torch.float32, device=self.device).unsqueeze(0)
            if explore:
                a, _ = self.actors[k].act(o)
            else:
                a = self.actors[k].scaled_mean(o)
            actions.append(a.squeeze(0).cpu().numpy().astype(np.float32))
        return actions

    def store(self, obs, act, rew, next_obs, done):
        self.buffer.add(np.stack(obs), np.stack(act), np.array(rew), np.stack(next_obs), float(done))

    def update(self):
        if self.buffer.size < self.batch_size:
            return None
        obs, act, rew, next_obs, done = self.buffer.sample(self.batch_size, self.device)
        B = obs.shape[0]
        joint_obs, joint_act = obs.reshape(B, -1), act.reshape(B, -1)

        with torch.no_grad():
            next_actions, next_logps = [], []
            for k in range(self.K):
                a, logp = self.actors[k].act(next_obs[:, k])
                next_actions.append(a); next_logps.append(logp)
            joint_next_act = torch.cat(next_actions, dim=-1)
            joint_next_obs = next_obs.reshape(B, -1)

        for k in range(self.K):
            with torch.no_grad():
                q1n = self.target_critics1[k](joint_next_obs, joint_next_act).squeeze(-1)
                q2n = self.target_critics2[k](joint_next_obs, joint_next_act).squeeze(-1)
                y = rew[:, k] + self.gamma * (1 - done) * (torch.min(q1n, q2n) - self.alpha * next_logps[k])
            q1 = self.critics1[k](joint_obs, joint_act).squeeze(-1)
            q2 = self.critics2[k](joint_obs, joint_act).squeeze(-1)
            critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            self.critic_opts[k].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(list(self.critics1[k].parameters()) + list(self.critics2[k].parameters()), self.grad_clip)
            self.critic_opts[k].step()

            new_a, logp = self.actors[k].act(obs[:, k])
            acted = [new_a if j == k else act[:, j] for j in range(self.K)]
            joint_act_for_actor = torch.cat(acted, dim=-1)
            q_pi = torch.min(self.critics1[k](joint_obs, joint_act_for_actor),
                              self.critics2[k](joint_obs, joint_act_for_actor)).squeeze(-1)
            actor_loss = (self.alpha * logp - q_pi).mean()
            self.actor_opts[k].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[k].parameters(), self.grad_clip)
            self.actor_opts[k].step()

        for k in range(self.K):
            for tp, p in zip(self.target_critics1[k].parameters(), self.critics1[k].parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for tp, p in zip(self.target_critics2[k].parameters(), self.critics2[k].parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
        return {}
