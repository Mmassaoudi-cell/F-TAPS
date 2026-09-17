"""MADDPG reproduction per Table II / Algorithm 1 of the source paper.

Simplification (documented): the paper stores joint transitions in a
per-agent buffer Pi_k; since every agent observes the same joint transition
at a given timestep (only the reward differs), we use a single shared replay
buffer of joint transitions and slice out each agent's reward at sample time.
This is functionally equivalent to K synchronized per-agent buffers and
avoids K-fold memory duplication.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def scale_to_range(x01, lo, hi):
    return lo + x01 * (hi - lo)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, p_range, tau_range):
        super().__init__()
        self.N = act_dim // 2
        self.p_min, self.p_max = p_range
        self.tau_min, self.tau_max = tau_range
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim), nn.Sigmoid(),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, obs):
        x01 = self.net(obs)
        p01, tau01 = x01[..., : self.N], x01[..., self.N :]
        p = scale_to_range(p01, self.p_min, self.p_max)
        tau = scale_to_range(tau01, self.tau_min, self.tau_max)
        return torch.cat([p, tau], dim=-1)


class Critic(nn.Module):
    def __init__(self, joint_obs_dim, joint_act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(joint_obs_dim + joint_act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, joint_obs, joint_act):
        return self.net(torch.cat([joint_obs, joint_act], dim=-1))


class ReplayBuffer:
    def __init__(self, capacity, K, obs_dim, act_dim):
        self.capacity = capacity
        self.K, self.obs_dim, self.act_dim = K, obs_dim, act_dim
        self.obs = np.zeros((capacity, K, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, K, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, K), dtype=np.float32)
        self.next_obs = np.zeros((capacity, K, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done):
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.done[i] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        return (to_t(self.obs[idx]), to_t(self.act[idx]), to_t(self.rew[idx]),
                to_t(self.next_obs[idx]), to_t(self.done[idx]))


class MADDPG:
    def __init__(self, K, obs_dim, act_dim, p_range, tau_range,
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.001,
                 buffer_size=100_000, batch_size=512, grad_clip=0.5,
                 noise_start=0.3, noise_end=0.01, noise_decay_steps=50_000,
                 device="cpu", seed=0):
        self.K, self.obs_dim, self.act_dim = K, obs_dim, act_dim
        self.gamma, self.tau, self.batch_size, self.grad_clip = gamma, tau, batch_size, grad_clip
        self.device = torch.device(device)
        torch.manual_seed(seed)

        self.actors, self.critics = [], []
        self.target_actors, self.target_critics = [], []
        self.actor_opts, self.critic_opts = [], []
        joint_obs_dim, joint_act_dim = K * obs_dim, K * act_dim
        for _ in range(K):
            a = Actor(obs_dim, act_dim, p_range, tau_range).to(self.device)
            c = Critic(joint_obs_dim, joint_act_dim).to(self.device)
            ta = Actor(obs_dim, act_dim, p_range, tau_range).to(self.device)
            tc = Critic(joint_obs_dim, joint_act_dim).to(self.device)
            ta.load_state_dict(a.state_dict())
            tc.load_state_dict(c.state_dict())
            self.actors.append(a); self.critics.append(c)
            self.target_actors.append(ta); self.target_critics.append(tc)
            self.actor_opts.append(torch.optim.Adam(a.parameters(), lr=actor_lr))
            self.critic_opts.append(torch.optim.Adam(c.parameters(), lr=critic_lr))

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
        for k in range(self.K):
            o = torch.as_tensor(obs_list[k], dtype=torch.float32, device=self.device).unsqueeze(0)
            a = self.actors[k](o).squeeze(0).cpu().numpy()
            if explore:
                a = a + np.random.normal(0, sigma * ((np.array(self.p_range[1] - self.p_range[0]).item()
                                                        + (self.tau_range[1] - self.tau_range[0])) / 2), size=a.shape)
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
        joint_obs = obs.reshape(B, -1)
        joint_act = act.reshape(B, -1)
        joint_next_obs = next_obs.reshape(B, -1)

        with torch.no_grad():
            next_actions = [self.target_actors[k](next_obs[:, k]) for k in range(self.K)]
            joint_next_act = torch.cat(next_actions, dim=-1)

        losses = {"actor": [], "critic": []}
        for k in range(self.K):
            with torch.no_grad():
                target_q = self.target_critics[k](joint_next_obs, joint_next_act).squeeze(-1)
                y = rew[:, k] + self.gamma * (1 - done) * target_q
            q = self.critics[k](joint_obs, joint_act).squeeze(-1)
            critic_loss = F.mse_loss(q, y)
            self.critic_opts[k].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[k].parameters(), self.grad_clip)
            self.critic_opts[k].step()

            acted = [self.actors[j](obs[:, j]) if j == k else act[:, j] for j in range(self.K)]
            joint_act_for_actor = torch.cat(acted, dim=-1)
            actor_loss = -self.critics[k](joint_obs, joint_act_for_actor).mean()
            self.actor_opts[k].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actors[k].parameters(), self.grad_clip)
            self.actor_opts[k].step()

            losses["actor"].append(actor_loss.item())
            losses["critic"].append(critic_loss.item())

        for k in range(self.K):
            for tp, p in zip(self.target_actors[k].parameters(), self.actors[k].parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for tp, p in zip(self.target_critics[k].parameters(), self.critics[k].parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
        return losses
