"""Unified trainer for Candidates C, D, E: permutation-invariant, parameter-
shared multi-agent actor-critic with a pluggable "backbone" (how an agent
aggregates information about the other K-1 agents) and an optional twin
critic (TD3/MATD3-style clipped double-Q).

backbone:
  "pool"      -> Candidate C: mean+max pooling over other agents (DeepSets-style)
  "attention" -> Candidate D: learned single-query self-attention over other agents
  "meanfield" -> Candidate E: plain mean over other agents (mean-field MARL)

twin_critic=True adds the MATD3-style min-of-two-critics target, used in D and E.

Unlike code/agents/maddpg.py (K independent actor/critic pairs, O(K) params),
here ONE actor and ONE (or two, if twin) critic are shared across all K
agents, giving O(1) parameter count in K and the ability to evaluate a
policy trained at one K on a different K at test time.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def scale_to_range(x01, lo, hi):
    return lo + x01 * (hi - lo)


class AttnContext(nn.Module):
    def __init__(self, in_dim, d_model=64, n_heads=4):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.out_dim = d_model

    def forward(self, feats, k):
        # feats: (B, K, in_dim)
        emb = self.embed(feats)
        query = emb[:, k : k + 1]
        others = torch.cat([emb[:, :k], emb[:, k + 1 :]], dim=1)
        if others.shape[1] == 0:
            return torch.zeros(feats.shape[0], self.out_dim, device=feats.device)
        out, _ = self.mha(query, others, others)
        return out.squeeze(1)


def pool_context(feats, k):
    others = torch.cat([feats[:, :k], feats[:, k + 1 :]], dim=1)
    if others.shape[1] == 0:
        return torch.zeros(feats.shape[0], feats.shape[-1] * 2, device=feats.device)
    return torch.cat([others.mean(dim=1), others.max(dim=1).values], dim=-1)


def meanfield_context(feats, k):
    others = torch.cat([feats[:, :k], feats[:, k + 1 :]], dim=1)
    if others.shape[1] == 0:
        return torch.zeros(feats.shape[0], feats.shape[-1], device=feats.device)
    return others.mean(dim=1)


class HybridActor(nn.Module):
    def __init__(self, obs_dim, act_dim, ctx_dim, p_range, tau_range):
        super().__init__()
        self.N = act_dim // 2
        self.p_range, self.tau_range = p_range, tau_range
        self.net = nn.Sequential(
            nn.Linear(obs_dim + ctx_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim), nn.Sigmoid(),
        )

    def forward(self, own_obs, ctx):
        x01 = self.net(torch.cat([own_obs, ctx], dim=-1))
        p01, tau01 = x01[..., : self.N], x01[..., self.N :]
        return torch.cat([scale_to_range(p01, *self.p_range),
                           scale_to_range(tau01, *self.tau_range)], dim=-1)


class HybridCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, ctx_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim + ctx_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, own_obs, own_act, ctx):
        return self.net(torch.cat([own_obs, own_act, ctx], dim=-1))


class ReplayBuffer:
    def __init__(self, capacity, K, obs_dim, act_dim):
        self.capacity, self.K = capacity, K
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
        idx = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)
        return (to_t(self.obs[idx]), to_t(self.act[idx]), to_t(self.rew[idx]),
                to_t(self.next_obs[idx]), to_t(self.done[idx]))


class HybridMARL:
    def __init__(self, K, obs_dim, act_dim, p_range, tau_range, backbone="attention",
                 twin_critic=True, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.001,
                 buffer_size=100_000, batch_size=512, grad_clip=0.5,
                 noise_start=0.3, noise_end=0.01, noise_decay_steps=50_000,
                 policy_noise=0.1, policy_noise_clip=0.2, device="cpu", seed=0):
        assert backbone in ("pool", "attention", "meanfield")
        self.K, self.obs_dim, self.act_dim = K, obs_dim, act_dim
        self.backbone, self.twin_critic = backbone, twin_critic
        self.gamma, self.tau, self.batch_size, self.grad_clip = gamma, tau, batch_size, grad_clip
        self.policy_noise, self.policy_noise_clip = policy_noise, policy_noise_clip
        self.device = torch.device(device)
        torch.manual_seed(seed)

        if backbone == "pool":
            actor_ctx_dim, critic_ctx_dim = 2 * obs_dim, 2 * (obs_dim + act_dim)
            self.actor_ctx_fn = pool_context
            self.critic_ctx_fn = pool_context
            self.actor_ctx_module = None
            self.critic_ctx_module = None
        elif backbone == "meanfield":
            actor_ctx_dim, critic_ctx_dim = obs_dim, obs_dim + act_dim
            self.actor_ctx_fn = meanfield_context
            self.critic_ctx_fn = meanfield_context
            self.actor_ctx_module = None
            self.critic_ctx_module = None
        else:  # attention
            self.actor_ctx_module = AttnContext(obs_dim).to(self.device)
            self.critic_ctx_module = AttnContext(obs_dim + act_dim).to(self.device)
            actor_ctx_dim = self.actor_ctx_module.out_dim
            critic_ctx_dim = self.critic_ctx_module.out_dim
            self.actor_ctx_fn = lambda feats, k: self.actor_ctx_module(feats, k)
            self.critic_ctx_fn = lambda feats, k: self.critic_ctx_module(feats, k)

        self.actor = HybridActor(obs_dim, act_dim, actor_ctx_dim, p_range, tau_range).to(self.device)
        self.target_actor = HybridActor(obs_dim, act_dim, actor_ctx_dim, p_range, tau_range).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())

        n_critics = 2 if twin_critic else 1
        self.critics = [HybridCritic(obs_dim, act_dim, critic_ctx_dim).to(self.device) for _ in range(n_critics)]
        self.target_critics = [HybridCritic(obs_dim, act_dim, critic_ctx_dim).to(self.device) for _ in range(n_critics)]
        for c, tc in zip(self.critics, self.target_critics):
            tc.load_state_dict(c.state_dict())

        actor_params = list(self.actor.parameters())
        if self.actor_ctx_module is not None:
            actor_params += list(self.actor_ctx_module.parameters())
        self.actor_opt = torch.optim.Adam(actor_params, lr=actor_lr)

        critic_params = []
        for c in self.critics:
            critic_params += list(c.parameters())
        if self.critic_ctx_module is not None:
            critic_params += list(self.critic_ctx_module.parameters())
        self.critic_opt = torch.optim.Adam(critic_params, lr=critic_lr)

        self.buffer = ReplayBuffer(buffer_size, K, obs_dim, act_dim)
        self.p_range, self.tau_range = p_range, tau_range
        self.noise_start, self.noise_end, self.noise_decay_steps = noise_start, noise_end, noise_decay_steps
        self._step = 0

    def _noise_scale(self):
        frac = min(1.0, self._step / max(1, self.noise_decay_steps))
        return self.noise_start + frac * (self.noise_end - self.noise_start)

    @torch.no_grad()
    def act(self, obs_list, explore=True):
        feats = torch.as_tensor(np.stack(obs_list), dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,K,obs)
        actions = []
        sigma = self._noise_scale()
        avg_range = ((self.p_range[1] - self.p_range[0]) + (self.tau_range[1] - self.tau_range[0])) / 2
        for k in range(self.K):
            ctx = self.actor_ctx_fn(feats, k)
            own = feats[:, k]
            a = self.actor(own, ctx).squeeze(0).cpu().numpy()
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

        # ---- critic update (shared weights, gradients summed over all K agents) ----
        with torch.no_grad():
            next_actions = torch.zeros_like(act)
            for k in range(self.K):
                ctx = self.actor_ctx_fn(next_obs, k)
                a = self.target_actor(next_obs[:, k], ctx)
                if self.twin_critic:
                    noise = (torch.randn_like(a) * self.policy_noise).clamp(
                        -self.policy_noise_clip, self.policy_noise_clip)
                    N = self.act_dim // 2
                    a = a.clone()
                    a[:, :N] = (a[:, :N] + noise[:, :N]).clamp(*self.p_range)
                    a[:, N:] = (a[:, N:] + noise[:, N:]).clamp(*self.tau_range)
                next_actions[:, k] = a

        critic_loss_total = 0.0
        for k in range(self.K):
            ctx_next = self.critic_ctx_fn(torch.cat([next_obs, next_actions], dim=-1), k) \
                if False else self.critic_ctx_fn(self._feat_cat(next_obs, next_actions), k)
            with torch.no_grad():
                qs_next = [tc(next_obs[:, k], next_actions[:, k], ctx_next) for tc in self.target_critics]
                q_next = torch.min(torch.stack(qs_next, dim=0), dim=0).values.squeeze(-1) if self.twin_critic \
                    else qs_next[0].squeeze(-1)
                y = rew[:, k] + self.gamma * (1 - done) * q_next

            ctx_cur = self.critic_ctx_fn(self._feat_cat(obs, act), k)
            for c in self.critics:
                q = c(obs[:, k], act[:, k], ctx_cur).squeeze(-1)
                critic_loss_total = critic_loss_total + F.mse_loss(q, y)

        self.critic_opt.zero_grad()
        critic_loss_total.backward()
        params = []
        for c in self.critics:
            params += list(c.parameters())
        if self.critic_ctx_module is not None:
            params += list(self.critic_ctx_module.parameters())
        nn.utils.clip_grad_norm_(params, self.grad_clip)
        self.critic_opt.step()

        # ---- actor update ----
        actor_loss_total = 0.0
        for k in range(self.K):
            ctx_a = self.actor_ctx_fn(obs, k)
            new_a = self.actor(obs[:, k], ctx_a)
            acted = act.clone()
            acted[:, k] = new_a
            ctx_c = self.critic_ctx_fn(self._feat_cat(obs, acted), k)
            actor_loss_total = actor_loss_total - self.critics[0](obs[:, k], acted[:, k], ctx_c).mean()

        self.actor_opt.zero_grad()
        actor_loss_total.backward()
        actor_params = list(self.actor.parameters())
        if self.actor_ctx_module is not None:
            actor_params += list(self.actor_ctx_module.parameters())
        nn.utils.clip_grad_norm_(actor_params, self.grad_clip)
        self.actor_opt.step()

        for tp, p in zip(self.target_actor.parameters(), self.actor.parameters()):
            tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
        for tc, c in zip(self.target_critics, self.critics):
            for tp, p in zip(tc.parameters(), c.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        return {"actor_loss": actor_loss_total.item() / self.K,
                "critic_loss": critic_loss_total.item() / (self.K * len(self.critics))}

    @staticmethod
    def _feat_cat(obs, act):
        return torch.cat([obs, act], dim=-1)
