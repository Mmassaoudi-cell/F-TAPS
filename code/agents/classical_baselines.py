"""Non-DRL benchmarks: a Bayesian-optimized static mechanism, a simple
evolutionary-search (ES) policy, and a closed-form contract-theory pricing
rule. These fill weakness Rank 5 (the source paper never compares against
any non-RL optimization or classical mechanism-design method)."""
from __future__ import annotations
import numpy as np


class BOStaticPolicy:
    """A single (price_n, time_n) pair per service, shared by all SPSs,
    tuned offline via Bayesian optimization (Optuna TPE) against validation
    rollouts. Represents "optimize a static mechanism" as a cheaper
    alternative to adaptive per-step DRL."""
    def __init__(self, N, params: np.ndarray):
        self.N = N
        self.params = params  # shape (2N,) = [p_1..p_N, t_1..t_N]

    def act(self, obs):
        return self.params.astype(np.float32)


def tune_bo_static(env_factory, N, p_range, tau_range, n_trials=40, n_eval_steps=500, seed=0):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        p = [trial.suggest_float(f"p_{n}", *p_range) for n in range(N)]
        t = [trial.suggest_float(f"t_{n}", *tau_range) for n in range(N)]
        params = np.array(p + t, dtype=np.float32)
        env = env_factory()
        obs = env.reset(seed=seed)
        total = 0.0
        for _ in range(n_eval_steps):
            actions = [params for _ in range(env.K)]
            obs, rewards, done, info = env.step(actions)
            total += float(info["welfare"])
            if done:
                obs = env.reset()
        return total

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    params = np.array([best[f"p_{n}"] for n in range(N)] + [best[f"t_{n}"] for n in range(N)],
                       dtype=np.float32)
    return BOStaticPolicy(N, params)


class LinearESPolicy:
    """Shared linear policy a = clip(W @ obs + b), weights evolved with a
    simple (mu, lambda) evolution strategy (no external CMA-ES dependency)."""
    def __init__(self, obs_dim, act_dim, p_range, tau_range, theta: np.ndarray | None = None):
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.p_range, self.tau_range = p_range, tau_range
        self.n_params = obs_dim * act_dim + act_dim
        self.theta = theta if theta is not None else np.zeros(self.n_params, dtype=np.float32)

    def _unpack(self, theta):
        W = theta[: self.obs_dim * self.act_dim].reshape(self.obs_dim, self.act_dim)
        b = theta[self.obs_dim * self.act_dim :]
        return W, b

    def act(self, obs, theta=None):
        theta = theta if theta is not None else self.theta
        W, b = self._unpack(theta)
        raw = obs @ W + b
        a = 1 / (1 + np.exp(-raw))  # sigmoid to [0,1]
        N = self.act_dim // 2
        p = self.p_range[0] + a[:N] * (self.p_range[1] - self.p_range[0])
        t = self.tau_range[0] + a[N:] * (self.tau_range[1] - self.tau_range[0])
        return np.concatenate([p, t]).astype(np.float32)


def tune_es_policy(env_factory, obs_dim, act_dim, p_range, tau_range, K,
                    generations=60, pop_size=24, elite_frac=0.25, n_eval_steps=300,
                    sigma=0.5, seed=0):
    rng = np.random.default_rng(seed)
    policy = LinearESPolicy(obs_dim, act_dim, p_range, tau_range)
    n_params = policy.n_params
    mean = np.zeros(n_params, dtype=np.float32)
    std = np.full(n_params, sigma, dtype=np.float32)
    n_elite = max(1, int(pop_size * elite_frac))

    def fitness(theta):
        env = env_factory()
        obs = env.reset(seed=seed)
        total = 0.0
        for _ in range(n_eval_steps):
            actions = [policy.act(obs[k], theta) for k in range(K)]
            obs, rewards, done, info = env.step(actions)
            total += float(np.sum(info["sps_util"]))
            if done:
                obs = env.reset()
        return total

    best_theta, best_fit = mean.copy(), -np.inf
    for gen in range(generations):
        pop = mean + std * rng.standard_normal((pop_size, n_params)).astype(np.float32)
        fits = np.array([fitness(pop[i]) for i in range(pop_size)])
        elite_idx = np.argsort(fits)[-n_elite:]
        elite = pop[elite_idx]
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) + 1e-3
        if fits.max() > best_fit:
            best_fit = fits.max()
            best_theta = pop[np.argmax(fits)].copy()
    policy.theta = best_theta
    return policy


class ContractTheoryPolicy:
    """Closed-form cost-recovery-plus-margin contract pricing (classical
    mechanism design, no learning). Higher-quality SPSs are offered a
    lower required margin (screening-contract intuition: more competitive
    types are induced to reveal type via a lower rent), time allocated
    proportional to available resources."""
    def __init__(self, N, op_cost, margin_base, p_range, tau_range):
        self.N, self.op_cost, self.margin_base = N, op_cost, margin_base
        self.p_range, self.tau_range = p_range, tau_range

    def act(self, obs, agent_idx):
        quality_norm = obs[: self.N]  # already normalized in [0,1]
        resources_norm = obs[-1]
        margin = self.margin_base / (1.0 + quality_norm)
        p = np.clip(self.op_cost[agent_idx] * (1.0 + margin), *self.p_range)
        t = np.clip(np.full(self.N, resources_norm * self.tau_range[1]), *self.tau_range)
        return np.concatenate([p, t]).astype(np.float32)
