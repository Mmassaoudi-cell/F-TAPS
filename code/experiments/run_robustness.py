"""Robustness experiments (revised protocol). Each (method, seed) is trained
ONCE for FULL_STEPS=15,000 steps -- identical to the main benchmark table's
protocol, so the resulting policy is the same trained artifact reported in
Table I -- then the single resulting policy is evaluated under four
deterministic (explore=False), noise-free-of-training rollouts branched from
that one checkpoint: nominal, a 50% budget cut, Gaussian observation noise,
and a 50% true-quality drop for a random quarter of nodes. Branching all four
conditions from one shared trained policy removes cross-condition training
variance as a confound, and matching the training length/protocol to the
main table means the "nominal" condition here is the same trained policy
(not merely the same recipe) as the one behind each seed's Table I number.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

from code.experiments.run_benchmarks import (
    K_MAIN, TEST_SEEDS_SCALE, make_env, FINAL_RHO, FINAL_ACTOR_LR, FINAL_CRITIC_LR, FULL_STEPS,
)

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(BASE, "results", "robustness")
os.makedirs(OUT_DIR, exist_ok=True)

EVAL_STEPS = 2_000


def train_agent(method, seed, kcfg=K_MAIN, train_steps=FULL_STEPS):
    from code.agents.maddpg import MADDPG
    from code.agents.hybrid_marl import HybridMARL
    rho = FINAL_RHO if method == "F_TAPS" else 0.0
    env = make_env(kcfg, seed, rho=rho, leader_mode="greedy")
    p_range, tau_range = (env.cfg.p_min, env.cfg.p_max), (env.cfg.tau_min, env.cfg.tau_max)
    K = env.K
    if method == "F_TAPS":
        agent = HybridMARL(K, env.obs_dim, env.act_dim, p_range, tau_range, backbone="pool",
                            twin_critic=False, actor_lr=FINAL_ACTOR_LR, critic_lr=FINAL_CRITIC_LR,
                            noise_decay_steps=train_steps, device="cpu", seed=seed)
    else:
        agent = MADDPG(K, env.obs_dim, env.act_dim, p_range, tau_range,
                        noise_decay_steps=train_steps, device="cpu", seed=seed)
    obs = env.reset(seed=seed)
    for step in range(train_steps):
        actions = agent.act(obs, explore=True)
        next_obs, rewards, done, info = env.step(actions)
        agent.store(obs, actions, rewards, next_obs, done)
        agent.update()
        obs = next_obs if not done else env.reset()
    return agent, env


def eval_perturbed(agent, env_state_seed, method, perturbation, seed, eval_steps=EVAL_STEPS):
    """Branches a fresh evaluation environment from the SAME trained agent
    (agent weights untouched across conditions), reseeded identically so all
    four conditions start from the same exogenous-process draw, differing
    only in the perturbation applied."""
    rho = FINAL_RHO if method == "F_TAPS" else 0.0
    env = make_env(K_MAIN, env_state_seed, rho=rho, leader_mode="greedy")
    K = env.K
    rng = np.random.default_rng(seed + 777)
    if perturbation == "budget_shock":
        env.cfg.budget = env.cfg.budget * 0.5
    if perturbation == "quality_shock":
        n_affected = max(1, K // 4)
        affected = rng.choice(K, size=n_affected, replace=False)
        env.quality[affected] *= 0.5

    obs = env._get_obs()
    sps_hist, tpr_hist, welfare_hist = [], [], []
    for step in range(eval_steps):
        if perturbation == "obs_noise":
            noisy_obs = [o + rng.normal(0, 0.1, size=o.shape).astype(np.float32) for o in obs]
            actions = agent.act(noisy_obs, explore=False)
        else:
            actions = agent.act(obs, explore=False)
        next_obs, rewards, done, info = env.step(actions)
        obs = next_obs if not done else env.reset()
        sps_hist.append(float(np.sum(info["sps_util"]))); tpr_hist.append(float(info["tpr_util"]))
        welfare_hist.append(float(info["welfare"]))
    return dict(sps_utility=float(np.mean(sps_hist)), tpr_utility=float(np.mean(tpr_hist)),
                welfare=float(np.mean(welfare_hist)))


def action_variance_diagnostic(agent, method, seed, n_samples=500):
    """Quantifies whether the trained policy's action meaningfully depends on
    its observation, or has converged to a near-constant output. Compares
    (a) the actual action std across n_samples diverse observations drawn
    from the environment's own state distribution, against (b) a synthetic
    sweep of each observation feature from its minimum to maximum (holding
    others at their mean), reporting the resulting action range as a
    fraction of the full action range -- the direct test the Devil's
    Advocate review requested."""
    env = make_env(K_MAIN, seed, rho=FINAL_RHO if method == "F_TAPS" else 0.0, leader_mode="greedy")
    obs = env.reset(seed=seed)
    K, obs_dim, act_dim = env.K, env.obs_dim, env.act_dim
    p_range, tau_range = (env.cfg.p_min, env.cfg.p_max), (env.cfg.tau_min, env.cfg.tau_max)
    act_range = np.concatenate([np.full(act_dim // 2, p_range[1] - p_range[0]),
                                 np.full(act_dim // 2, tau_range[1] - tau_range[0])])

    collected_obs, collected_act = [], []
    for _ in range(n_samples):
        actions = agent.act(obs, explore=False)
        collected_obs.append(obs[0].copy())
        collected_act.append(actions[0].copy())
        next_obs, rewards, done, info = env.step(actions)
        obs = next_obs if not done else env.reset()
    collected_obs, collected_act = np.array(collected_obs), np.array(collected_act)
    action_std_over_rollout = collected_act.std(axis=0)
    action_std_frac_of_range = float(np.mean(action_std_over_rollout / act_range))

    sweep_ranges = []
    for feat_idx in range(obs_dim):
        base = collected_obs.mean(axis=0)
        sweep_vals = np.linspace(0.0, 1.0, 20)
        outs = []
        for v in sweep_vals:
            o = base.copy(); o[feat_idx] = v
            a = agent.act([o] * K, explore=False)[0]
            outs.append(a)
        outs = np.array(outs)
        sweep_range = (outs.max(axis=0) - outs.min(axis=0)) / act_range
        sweep_ranges.append(float(np.mean(sweep_range)))

    return dict(method=method, seed=seed,
                rollout_action_std_frac_of_range=action_std_frac_of_range,
                mean_feature_sweep_range_frac=float(np.mean(sweep_ranges)),
                max_feature_sweep_range_frac=float(np.max(sweep_ranges)))


def run_one(method, seed):
    torch.set_num_threads(1)
    agent, _ = train_agent(method, seed)
    results = []
    for perturbation in ["nominal", "budget_shock", "obs_noise", "quality_shock"]:
        r = eval_perturbed(agent, seed, method, perturbation, seed)
        r.update(method=method, seed=seed, perturbation=perturbation)
        results.append(r)
    diag = action_variance_diagnostic(agent, method, seed)
    return results, diag


def main():
    print("=== Robustness (revised protocol, shared checkpoint per seed) + action-variance diagnostic ===")
    with ProcessPoolExecutor(max_workers=min(20, os.cpu_count())) as ex:
        futures = {}
        for method in ["MADDPG_repro", "F_TAPS"]:
            for seed in TEST_SEEDS_SCALE:
                futures[ex.submit(run_one, method, seed)] = (method, seed)
        all_results, all_diag = [], []
        for fut in as_completed(futures):
            method, seed = futures[fut]
            try:
                rs, diag = fut.result()
                all_results.extend(rs)
                all_diag.append(diag)
                print(f"  done: {method} seed={seed} -> " +
                      ", ".join(f"{r['perturbation']}={r['welfare']:.1f}" for r in rs) +
                      f" | action_std_frac={diag['rollout_action_std_frac_of_range']:.4f}"
                      f" sweep_frac={diag['mean_feature_sweep_range_frac']:.4f}")
            except Exception as e:
                print(f"  FAILED: {method} seed={seed}: {e}")

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(OUT_DIR, "robustness_raw.csv"), index=False)
    nominal = df[df.perturbation == "nominal"].set_index(["method", "seed"])["welfare"]
    df["nominal_welfare"] = df.apply(lambda row: nominal.loc[(row["method"], row["seed"])], axis=1)
    df["pct_degradation"] = 100 * (df["nominal_welfare"] - df["welfare"]) / df["nominal_welfare"].abs()
    df.to_csv(os.path.join(OUT_DIR, "robustness_with_degradation.csv"), index=False)
    agg = df.groupby(["method", "perturbation"]).agg(
        mean_welfare=("welfare", "mean"), std_welfare=("welfare", "std"),
        mean_pct_degradation=("pct_degradation", "mean"),
    ).reset_index()
    agg.to_csv(os.path.join(OUT_DIR, "robustness_aggregated.csv"), index=False)
    print(agg.to_string(index=False))

    diag_df = pd.DataFrame(all_diag)
    diag_df.to_csv(os.path.join(OUT_DIR, "action_variance_diagnostic_raw.csv"), index=False)
    diag_agg = diag_df.groupby("method").agg(
        mean_rollout_action_std_frac=("rollout_action_std_frac_of_range", "mean"),
        mean_feature_sweep_range_frac=("mean_feature_sweep_range_frac", "mean"),
        max_feature_sweep_range_frac=("max_feature_sweep_range_frac", "mean"),
    ).reset_index()
    diag_agg.to_csv(os.path.join(OUT_DIR, "action_variance_diagnostic_aggregated.csv"), index=False)
    print(diag_agg.to_string(index=False))
    print("Robustness + diagnostic complete. Results in", OUT_DIR)


if __name__ == "__main__":
    main()
