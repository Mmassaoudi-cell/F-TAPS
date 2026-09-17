"""Reproduce the source paper's core convergence study (Fig. 7-10) and the
6-method steady-state comparison (Fig. 6), config K=4, N=3, budget=100.

Outputs (results/reproduction/):
  maddpg_learning_curve.csv   - per-step joint SPS/TPR utility, MADDPG
  ippo_learning_curve.csv     - per-step joint SPS/TPR utility, I-PPO
  steady_state_comparison.csv - 6-method steady-state utilities (raw + normalized)
  actor_critic_losses_K{4,10}.csv
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)

from code.env.ami_stackelberg_env import AMIStackelbergEnv, EnvConfig
from code.agents.maddpg import MADDPG
from code.agents.ippo import IPPOAgent
from code.agents.baselines import RandomPolicy, FixedPolicy, HeuristicPolicy, GreedyPolicy

SEED = 42
TOTAL_STEPS = 50_000
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "results", "reproduction")
os.makedirs(OUT_DIR, exist_ok=True)


def make_env(K=4, N=3, budget=100.0, seed=SEED, continuing=True):
    cfg = EnvConfig(K=K, N=N, budget=budget, seed=seed,
                     max_steps=10**9 if continuing else 500)
    return AMIStackelbergEnv(cfg)


def run_maddpg(K=4, N=3, budget=100.0, total_steps=TOTAL_STEPS, seed=SEED, log_every=1,
               log_losses=False):
    np.random.seed(seed)
    env = make_env(K, N, budget, seed=seed)
    p_range, tau_range = (env.cfg.p_min, env.cfg.p_max), (env.cfg.tau_min, env.cfg.tau_max)
    agent = MADDPG(K, env.obs_dim, env.act_dim, p_range, tau_range,
                    noise_decay_steps=total_steps, device="cpu", seed=seed)
    obs = env.reset(seed=seed)
    sps_hist, tpr_hist = [], []
    actor_loss_hist, critic_loss_hist = [], []
    t0 = time.time()
    for step in range(total_steps):
        actions = agent.act(obs, explore=True)
        next_obs, rewards, done, info = env.step(actions)
        agent.store(obs, actions, rewards, next_obs, done)
        losses = agent.update()
        obs = next_obs if not done else env.reset()
        sps_hist.append(float(np.sum(info["sps_util"])))
        tpr_hist.append(float(info["tpr_util"]))
        if log_losses and losses is not None:
            actor_loss_hist.append(losses["actor"])
            critic_loss_hist.append(losses["critic"])
    print(f"[MADDPG K={K},N={N},B={budget}] {total_steps} steps in {time.time()-t0:.1f}s")
    return (np.array(sps_hist), np.array(tpr_hist),
            np.array(actor_loss_hist) if log_losses else None,
            np.array(critic_loss_hist) if log_losses else None)


def run_ippo(K=4, N=3, budget=100.0, total_steps=TOTAL_STEPS, seed=SEED, rollout_len=128):
    np.random.seed(seed)
    env = make_env(K, N, budget, seed=seed)
    p_range, tau_range = (env.cfg.p_min, env.cfg.p_max), (env.cfg.tau_min, env.cfg.tau_max)
    agents = [IPPOAgent(env.obs_dim, env.act_dim, p_range, tau_range, device="cpu")
              for _ in range(K)]
    obs = env.reset(seed=seed)
    sps_hist, tpr_hist = [], []
    t0 = time.time()
    step = 0
    while step < total_steps:
        for _ in range(rollout_len):
            if step >= total_steps:
                break
            acts, logps, vals = [], [], []
            for k in range(K):
                a, lp, v = agents[k].act(obs[k])
                acts.append(a); logps.append(lp); vals.append(v)
            next_obs, rewards, done, info = env.step(acts)
            for k in range(K):
                agents[k].store(obs[k], acts[k], logps[k], vals[k], float(rewards[k]), float(done))
            obs = next_obs if not done else env.reset()
            sps_hist.append(float(np.sum(info["sps_util"])))
            tpr_hist.append(float(info["tpr_util"]))
            step += 1
        last_vals = []
        for k in range(K):
            with torch.no_grad():
                o = torch.as_tensor(obs[k], dtype=torch.float32).unsqueeze(0)
                last_vals.append(agents[k].critic(o).item())
        for k in range(K):
            agents[k].finish_rollout(last_vals[k])
    print(f"[I-PPO K={K},N={N},B={budget}] {total_steps} steps in {time.time()-t0:.1f}s")
    return np.array(sps_hist), np.array(tpr_hist)


def rollout_rulebased(policy_name, K=4, N=3, budget=100.0, total_steps=TOTAL_STEPS, seed=SEED):
    np.random.seed(seed)
    env = make_env(K, N, budget, seed=seed)
    p_range, tau_range = (env.cfg.p_min, env.cfg.p_max), (env.cfg.tau_min, env.cfg.tau_max)
    rng = np.random.default_rng(seed)
    if policy_name == "random":
        pols = [RandomPolicy(N, p_range, tau_range, rng) for _ in range(K)]
    elif policy_name == "fixed":
        pols = [FixedPolicy(N, p_range, tau_range) for _ in range(K)]
    elif policy_name == "heuristic":
        pols = [HeuristicPolicy(N, p_range, tau_range) for _ in range(K)]
    elif policy_name == "greedy":
        pols = [GreedyPolicy(k, env, N, p_range, tau_range, rng=rng) for k in range(K)]
    else:
        raise ValueError(policy_name)

    obs = env.reset(seed=seed)
    prev_actions = [np.concatenate([np.full(N, (p_range[0]+p_range[1])/2),
                                     np.full(N, (tau_range[0]+tau_range[1])/2)]).astype(np.float32)
                     for _ in range(K)]
    sps_hist, tpr_hist = [], []
    for step in range(total_steps):
        if policy_name == "greedy":
            actions = [pols[k].act(obs[k], prev_actions) for k in range(K)]
        else:
            actions = [pols[k].act(obs[k]) for k in range(K)]
        next_obs, rewards, done, info = env.step(actions)
        prev_actions = actions
        obs = next_obs if not done else env.reset()
        sps_hist.append(float(np.sum(info["sps_util"])))
        tpr_hist.append(float(info["tpr_util"]))
    return np.array(sps_hist), np.array(tpr_hist)


def steady_state(x, frac=0.1):
    n = max(1, int(len(x) * frac))
    return float(np.mean(x[-n:]))


def main():
    K, N, budget = 4, 3, 100.0

    print("=== Running MADDPG (with loss logging) ===")
    maddpg_sps, maddpg_tpr, actor_losses, critic_losses = run_maddpg(K, N, budget, log_losses=True)
    pd.DataFrame({"step": np.arange(len(maddpg_sps)),
                  "sps_utility": maddpg_sps, "tpr_utility": maddpg_tpr}
                 ).to_csv(os.path.join(OUT_DIR, "maddpg_learning_curve.csv"), index=False)
    if actor_losses is not None and len(actor_losses) > 0:
        cols = {f"sps{k}_actor_loss": actor_losses[:, k] for k in range(K)}
        cols.update({f"sps{k}_critic_loss": critic_losses[:, k] for k in range(K)})
        pd.DataFrame(cols).to_csv(os.path.join(OUT_DIR, f"actor_critic_losses_K{K}.csv"), index=False)

    print("=== Running I-PPO ===")
    ippo_sps, ippo_tpr = run_ippo(K, N, budget)
    pd.DataFrame({"step": np.arange(len(ippo_sps)),
                  "sps_utility": ippo_sps, "tpr_utility": ippo_tpr}
                 ).to_csv(os.path.join(OUT_DIR, "ippo_learning_curve.csv"), index=False)

    print("=== Running rule-based baselines ===")
    results = {}
    for name in ["random", "fixed", "heuristic", "greedy"]:
        sps, tpr = rollout_rulebased(name, K, N, budget, total_steps=5000)
        results[name] = (steady_state(sps), steady_state(tpr))
        print(f"  {name}: SPS={results[name][0]:.2f}  TPR={results[name][1]:.2f}")

    results["maddpg"] = (steady_state(maddpg_sps), steady_state(maddpg_tpr))
    results["ippo"] = (steady_state(ippo_sps), steady_state(ippo_tpr))

    maddpg_sps_ss, maddpg_tpr_ss = results["maddpg"]
    rows = []
    for name, (sps_ss, tpr_ss) in results.items():
        rows.append({
            "method": name,
            "steady_state_sps_utility": sps_ss,
            "steady_state_tpr_utility": tpr_ss,
            "normalized_sps_utility": sps_ss / maddpg_sps_ss if maddpg_sps_ss != 0 else np.nan,
            "normalized_tpr_utility": tpr_ss / maddpg_tpr_ss if maddpg_tpr_ss != 0 else np.nan,
        })
    df = pd.DataFrame(rows).sort_values("normalized_sps_utility", ascending=False)
    df.to_csv(os.path.join(OUT_DIR, "steady_state_comparison.csv"), index=False)
    print(df.to_string(index=False))

    print("=== Running MADDPG at K=10 for loss-curve comparison ===")
    _, _, al10, cl10 = run_maddpg(K=10, N=3, budget=200.0, total_steps=20_000,
                                    seed=SEED, log_losses=True)
    if al10 is not None and len(al10) > 0:
        cols = {f"sps{k}_actor_loss": al10[:, k] for k in range(10)}
        cols.update({f"sps{k}_critic_loss": cl10[:, k] for k in range(10)})
        pd.DataFrame(cols).to_csv(os.path.join(OUT_DIR, "actor_critic_losses_K10.csv"), index=False)

    print("Done. Results in", OUT_DIR)


if __name__ == "__main__":
    main()
