"""Real-data-grounded generalization check: retrains and re-evaluates
MADDPG_repro and F_TAPS with the environment's attack-rate process replaced
by the empirical CIC-IDS-2017-derived trace (build_attack_trace.py) instead
of a synthetic bounded random walk, at the main K=4,N=3,budget=100
configuration, 10 test seeds. All other environment/agent settings are
unchanged from the main benchmark table. This tests whether the welfare
ranking established under the synthetic exogenous process also holds when
attack intensity follows a real, bursty, non-stationary trace.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

from code.experiments.run_benchmarks import (
    K_MAIN, TEST_SEEDS_FULL, FINAL_RHO, FINAL_ACTOR_LR, FINAL_CRITIC_LR, FULL_STEPS,
)

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRACE_PATH = os.path.join(BASE, "data", "real_attack_intensity_ddos.npy")
OUT_DIR = os.path.join(BASE, "results", "real_attack_generalization")
os.makedirs(OUT_DIR, exist_ok=True)


def make_real_trace_env(seed, rho):
    from code.env.ami_stackelberg_env import AMIStackelbergEnv, EnvConfig
    trace = np.load(TRACE_PATH)
    cfg = EnvConfig(seed=seed, max_steps=10**9, rho=rho, leader_mode="greedy",
                     attack_trace=trace, **K_MAIN)
    return AMIStackelbergEnv(cfg)


def run_one(method, seed, total_steps=FULL_STEPS):
    torch.set_num_threads(1)
    from code.agents.maddpg import MADDPG
    from code.agents.hybrid_marl import HybridMARL
    rho = FINAL_RHO if method == "F_TAPS" else 0.0
    env = make_real_trace_env(seed, rho)
    p_range, tau_range = (env.cfg.p_min, env.cfg.p_max), (env.cfg.tau_min, env.cfg.tau_max)
    K = env.K
    if method == "F_TAPS":
        agent = HybridMARL(K, env.obs_dim, env.act_dim, p_range, tau_range, backbone="pool",
                            twin_critic=False, actor_lr=FINAL_ACTOR_LR, critic_lr=FINAL_CRITIC_LR,
                            noise_decay_steps=total_steps, device="cpu", seed=seed)
    else:
        agent = MADDPG(K, env.obs_dim, env.act_dim, p_range, tau_range,
                        noise_decay_steps=total_steps, device="cpu", seed=seed)
    obs = env.reset(seed=seed)
    sps_hist, tpr_hist, welfare_hist = [], [], []
    t0 = time.time()
    for step in range(total_steps):
        actions = agent.act(obs, explore=True)
        next_obs, rewards, done, info = env.step(actions)
        agent.store(obs, actions, rewards, next_obs, done)
        agent.update()
        obs = next_obs if not done else env.reset()
        sps_hist.append(float(np.sum(info["sps_util"]))); tpr_hist.append(float(info["tpr_util"]))
        welfare_hist.append(float(info["welfare"]))
    elapsed = time.time() - t0
    n = max(1, int(total_steps * 0.1))
    return dict(method=method, seed=seed, sps_utility=float(np.mean(sps_hist[-n:])),
                tpr_utility=float(np.mean(tpr_hist[-n:])), welfare=float(np.mean(welfare_hist[-n:])),
                train_time_sec=elapsed)


def main():
    print("=== Real-attack-trace generalization: MADDPG_repro vs F_TAPS, CIC-IDS-2017-derived attack dynamics ===")
    results = []
    with ProcessPoolExecutor(max_workers=min(20, os.cpu_count())) as ex:
        futures = {}
        for method in ["MADDPG_repro", "F_TAPS"]:
            for seed in TEST_SEEDS_FULL:
                futures[ex.submit(run_one, method, seed)] = (method, seed)
        for fut in as_completed(futures):
            method, seed = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f"  done: {method} seed={seed} -> welfare={r['welfare']:.2f}")
            except Exception as e:
                print(f"  FAILED: {method} seed={seed}: {e}")
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, "real_attack_raw.csv"), index=False)
    agg = df.groupby("method").agg(
        mean_welfare=("welfare", "mean"), std_welfare=("welfare", "std"),
        mean_sps=("sps_utility", "mean"), mean_tpr=("tpr_utility", "mean"),
    ).reset_index()
    agg.to_csv(os.path.join(OUT_DIR, "real_attack_aggregated.csv"), index=False)
    print(agg.to_string(index=False))

    from scipy import stats
    pivot = df.pivot(index="seed", columns="method", values="welfare")
    w_stat, w_p = stats.wilcoxon(pivot["F_TAPS"], pivot["MADDPG_repro"])
    diff = pivot["F_TAPS"] - pivot["MADDPG_repro"]
    d = float(diff.mean() / (diff.std(ddof=1) + 1e-12))
    with open(os.path.join(OUT_DIR, "real_attack_significance.txt"), "w") as f:
        f.write(f"Wilcoxon signed-rank F_TAPS vs MADDPG_repro (real attack trace, n={len(pivot)}): "
                f"stat={w_stat}, p={w_p}, Cohen's d={d:.3f}\n")
    print(f"Wilcoxon p={w_p}, Cohen's d={d:.3f}")
    print("Real-attack-trace experiment complete. Results in", OUT_DIR)


if __name__ == "__main__":
    main()
