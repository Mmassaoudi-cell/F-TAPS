"""Validation-only candidate screening (Stage 1 smoke test + Stage 2
multi-seed screening) for Candidates A-E vs. the SOURCE_METHOD_REPRODUCTION
(MADDPG) baseline. Uses ONLY validation seeds (100-109), per
DATA_SPLIT_MANIFEST.csv -- test seeds (1000+) are never touched here.

Runs all (candidate, K-config, seed) combinations in parallel worker
processes since each run is independent and CPU-bound on small networks.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "results", "screening")
os.makedirs(OUT_DIR, exist_ok=True)

VALID_SEEDS_STAGE2 = [100, 101, 102]
K_CONFIGS = [dict(K=4, N=3, budget=100.0), dict(K=10, N=4, budget=200.0)]
STAGE2_STEPS = 10_000
STAGE1_STEPS = 1_500


def build_agent(name, env, seed):
    from code.agents.maddpg import MADDPG
    from code.agents.hybrid_marl import HybridMARL
    p_range, tau_range = (env.cfg.p_min, env.cfg.p_max), (env.cfg.tau_min, env.cfg.tau_max)
    K = env.K
    common = dict(noise_decay_steps=STAGE2_STEPS, device="cpu", seed=seed)
    if name in ("MADDPG_repro", "A_ThreatWeighted", "B_ExactLeader"):
        return MADDPG(K, env.obs_dim, env.act_dim, p_range, tau_range, **common)
    if name == "C_PIShare":
        return HybridMARL(K, env.obs_dim, env.act_dim, p_range, tau_range,
                           backbone="pool", twin_critic=False, **common)
    if name == "D_ATAC":
        return HybridMARL(K, env.obs_dim, env.act_dim, p_range, tau_range,
                           backbone="attention", twin_critic=True, **common)
    if name == "E_MFTAC":
        return HybridMARL(K, env.obs_dim, env.act_dim, p_range, tau_range,
                           backbone="meanfield", twin_critic=True, **common)
    if name == "F_TAPS":
        return HybridMARL(K, env.obs_dim, env.act_dim, p_range, tau_range,
                           backbone="pool", twin_critic=False, **common)
    raise ValueError(name)


def env_overrides(name):
    """(rho, leader_mode) per candidate, per MODEL_CANDIDATES.md."""
    return {
        "MADDPG_repro": (0.0, "greedy"),
        "A_ThreatWeighted": (0.5, "greedy"),
        "B_ExactLeader": (0.0, "exact"),
        "C_PIShare": (0.0, "greedy"),
        "D_ATAC": (0.5, "exact"),
        "E_MFTAC": (0.5, "exact"),
        "F_TAPS": (0.5, "greedy"),  # post-hoc candidate: A's threat-weighting + C's pooled shared backbone
    }[name]


def run_one(name, kcfg, seed, total_steps):
    torch.set_num_threads(1)
    from code.env.ami_stackelberg_env import AMIStackelbergEnv, EnvConfig
    rho, leader_mode = env_overrides(name)
    cfg = EnvConfig(seed=seed, max_steps=10**9, rho=rho, leader_mode=leader_mode, **kcfg)
    env = AMIStackelbergEnv(cfg)
    agent = build_agent(name, env, seed)
    obs = env.reset(seed=seed)
    t0 = time.time()
    sps_hist, tpr_hist, welfare_hist = [], [], []
    per_agent_util_hist = []
    for step in range(total_steps):
        actions = agent.act(obs, explore=True)
        next_obs, rewards, done, info = env.step(actions)
        agent.store(obs, actions, rewards, next_obs, done)
        agent.update()
        obs = next_obs if not done else env.reset()
        sps_hist.append(float(np.sum(info["sps_util"])))
        tpr_hist.append(float(info["tpr_util"]))
        welfare_hist.append(float(info["welfare"]))
        per_agent_util_hist.append(info["sps_util"].copy())
    elapsed = time.time() - t0

    n = max(1, int(total_steps * 0.1))
    ss_sps = float(np.mean(sps_hist[-n:]))
    ss_tpr = float(np.mean(tpr_hist[-n:]))
    ss_welfare = float(np.mean(welfare_hist[-n:]))
    last_util = np.array(per_agent_util_hist[-n:])  # (n, K)
    mean_per_agent = last_util.mean(axis=0)
    jain = (mean_per_agent.sum() ** 2) / (env.K * np.sum(mean_per_agent ** 2) + 1e-8)
    collapsed = bool(ss_sps <= 1e-6 and ss_tpr <= 1e-6)

    n_params = 0
    for attr in ("actors", "critics", "critics1", "critics2"):
        if hasattr(agent, attr):
            for m in getattr(agent, attr):
                n_params += sum(p.numel() for p in m.parameters())
    if hasattr(agent, "actor"):
        n_params += sum(p.numel() for p in agent.actor.parameters())

    return {
        "candidate": name, "K": kcfg["K"], "N": kcfg["N"], "budget": kcfg["budget"], "seed": seed,
        "steady_state_sps_utility": ss_sps, "steady_state_tpr_utility": ss_tpr,
        "steady_state_welfare": ss_welfare, "jain_fairness_index": float(jain),
        "collapsed": collapsed, "train_time_sec": elapsed, "n_params": n_params,
        "total_steps": total_steps,
    }


def stage1_smoke():
    print("=== Stage 1: smoke test (1 seed, reduced budget) ===")
    results = []
    for name in ["MADDPG_repro", "A_ThreatWeighted", "B_ExactLeader",
                 "C_PIShare", "D_ATAC", "E_MFTAC"]:
        r = run_one(name, K_CONFIGS[0], seed=1, total_steps=STAGE1_STEPS)
        status = "COLLAPSED" if r["collapsed"] else "ok"
        print(f"  {name}: sps={r['steady_state_sps_utility']:.2f} tpr={r['steady_state_tpr_utility']:.2f} "
              f"welfare={r['steady_state_welfare']:.2f} [{status}] ({r['train_time_sec']:.1f}s)")
        results.append(r)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, "stage1_smoke_test.csv"), index=False)
    survivors = df[~df["collapsed"]]["candidate"].tolist()
    print("Survivors:", survivors)
    return survivors


def stage2_screening(survivors):
    print("=== Stage 2: validation screening (3 seeds x 2 K-configs) ===")
    jobs = []
    with ProcessPoolExecutor(max_workers=min(20, os.cpu_count())) as ex:
        futures = {}
        for name in survivors:
            for kcfg in K_CONFIGS:
                for seed in VALID_SEEDS_STAGE2:
                    fut = ex.submit(run_one, name, kcfg, seed, STAGE2_STEPS)
                    futures[fut] = (name, kcfg, seed)
        results = []
        for fut in as_completed(futures):
            name, kcfg, seed = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f"  done: {name} K={kcfg['K']} seed={seed} -> "
                      f"sps={r['steady_state_sps_utility']:.2f} tpr={r['steady_state_tpr_utility']:.2f}")
            except Exception as e:
                print(f"  FAILED: {name} K={kcfg['K']} seed={seed}: {e}")
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, "stage2_validation_screening.csv"), index=False)

    agg = df.groupby(["candidate", "K"]).agg(
        mean_sps=("steady_state_sps_utility", "mean"), std_sps=("steady_state_sps_utility", "std"),
        mean_tpr=("steady_state_tpr_utility", "mean"), std_tpr=("steady_state_tpr_utility", "std"),
        mean_welfare=("steady_state_welfare", "mean"),
        mean_jain=("jain_fairness_index", "mean"),
        any_collapsed=("collapsed", "any"),
        mean_train_time=("train_time_sec", "mean"),
        n_params=("n_params", "first"),
    ).reset_index()
    agg.to_csv(os.path.join(OUT_DIR, "stage2_aggregated.csv"), index=False)
    print(agg.to_string(index=False))
    return df, agg


if __name__ == "__main__":
    survivors = stage1_smoke()
    if len(survivors) == 0:
        print("ALL CANDIDATES COLLAPSED IN SMOKE TEST -- aborting Stage 2")
        sys.exit(1)
    stage2_screening(survivors)
    print("Screening complete. Results in", OUT_DIR)
