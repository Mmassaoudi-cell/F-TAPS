"""Step 17: ablation study. F_TAPS = {parameter-shared pool backbone} x
{threat-weighted objective, rho=0.9936}. This is a full 2x2 factorial over
those two components, holding all other hyperparameters at the FROZEN
FINAL_MODEL_CONFIG.yaml values (actor_lr, critic_lr) constant across all four
cells so the only things that vary are the two ablated components:

  architecture=independent (MADDPG), rho=0        -> MADDPG_repro   (reused from main_table_raw.csv)
  architecture=independent (MADDPG), rho=0.9936   -> NEW: "NoShare_ThreatWeighted"
  architecture=shared (pool),         rho=0        -> NEW: "Share_NoThreat"
  architecture=shared (pool),         rho=0.9936   -> F_TAPS         (reused from main_table_raw.csv)

Run on the same 10 TEST seeds, K=4,N=3,budget=100, 15,000 steps, for direct
comparability with the main benchmark table.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

from code.experiments.run_benchmarks import (
    OUT_DIR as BENCH_DIR, K_MAIN, TEST_SEEDS_FULL, FULL_STEPS, make_env,
    FINAL_RHO, FINAL_ACTOR_LR, FINAL_CRITIC_LR,
)

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(BASE, "results", "ablation")
os.makedirs(OUT_DIR, exist_ok=True)


def run_cell(architecture, rho, seed, kcfg=K_MAIN, total_steps=FULL_STEPS):
    torch.set_num_threads(1)
    from code.agents.maddpg import MADDPG
    from code.agents.hybrid_marl import HybridMARL

    env = make_env(kcfg, seed, rho=rho, leader_mode="greedy")
    p_range, tau_range = (env.cfg.p_min, env.cfg.p_max), (env.cfg.tau_min, env.cfg.tau_max)
    K = env.K

    if architecture == "independent":
        agent = MADDPG(K, env.obs_dim, env.act_dim, p_range, tau_range,
                        actor_lr=FINAL_ACTOR_LR, critic_lr=FINAL_CRITIC_LR,
                        noise_decay_steps=total_steps, device="cpu", seed=seed)
    else:
        agent = HybridMARL(K, env.obs_dim, env.act_dim, p_range, tau_range,
                            backbone="pool", twin_critic=False, actor_lr=FINAL_ACTOR_LR,
                            critic_lr=FINAL_CRITIC_LR, noise_decay_steps=total_steps,
                            device="cpu", seed=seed)

    obs = env.reset(seed=seed)
    sps_hist, tpr_hist, welfare_hist, per_agent = [], [], [], []
    t0 = time.time()
    for step in range(total_steps):
        actions = agent.act(obs, explore=True)
        next_obs, rewards, done, info = env.step(actions)
        agent.store(obs, actions, rewards, next_obs, done)
        agent.update()
        obs = next_obs if not done else env.reset()
        sps_hist.append(float(np.sum(info["sps_util"]))); tpr_hist.append(float(info["tpr_util"]))
        welfare_hist.append(float(info["welfare"])); per_agent.append(info["sps_util"].copy())
    elapsed = time.time() - t0
    n = max(1, int(total_steps * 0.1))
    mean_pa = np.array(per_agent[-n:]).mean(axis=0)
    jain = float((mean_pa.sum() ** 2) / (K * np.sum(mean_pa ** 2) + 1e-8))
    cell_name = f"{'Share' if architecture=='shared' else 'NoShare'}_{'ThreatWeighted' if rho>0 else 'NoThreat'}"
    return dict(cell=cell_name, architecture=architecture, rho=rho, seed=seed,
                sps_utility=float(np.mean(sps_hist[-n:])), tpr_utility=float(np.mean(tpr_hist[-n:])),
                welfare=float(np.mean(welfare_hist[-n:])), jain_fairness=jain, train_time_sec=elapsed)


def main():
    print("=== Ablation: 2 new cells x 10 TEST seeds (other 2 cells reused from main_table_raw.csv) ===")
    new_cells = [("independent", FINAL_RHO), ("shared", 0.0)]
    results = []
    with ProcessPoolExecutor(max_workers=min(20, os.cpu_count())) as ex:
        futures = {}
        for architecture, rho in new_cells:
            for seed in TEST_SEEDS_FULL:
                futures[ex.submit(run_cell, architecture, rho, seed)] = (architecture, rho, seed)
        for fut in as_completed(futures):
            architecture, rho, seed = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f"  done: {r['cell']} seed={seed} -> welfare={r['welfare']:.2f}")
            except Exception as e:
                print(f"  FAILED: {architecture} rho={rho} seed={seed}: {e}")
    df_new = pd.DataFrame(results)
    df_new.to_csv(os.path.join(OUT_DIR, "ablation_new_cells_raw.csv"), index=False)

    # pull the other 2 cells from the already-computed main benchmark table
    main_raw = pd.read_csv(os.path.join(BENCH_DIR, "main_table_raw.csv"))
    maddpg_rows = main_raw[main_raw.method == "MADDPG_repro"].copy()
    maddpg_rows["cell"] = "NoShare_NoThreat"
    ftaps_rows = main_raw[main_raw.method == "F_TAPS"].copy()
    ftaps_rows["cell"] = "Share_ThreatWeighted"
    reused = pd.concat([maddpg_rows, ftaps_rows])[["cell", "seed", "sps_utility", "tpr_utility", "welfare",
                                                      "jain_fairness", "train_time_sec"]]

    df_all = pd.concat([df_new[["cell", "seed", "sps_utility", "tpr_utility", "welfare",
                                  "jain_fairness", "train_time_sec"]], reused], ignore_index=True)
    df_all.to_csv(os.path.join(OUT_DIR, "ablation_full_2x2.csv"), index=False)

    agg = df_all.groupby("cell").agg(
        mean_welfare=("welfare", "mean"), std_welfare=("welfare", "std"),
        mean_sps=("sps_utility", "mean"), mean_tpr=("tpr_utility", "mean"),
        mean_jain=("jain_fairness", "mean"),
    ).reset_index().sort_values("mean_welfare", ascending=False)
    agg.to_csv(os.path.join(OUT_DIR, "ablation_2x2_aggregated.csv"), index=False)
    print(agg.to_string(index=False))
    print("Ablation complete. Results in", OUT_DIR)


if __name__ == "__main__":
    main()
