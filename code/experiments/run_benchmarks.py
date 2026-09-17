"""Step 14-15: full benchmark suite on held-out TEST seeds (1000-1009).
FIRST TIME test seeds are touched in this project -- FINAL_MODEL_CONFIG.yaml
is frozen before this script's first real run.

Main table: 12 methods x K=4,N=3,budget=100 (paper's headline config), 10 test seeds.
Scalability experiment: 3 methods x {K=4,10,15} x 5 test seeds (Rank-6 check).

Methods (>=10 benchmarks + final model, satisfies master-plan requirement):
  1. MADDPG (SOURCE_METHOD_REPRODUCTION)   7. MATD3
  2. Independent PPO (I-PPO)                8. MASAC
  3. Random policy                          9. Bayesian-optimized static mechanism
  4. Fixed policy                          10. Evolutionary-search (ES) linear policy
  5. Heuristic policy                      11. Contract-theory closed-form policy
  6. Greedy policy                         12. F_TAPS (FINAL proposed model)
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(BASE, "results", "benchmarks")
os.makedirs(OUT_DIR, exist_ok=True)

TEST_SEEDS_FULL = list(range(1000, 1010))   # 10 seeds, headline K=4 table
TEST_SEEDS_SCALE = list(range(1000, 1005))  # 5 seeds, scalability experiment
K_MAIN = dict(K=4, N=3, budget=100.0)
K_SCALE = [dict(K=4, N=3, budget=100.0), dict(K=10, N=4, budget=200.0), dict(K=15, N=5, budget=300.0)]
FULL_STEPS = 15_000
ROLLOUT_STEPS = 5_000  # for non-learned methods (no training needed)

FINAL_RHO = 0.9935931423871028
FINAL_ACTOR_LR = 4.335184350704857e-05
FINAL_CRITIC_LR = 0.004557048882646388

TUNING_DIR = os.path.join(BASE, "results", "screening", "tuning")


def load_tuned(name):
    """Loads MATD3/MASAC's matched-budget Optuna tuning result (produced by
    run_tuning_baselines.py); falls back to untuned defaults if not yet run."""
    path = os.path.join(TUNING_DIR, f"{name}_best.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["best_params"]
    return None


def make_env(kcfg, seed, rho=FINAL_RHO, leader_mode="greedy"):
    from code.env.ami_stackelberg_env import AMIStackelbergEnv, EnvConfig
    cfg = EnvConfig(seed=seed, max_steps=10**9, rho=rho, leader_mode=leader_mode, **kcfg)
    return AMIStackelbergEnv(cfg)


def eval_rollout(env, act_fn, seed, total_steps):
    obs = env.reset(seed=seed)
    sps_hist, tpr_hist, welfare_hist, per_agent = [], [], [], []
    t0 = time.time()
    for step in range(total_steps):
        actions = act_fn(obs, step)
        next_obs, rewards, done, info = env.step(actions)
        obs = next_obs if not done else env.reset()
        sps_hist.append(float(np.sum(info["sps_util"])))
        tpr_hist.append(float(info["tpr_util"]))
        welfare_hist.append(float(info["welfare"]))
        per_agent.append(info["sps_util"].copy())
    elapsed = time.time() - t0
    n = max(1, int(total_steps * 0.1))
    mean_pa = np.array(per_agent[-n:]).mean(axis=0)
    jain = float((mean_pa.sum() ** 2) / (env.K * np.sum(mean_pa ** 2) + 1e-8))
    return dict(sps_utility=float(np.mean(sps_hist[-n:])), tpr_utility=float(np.mean(tpr_hist[-n:])),
                welfare=float(np.mean(welfare_hist[-n:])), jain_fairness=jain, elapsed_sec=elapsed)


def run_learned(name, kcfg, seed, total_steps=FULL_STEPS):
    torch.set_num_threads(1)
    from code.agents.maddpg import MADDPG
    from code.agents.ippo import IPPOAgent
    from code.agents.matd3 import MATD3
    from code.agents.masac import MASAC
    from code.agents.hybrid_marl import HybridMARL

    if name == "F_TAPS":
        env = make_env(kcfg, seed, rho=FINAL_RHO, leader_mode="greedy")
    else:
        env = make_env(kcfg, seed, rho=0.0, leader_mode="greedy")  # benchmarks trained under source-faithful objective

    p_range, tau_range = (env.cfg.p_min, env.cfg.p_max), (env.cfg.tau_min, env.cfg.tau_max)
    K = env.K
    n_params = 0

    if name == "MADDPG_repro":
        agent = MADDPG(K, env.obs_dim, env.act_dim, p_range, tau_range,
                        noise_decay_steps=total_steps, device="cpu", seed=seed)
        n_params = sum(sum(p.numel() for p in m.parameters()) for m in agent.actors + agent.critics)
    elif name == "MATD3":
        hp = load_tuned("MATD3") or {}
        agent = MATD3(K, env.obs_dim, env.act_dim, p_range, tau_range,
                       actor_lr=hp.get("actor_lr", 1e-4), critic_lr=hp.get("critic_lr", 1e-3),
                       policy_noise=hp.get("policy_noise", 0.1),
                       noise_decay_steps=total_steps, device="cpu", seed=seed)
        n_params = sum(sum(p.numel() for p in m.parameters())
                        for m in agent.actors + agent.critics1 + agent.critics2)
    elif name == "MASAC":
        hp = load_tuned("MASAC") or {}
        agent = MASAC(K, env.obs_dim, env.act_dim, p_range, tau_range,
                       actor_lr=hp.get("actor_lr", 1e-4), critic_lr=hp.get("critic_lr", 1e-3),
                       alpha=hp.get("alpha", 0.01), device="cpu", seed=seed)
        n_params = sum(sum(p.numel() for p in m.parameters())
                        for m in agent.actors + agent.critics1 + agent.critics2)
    elif name == "F_TAPS":
        agent = HybridMARL(K, env.obs_dim, env.act_dim, p_range, tau_range,
                            backbone="pool", twin_critic=False, actor_lr=FINAL_ACTOR_LR,
                            critic_lr=FINAL_CRITIC_LR, noise_decay_steps=total_steps,
                            device="cpu", seed=seed)
        n_params = sum(p.numel() for p in agent.actor.parameters())
        n_params += sum(sum(p.numel() for p in c.parameters()) for c in agent.critics)
    elif name == "IPPO":
        agents = [IPPOAgent(env.obs_dim, env.act_dim, p_range, tau_range, device="cpu") for _ in range(K)]
        n_params = sum(sum(p.numel() for p in a.actor.parameters()) + sum(p.numel() for p in a.critic.parameters())
                        for a in agents)
    else:
        raise ValueError(name)

    obs = env.reset(seed=seed)
    sps_hist, tpr_hist, welfare_hist, per_agent = [], [], [], []
    t0 = time.time()

    if name == "IPPO":
        rollout_len = 128
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
                sps_hist.append(float(np.sum(info["sps_util"]))); tpr_hist.append(float(info["tpr_util"]))
                welfare_hist.append(float(info["welfare"])); per_agent.append(info["sps_util"].copy())
                step += 1
            last_vals = []
            for k in range(K):
                with torch.no_grad():
                    o = torch.as_tensor(obs[k], dtype=torch.float32).unsqueeze(0)
                    last_vals.append(agents[k].critic(o).item())
            for k in range(K):
                agents[k].finish_rollout(last_vals[k])
    else:
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
    return dict(method=name, K=kcfg["K"], N=kcfg["N"], budget=kcfg["budget"], seed=seed,
                sps_utility=float(np.mean(sps_hist[-n:])), tpr_utility=float(np.mean(tpr_hist[-n:])),
                welfare=float(np.mean(welfare_hist[-n:])), jain_fairness=jain,
                train_time_sec=elapsed, n_params=n_params, total_steps=total_steps)


def run_classical(name, kcfg, seed, total_steps=ROLLOUT_STEPS):
    torch.set_num_threads(1)
    from code.agents.baselines import RandomPolicy, FixedPolicy, HeuristicPolicy, GreedyPolicy
    from code.agents.classical_baselines import ContractTheoryPolicy, tune_bo_static, tune_es_policy

    env = make_env(kcfg, seed, rho=FINAL_RHO, leader_mode="greedy")
    p_range, tau_range = (env.cfg.p_min, env.cfg.p_max), (env.cfg.tau_min, env.cfg.tau_max)
    K, N = env.K, env.N
    rng = np.random.default_rng(seed)
    pol = None  # set below for methods with a single shared policy object exposing n_params (e.g., ESPolicy)

    if name == "Random":
        pols = [RandomPolicy(N, p_range, tau_range, rng) for _ in range(K)]
        act_fn = lambda obs, t: [pols[k].act(obs[k]) for k in range(K)]
    elif name == "Fixed":
        pols = [FixedPolicy(N, p_range, tau_range) for _ in range(K)]
        act_fn = lambda obs, t: [pols[k].act(obs[k]) for k in range(K)]
    elif name == "Heuristic":
        pols = [HeuristicPolicy(N, p_range, tau_range) for _ in range(K)]
        act_fn = lambda obs, t: [pols[k].act(obs[k]) for k in range(K)]
    elif name == "Greedy":
        pols = [GreedyPolicy(k, env, N, p_range, tau_range, rng=rng) for k in range(K)]
        state = {"prev": [np.concatenate([np.full(N, (p_range[0]+p_range[1])/2),
                                           np.full(N, (tau_range[0]+tau_range[1])/2)]).astype(np.float32)
                           for _ in range(K)]}
        def act_fn(obs, t):
            acts = [pols[k].act(obs[k], state["prev"]) for k in range(K)]
            state["prev"] = acts
            return acts
    elif name == "ContractTheory":
        pol = ContractTheoryPolicy(N, env.op_cost, margin_base=0.5, p_range=p_range, tau_range=tau_range)
        act_fn = lambda obs, t: [pol.act(obs[k], k) for k in range(K)]
    elif name == "BOStatic":
        env_factory = lambda: make_env(kcfg, 100, rho=FINAL_RHO, leader_mode="greedy")  # tuned on validation seed 100
        pol = tune_bo_static(env_factory, N, p_range, tau_range, n_trials=25, n_eval_steps=300, seed=100)
        act_fn = lambda obs, t: [pol.act(obs[k]) for k in range(K)]
    elif name == "ESPolicy":
        env_factory = lambda: make_env(kcfg, 100, rho=FINAL_RHO, leader_mode="greedy")
        pol = tune_es_policy(env_factory, env.obs_dim, env.act_dim, p_range, tau_range, K,
                              generations=40, pop_size=16, n_eval_steps=150, seed=100)
        act_fn = lambda obs, t: [pol.act(obs[k]) for k in range(K)]
    else:
        raise ValueError(name)

    n_params = getattr(pol, "n_params", 0)  # BOStatic/Random/Fixed/Heuristic/Greedy/ContractTheory: 0 (no
    # trainable parameters); ESPolicy: its true (small) evolved-linear-policy parameter count.
    r = eval_rollout(env, act_fn, seed, total_steps)
    return dict(method=name, K=kcfg["K"], N=kcfg["N"], budget=kcfg["budget"], seed=seed,
                sps_utility=r["sps_utility"], tpr_utility=r["tpr_utility"], welfare=r["welfare"],
                jain_fairness=r["jain_fairness"], train_time_sec=r["elapsed_sec"], n_params=n_params,
                total_steps=total_steps)


LEARNED_METHODS = ["MADDPG_repro", "IPPO", "MATD3", "MASAC", "F_TAPS"]
CLASSICAL_METHODS = ["Random", "Fixed", "Heuristic", "Greedy", "ContractTheory", "BOStatic", "ESPolicy"]


def run_one(method, kcfg, seed, total_steps=None):
    if method in LEARNED_METHODS:
        return run_learned(method, kcfg, seed, total_steps or FULL_STEPS)
    return run_classical(method, kcfg, seed, total_steps or ROLLOUT_STEPS)


def main_table():
    print("=== Main benchmark table: 12 methods @ K=4,N=3,budget=100, 10 TEST seeds ===")
    jobs = []
    with ProcessPoolExecutor(max_workers=min(22, os.cpu_count())) as ex:
        futures = {}
        for method in LEARNED_METHODS + CLASSICAL_METHODS:
            for seed in TEST_SEEDS_FULL:
                futures[ex.submit(run_one, method, K_MAIN, seed)] = (method, seed)
        results = []
        for fut in as_completed(futures):
            method, seed = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f"  done: {method} seed={seed} -> welfare={r['welfare']:.2f} sps={r['sps_utility']:.2f} tpr={r['tpr_utility']:.2f}")
            except Exception as e:
                print(f"  FAILED: {method} seed={seed}: {e}")
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, "main_table_raw.csv"), index=False)
    agg = df.groupby("method").agg(
        mean_welfare=("welfare", "mean"), std_welfare=("welfare", "std"),
        mean_sps=("sps_utility", "mean"), std_sps=("sps_utility", "std"),
        mean_tpr=("tpr_utility", "mean"), std_tpr=("tpr_utility", "std"),
        mean_jain=("jain_fairness", "mean"), mean_train_time=("train_time_sec", "mean"),
        n_params=("n_params", "first"),
    ).reset_index().sort_values("mean_welfare", ascending=False)
    agg.to_csv(os.path.join(OUT_DIR, "main_table_aggregated.csv"), index=False)
    print(agg.to_string(index=False))
    return df, agg


def scalability_experiment():
    print("=== Scalability experiment: MADDPG_repro vs BOStatic vs F_TAPS @ K in {4,10,15}, 5 TEST seeds ===")
    methods = ["MADDPG_repro", "BOStatic", "F_TAPS"]
    jobs = []
    with ProcessPoolExecutor(max_workers=min(22, os.cpu_count())) as ex:
        futures = {}
        for method in methods:
            for kcfg in K_SCALE:
                for seed in TEST_SEEDS_SCALE:
                    futures[ex.submit(run_one, method, kcfg, seed)] = (method, kcfg["K"], seed)
        results = []
        for fut in as_completed(futures):
            method, K, seed = futures[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f"  done: {method} K={K} seed={seed} -> welfare={r['welfare']:.2f} params={r['n_params']}")
            except Exception as e:
                print(f"  FAILED: {method} K={K} seed={seed}: {e}")
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT_DIR, "scalability_raw.csv"), index=False)
    agg = df.groupby(["method", "K"]).agg(
        mean_welfare=("welfare", "mean"), std_welfare=("welfare", "std"),
        mean_train_time=("train_time_sec", "mean"), n_params=("n_params", "first"),
    ).reset_index()
    agg.to_csv(os.path.join(OUT_DIR, "scalability_aggregated.csv"), index=False)
    print(agg.to_string(index=False))
    return df, agg


if __name__ == "__main__":
    main_table()
    scalability_experiment()
    print("Benchmarking complete. Results in", OUT_DIR)
