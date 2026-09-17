"""Tuning-parity companion to run_tuning.py: gives MATD3 and MASAC the same
kind of Optuna (TPE) hyperparameter search F_TAPS received (15 trials,
validation seeds 103-105, K=4,N=3,budget=100, maximizing mean steady-state
welfare), so the main benchmark comparison does not confound "better
architecture" with "more tuning budget" for the two most directly comparable
modern MARL baselines. I-PPO and the reproduced source MADDPG intentionally
keep the source paper's own specified hyperparameters (Table II/III) for
reproduction fidelity and are not tuned here; the rule-based/classical
baselines (Random/Fixed/Heuristic/Greedy/ContractTheory) have no
hyperparameters, and BO-static/ES are themselves optimization procedures.
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import optuna
import torch
from concurrent.futures import ProcessPoolExecutor

from code.env.ami_stackelberg_env import AMIStackelbergEnv, EnvConfig
from code.experiments.run_screening import K_CONFIGS

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "results", "screening", "tuning")
os.makedirs(OUT_DIR, exist_ok=True)
VALID_SEEDS = [103, 104, 105]
TUNE_STEPS = 4000


def build_untuned(name, K, obs_dim, act_dim, p_range, tau_range, seed, hp):
    from code.agents.matd3 import MATD3
    from code.agents.masac import MASAC
    if name == "MATD3":
        return MATD3(K, obs_dim, act_dim, p_range, tau_range, actor_lr=hp["actor_lr"],
                     critic_lr=hp["critic_lr"], policy_noise=hp.get("policy_noise", 0.1),
                     noise_decay_steps=TUNE_STEPS, device="cpu", seed=seed)
    if name == "MASAC":
        return MASAC(K, obs_dim, act_dim, p_range, tau_range, actor_lr=hp["actor_lr"],
                     critic_lr=hp["critic_lr"], alpha=hp.get("alpha", 0.01), device="cpu", seed=seed)
    raise ValueError(name)


def evaluate(name, seed, hp, total_steps=TUNE_STEPS):
    torch.set_num_threads(1)
    kcfg = K_CONFIGS[0]
    cfg = EnvConfig(seed=seed, max_steps=10**9, rho=0.0, leader_mode="greedy", **kcfg)
    env = AMIStackelbergEnv(cfg)
    p_range, tau_range = (env.cfg.p_min, env.cfg.p_max), (env.cfg.tau_min, env.cfg.tau_max)
    agent = build_untuned(name, env.K, env.obs_dim, env.act_dim, p_range, tau_range, seed, hp)
    obs = env.reset(seed=seed)
    sps_hist, tpr_hist = [], []
    for step in range(total_steps):
        actions = agent.act(obs, explore=True)
        next_obs, rewards, done, info = env.step(actions)
        agent.store(obs, actions, rewards, next_obs, done)
        agent.update()
        obs = next_obs if not done else env.reset()
        sps_hist.append(float(np.sum(info["sps_util"])))
        tpr_hist.append(float(info["tpr_util"]))
    n = max(1, int(total_steps * 0.1))
    return float(np.mean(sps_hist[-n:])) + float(np.mean(tpr_hist[-n:]))


def objective(trial, name, executor):
    if name == "MATD3":
        hp = {"actor_lr": trial.suggest_float("actor_lr", 1e-5, 1e-3, log=True),
              "critic_lr": trial.suggest_float("critic_lr", 1e-4, 5e-3, log=True),
              "policy_noise": trial.suggest_float("policy_noise", 0.02, 0.3)}
    else:  # MASAC
        hp = {"actor_lr": trial.suggest_float("actor_lr", 1e-5, 1e-3, log=True),
              "critic_lr": trial.suggest_float("critic_lr", 1e-4, 5e-3, log=True),
              "alpha": trial.suggest_float("alpha", 0.001, 0.2, log=True)}
    futs = [executor.submit(evaluate, name, seed, hp) for seed in VALID_SEEDS]
    scores = [f.result() for f in futs]
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, choices=["MATD3", "MASAC"])
    parser.add_argument("--n_trials", type=int, default=15)
    args = parser.parse_args()

    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    with ProcessPoolExecutor(max_workers=len(VALID_SEEDS)) as executor:
        study.optimize(lambda t: objective(t, args.candidate, executor), n_trials=args.n_trials)

    result = {"candidate": args.candidate, "best_params": study.best_params, "best_value": study.best_value}
    with open(os.path.join(OUT_DIR, f"{args.candidate}_best.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
