"""Stage 3: Optuna (TPE) hyperparameter tuning of the top 2-3 candidates
surviving Stage 2 screening, on VALIDATION seeds only (100-109).

Usage: python code/experiments/run_tuning.py --candidate D_ATAC
(candidate name is filled in after inspecting results/screening/stage2_aggregated.csv)
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import optuna
import torch
from concurrent.futures import ProcessPoolExecutor

from code.env.ami_stackelberg_env import AMIStackelbergEnv, EnvConfig
from code.experiments.run_screening import build_agent, env_overrides, K_CONFIGS

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "results", "screening", "tuning")
os.makedirs(OUT_DIR, exist_ok=True)
VALID_SEEDS = [103, 104, 105]  # distinct from the seeds used in Stage-2 candidate screening (100-102)
TUNE_STEPS = 4000


def evaluate(name, kcfg, seed, hp_overrides, total_steps=TUNE_STEPS):
    torch.set_num_threads(1)
    rho, leader_mode = env_overrides(name)
    rho = hp_overrides.get("rho", rho)
    cfg = EnvConfig(seed=seed, max_steps=10**9, rho=rho, leader_mode=leader_mode, **kcfg)
    env = AMIStackelbergEnv(cfg)
    agent = build_agent(name, env, seed)
    for pname, val in hp_overrides.items():
        if pname == "rho":
            continue
        if pname == "actor_lr":
            for opt in getattr(agent, "actor_opts", [getattr(agent, "actor_opt", None)]):
                if opt is not None:
                    for g in opt.param_groups:
                        g["lr"] = val
        if pname == "critic_lr":
            for opt in getattr(agent, "critic_opts", [getattr(agent, "critic_opt", None)]):
                if opt is not None:
                    for g in opt.param_groups:
                        g["lr"] = val
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
    hp = {
        "actor_lr": trial.suggest_float("actor_lr", 1e-5, 1e-3, log=True),
        "critic_lr": trial.suggest_float("critic_lr", 1e-4, 5e-3, log=True),
        "rho": trial.suggest_float("rho", 0.0, 1.0),
    }
    futs = [executor.submit(evaluate, name, K_CONFIGS[0], seed, hp) for seed in VALID_SEEDS]
    scores = [f.result() for f in futs]
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
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
