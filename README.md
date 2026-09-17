# F-TAPS: Follower-side Threat-Aware Parameter-Shared Multi-Agent Actor-Critic

Reference implementation of **F-TAPS**, a parameter-shared, threat-risk-weighted multi-agent actor-critic for incentive-driven honeypot deployment in smart-grid Advanced Metering Infrastructure (AMI), together with the full simulation environment, benchmark suite, and experiment scripts used to evaluate it.


## Setup

```bash
pip install -r requirements.txt
```

Python 3.11+ recommended. All models are small MLP/pooled-attention networks on a low-dimensional simulation; CPU-only execution is sufficient (CUDA is used opportunistically if available).

## Repository structure

```
code/env/            AMI Stackelberg honeypot-incentive simulation environment,
                      including the real-attack-trace generalization mode
code/agents/         F-TAPS, MADDPG, I-PPO, MATD3, MASAC, and classical
                      baselines (Bayesian-optimized static mechanism,
                      evolution-strategy policy, contract-theory policy)
code/experiments/    Runnable scripts: benchmark suite, scalability sweep,
                      ablation, robustness + policy state-dependence
                      diagnostic, hyperparameter tuning, real-attack-trace
                      generalization check, figure/table generation
data/                Derived, non-stationary attack-intensity trace used for
                      the real-data generalization experiment
FINAL_MODEL_CONFIG.yaml   Frozen F-TAPS architecture/hyperparameter configuration
DATA_SPLIT_MANIFEST.csv   Seed-based validation/test split protocol
```

## Running the experiments

```bash
# Full benchmark suite (12 methods, held-out test seeds) + scalability sweep
python code/experiments/run_benchmarks.py

# 2x2 ablation over F-TAPS's two components
python code/experiments/run_ablation.py

# Robustness (budget/observation/quality shocks) + action-variance diagnostic
python code/experiments/run_robustness.py

# Statistical significance testing (Wilcoxon, Holm-Bonferroni, Cohen's d)
python code/experiments/run_statistics.py

# Real attack-intensity trace generalization check
python code/experiments/run_real_attack_experiment.py

# Hyperparameter tuning (Optuna TPE)
python code/experiments/run_tuning.py --candidate F_TAPS
python code/experiments/run_tuning_baselines.py --candidate MATD3
python code/experiments/run_tuning_baselines.py --candidate MASAC
```

Each script writes its outputs (raw and aggregated CSVs) to a `results/<stage>/` directory, created on first run.

## Reproducibility notes

- All environment randomness is seeded via `numpy.random.default_rng(seed)`; the seed is the sole source of run-to-run variation.
- `DATA_SPLIT_MANIFEST.csv` fixes disjoint seed pools for validation (candidate comparison and hyperparameter tuning) and held-out test evaluation, so no test-seed result influenced any design or tuning decision upstream.
- `FINAL_MODEL_CONFIG.yaml` records F-TAPS's exact frozen architecture and hyperparameters, tuned on validation seeds only, prior to any test-seed evaluation.

## License

MIT License (see `LICENSE`).
