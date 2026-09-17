"""Step 22: generate the data-driven manuscript figures (Fig. 3-6) directly
from the saved CSV result files. Figures 1-2 (architecture / conceptual
workflow diagrams) are schematic and are built directly in the LaTeX
manuscript (TikZ), not generated from data here."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH_DIR = os.path.join(BASE, "results", "benchmarks")
ABL_DIR = os.path.join(BASE, "results", "ablation")
ROB_DIR = os.path.join(BASE, "results", "robustness")
FIG_DIR = os.path.join(BASE, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FINAL = "F_TAPS"
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

NICE_NAME = {
    "F_TAPS": "F-TAPS", "MADDPG_repro": "MADDPG", "BOStatic": "BO-static",
    "ContractTheory": "Contract theory", "ESPolicy": "ES policy", "IPPO": "I-PPO",
    "MATD3": "MATD3", "MASAC": "MASAC", "Random": "Random", "Fixed": "Fixed",
    "Heuristic": "Heuristic", "Greedy": "Greedy",
}


def fig3_main_benchmark():
    agg = pd.read_csv(os.path.join(BENCH_DIR, "main_table_aggregated.csv")).sort_values("mean_welfare")
    colors = ["#d62728" if m == FINAL else "#4c72b0" for m in agg["method"]]
    labels = [NICE_NAME.get(m, m) for m in agg["method"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, agg["mean_welfare"], xerr=agg["std_welfare"], color=colors, capsize=3)
    ax.set_xlabel("Steady-state social welfare (test seeds, mean ± std, n=10)")
    ax.set_title("Main benchmark comparison (K=4, N=3, budget=100)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_main_benchmark.png"))
    fig.savefig(os.path.join(FIG_DIR, "fig3_main_benchmark.pdf"))
    plt.close(fig)


def fig4_scalability():
    agg = pd.read_csv(os.path.join(BENCH_DIR, "scalability_aggregated.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for method, marker, color in [("MADDPG_repro", "o", "#4c72b0"), ("BOStatic", "^", "#55a868"),
                                    (FINAL, "s", "#d62728")]:
        sub = agg[agg.method == method].sort_values("K")
        label = NICE_NAME.get(method, method)
        axes[0].errorbar(sub["K"], sub["mean_welfare"], yerr=sub["std_welfare"],
                          marker=marker, label=label, color=color, capsize=3)
        if method != "BOStatic":  # 0 params, would flatten the log-scale axis at the bottom uninformatively
            axes[1].plot(sub["K"], sub["n_params"], marker=marker, label=label, color=color)
    axes[0].set_xlabel("Number of SPSs (K)"); axes[0].set_ylabel("Steady-state welfare")
    axes[0].set_title("(a) Welfare vs. scale"); axes[0].legend()
    axes[1].set_xlabel("Number of SPSs (K)"); axes[1].set_ylabel("Trainable parameters")
    axes[1].set_yscale("log"); axes[1].set_title("(b) Parameter growth vs. scale"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_scalability.png"))
    fig.savefig(os.path.join(FIG_DIR, "fig4_scalability.pdf"))
    plt.close(fig)


def fig5_ablation_robustness():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    abl = pd.read_csv(os.path.join(ABL_DIR, "ablation_2x2_aggregated.csv")).sort_values("mean_welfare")
    abl_nice = {"Share_ThreatWeighted": "Shared + threat-weighted (F-TAPS)",
                "NoShare_ThreatWeighted": "Independent + threat-weighted",
                "Share_NoThreat": "Shared + no threat-weight",
                "NoShare_NoThreat": "Independent + no threat-weight (MADDPG)"}
    colors = ["#d62728" if c == "Share_ThreatWeighted" else "#4c72b0" for c in abl["cell"]]
    labels = [abl_nice.get(c, c) for c in abl["cell"]]
    axes[0].barh(labels, abl["mean_welfare"], xerr=abl["std_welfare"], color=colors, capsize=3)
    axes[0].set_xlabel("Steady-state welfare"); axes[0].set_title("(a) 2x2 component ablation")

    rob = pd.read_csv(os.path.join(ROB_DIR, "robustness_aggregated.csv"))
    perturbations = ["budget_shock", "obs_noise", "quality_shock"]
    x = np.arange(len(perturbations)); width = 0.35
    for i, method in enumerate(["MADDPG_repro", FINAL]):
        vals = [rob[(rob.method == method) & (rob.perturbation == p)]["mean_pct_degradation"].values[0]
                for p in perturbations]
        axes[1].bar(x + (i - 0.5) * width, vals, width, label=NICE_NAME.get(method, method),
                    color="#d62728" if method == FINAL else "#4c72b0")
    axes[1].set_xticks(x); axes[1].set_xticklabels(perturbations, rotation=15)
    axes[1].set_ylabel("% welfare degradation vs. nominal"); axes[1].set_title("(b) Robustness")
    axes[1].legend(); axes[1].axhline(0, color="k", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_ablation_robustness.png"))
    fig.savefig(os.path.join(FIG_DIR, "fig5_ablation_robustness.pdf"))
    plt.close(fig)


def fig6_pareto():
    agg = pd.read_csv(os.path.join(BENCH_DIR, "main_table_aggregated.csv"))
    agg = agg[agg["n_params"] > 0]  # exclude non-learned (0-param) methods from a param-count Pareto plot
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for _, row in agg.iterrows():
        color = "#d62728" if row["method"] == FINAL else "#4c72b0"
        ax.scatter(row["n_params"], row["mean_welfare"], color=color, s=60, zorder=3)
        ax.annotate(NICE_NAME.get(row["method"], row["method"]), (row["n_params"], row["mean_welfare"]),
                    textcoords="offset points", xytext=(6, 3), fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Steady-state welfare")
    ax.set_title("Performance-efficiency Pareto (learned methods, K=4)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig6_pareto.png"))
    fig.savefig(os.path.join(FIG_DIR, "fig6_pareto.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig3_main_benchmark()
    fig4_scalability()
    if os.path.exists(os.path.join(ABL_DIR, "ablation_2x2_aggregated.csv")) and \
       os.path.exists(os.path.join(ROB_DIR, "robustness_aggregated.csv")):
        fig5_ablation_robustness()
    else:
        print("Skipping fig5 (ablation/robustness results not yet available)")
    fig6_pareto()
    print("Figures written to", FIG_DIR)
