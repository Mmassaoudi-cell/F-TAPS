"""Step 22: generate standalone IEEE-ready LaTeX table snippets directly
from the saved CSV result files, so every number in the manuscript's tables
can be independently regenerated and checked against source data."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH_DIR = os.path.join(BASE, "results", "benchmarks")
ABL_DIR = os.path.join(BASE, "results", "ablation")
ROB_DIR = os.path.join(BASE, "results", "robustness")
OUT_DIR = os.path.join(BASE, "results", "tables")
os.makedirs(OUT_DIR, exist_ok=True)

NICE_NAME = {
    "F_TAPS": "\\textbf{F-TAPS (ours)}", "ContractTheory": "Contract theory", "MATD3": "MATD3",
    "MADDPG_repro": "MADDPG", "Greedy": "Greedy", "BOStatic": "BO-static",
    "ESPolicy": "ES policy", "Random": "Random", "IPPO": "I-PPO", "Heuristic": "Heuristic",
    "MASAC": "MASAC", "Fixed": "Fixed",
}


def main_table():
    agg = pd.read_csv(os.path.join(BENCH_DIR, "main_table_aggregated.csv")).sort_values("mean_welfare", ascending=False)
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Main benchmark comparison, $K{=}4,N{=}3$, budget${=}100$, 10 test seeds.}",
             r"\label{tab:main}", r"\begin{tabular}{lrrr}", r"\toprule",
             r"Method & Welfare (mean$\pm$std) & Jain & Params \\", r"\midrule"]
    for _, row in agg.iterrows():
        name = NICE_NAME.get(row["method"], row["method"])
        lines.append(f"{name} & {row['mean_welfare']:.1f} $\\pm$ {row['std_welfare']:.1f} & "
                      f"{row['mean_jain']:.3f} & {int(row['n_params']):,} \\\\".replace(",", "{,}"))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(OUT_DIR, "table1_main_benchmark.tex"), "w") as f:
        f.write("\n".join(lines))


def ablation_table():
    agg = pd.read_csv(os.path.join(ABL_DIR, "ablation_2x2_aggregated.csv")).sort_values("mean_welfare", ascending=False)
    nice = {"Share_ThreatWeighted": "Shared + threat-weighted (F-TAPS)",
            "NoShare_ThreatWeighted": "Independent + threat-weighted",
            "Share_NoThreat": "Shared + no threat-weight",
            "NoShare_NoThreat": "Independent + no threat-weight (MADDPG)"}
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{$2\times2$ ablation, $K{=}4$, 10 test seeds.}",
             r"\label{tab:ablation}", r"\begin{tabular}{lrr}", r"\toprule",
             r"Configuration & Welfare (mean$\pm$std) & Jain \\", r"\midrule"]
    for _, row in agg.iterrows():
        name = nice.get(row["cell"], row["cell"])
        bold = row["cell"] == "Share_ThreatWeighted"
        cell = f"\\textbf{{{name}}}" if bold else name
        val = f"\\textbf{{{row['mean_welfare']:.1f} $\\pm$ {row['std_welfare']:.1f}}}" if bold \
            else f"{row['mean_welfare']:.1f} $\\pm$ {row['std_welfare']:.1f}"
        lines.append(f"{cell} & {val} & {row['mean_jain']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(OUT_DIR, "table2_ablation.tex"), "w") as f:
        f.write("\n".join(lines))


def robustness_table():
    agg = pd.read_csv(os.path.join(ROB_DIR, "robustness_aggregated.csv"))
    conditions = [("nominal", "Nominal"), ("budget_shock", "Budget shock (-50\\%)"),
                  ("obs_noise", "Observation noise"), ("quality_shock", "Quality shock (25\\% nodes)")]
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Robustness: welfare (mean over 5 test seeds) and \% degradation vs.\ own nominal.}",
             r"\label{tab:robust}", r"\begin{tabular}{lrrr}", r"\toprule",
             r"Condition & F-TAPS welfare & MADDPG welfare & F-TAPS \% degr. \\", r"\midrule"]
    for key, label in conditions:
        f_row = agg[(agg.method == "F_TAPS") & (agg.perturbation == key)].iloc[0]
        m_row = agg[(agg.method == "MADDPG_repro") & (agg.perturbation == key)].iloc[0]
        degr = "---" if key == "nominal" else f"{f_row['mean_pct_degradation']:.1f}\\%"
        lines.append(f"{label} & {f_row['mean_welfare']:.1f} & {m_row['mean_welfare']:.1f} & {degr} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(OUT_DIR, "table3_robustness.tex"), "w") as f:
        f.write("\n".join(lines))


def scalability_table():
    agg = pd.read_csv(os.path.join(BENCH_DIR, "scalability_aggregated.csv"))
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Scalability: welfare and parameters vs.\ $K$.}",
             r"\label{tab:scale}", r"\begin{tabular}{lrrr}", r"\toprule",
             r"Method & $K$ & Welfare (mean$\pm$std) & Params \\", r"\midrule"]
    for _, row in agg.sort_values(["method", "K"]).iterrows():
        name = NICE_NAME.get(row["method"], row["method"])
        lines.append(f"{name} & {int(row['K'])} & {row['mean_welfare']:.1f} $\\pm$ {row['std_welfare']:.1f} & "
                      f"{int(row['n_params']):,} \\\\".replace(",", "{,}"))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(OUT_DIR, "table4_scalability.tex"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main_table()
    ablation_table()
    robustness_table()
    scalability_table()
    print("Tables written to", OUT_DIR)
