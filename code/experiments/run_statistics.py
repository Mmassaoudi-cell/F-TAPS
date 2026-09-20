"""Step 20/22: statistical significance testing of F_TAPS vs. every benchmark
on the main TEST-seed table (paired by seed). Wilcoxon signed-rank test
(non-parametric, appropriate for n=10 paired samples with no normality
assumption) with Holm-Bonferroni correction across the 11 comparisons, plus
paired t-test and Cohen's d effect size for reference. Also produces
BENCHMARK_WTL.csv (wins/ties/losses per dataset/config)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH_DIR = os.path.join(BASE, "results", "benchmarks")


def cohens_d(a, b):
    diff = a - b
    return float(diff.mean() / (diff.std(ddof=1) + 1e-12))


def main():
    df = pd.read_csv(os.path.join(BENCH_DIR, "main_table_raw.csv"))
    pivot = df.pivot(index="seed", columns="method", values="welfare")
    final = "F_TAPS"
    others = [c for c in pivot.columns if c != final]

    rows = []
    for other in others:
        a, b = pivot[final].values, pivot[other].values
        try:
            w_stat, w_p = stats.wilcoxon(a, b)
        except ValueError:
            w_stat, w_p = np.nan, 1.0
        t_stat, t_p = stats.ttest_rel(a, b)
        d = cohens_d(a, b)
        rows.append(dict(benchmark=other, mean_final=a.mean(), mean_benchmark=b.mean(),
                          mean_diff=a.mean() - b.mean(), wilcoxon_stat=w_stat, wilcoxon_p=w_p,
                          ttest_stat=t_stat, ttest_p=t_p, cohens_d=d))
    res = pd.DataFrame(rows).sort_values("wilcoxon_p")

    # Holm-Bonferroni correction on Wilcoxon p-values
    m = len(res)
    res = res.reset_index(drop=True)
    res["holm_rank"] = np.arange(1, m + 1)
    res["holm_alpha"] = 0.05 / (m - res["holm_rank"] + 1)
    res["holm_significant"] = res["wilcoxon_p"] < res["holm_alpha"]
    # enforce monotonicity of Holm procedure (once a hypothesis fails to reject, so do all with larger p)
    reject = res["holm_significant"].values.copy()
    for i in range(1, len(reject)):
        if not reject[i - 1]:
            reject[i] = False
    res["holm_significant"] = reject

    res["verdict"] = np.select(
        [(~res["holm_significant"]),
         (res["holm_significant"]) & (res["mean_diff"] > 0),
         (res["holm_significant"]) & (res["mean_diff"] < 0)],
        ["statistically indistinguishable", "statistically superior (F_TAPS)", "inferior (F_TAPS)"],
        default="statistically indistinguishable",
    )
    res.to_csv(os.path.join(BENCH_DIR, "statistical_tests.csv"), index=False)
    print(res.to_string(index=False))

    wins = int((res["verdict"] == "statistically superior (F_TAPS)").sum())
    ties = int((res["verdict"] == "statistically indistinguishable").sum())
    losses = int((res["verdict"] == "inferior (F_TAPS)").sum())
    print(f"\nF_TAPS: {wins} wins / {ties} ties / {losses} losses (Holm-corrected Wilcoxon, alpha=0.05, n={len(pivot)} seeds)")

    wtl = pd.DataFrame([{
        "final_model": final, "config": "K=4,N=3,budget=100 (main table)",
        "n_benchmarks": len(others), "wins": wins, "ties": ties, "losses": losses,
    }])
    wtl.to_csv(os.path.join(BASE, "BENCHMARK_WTL.csv"), index=False)


if __name__ == "__main__":
    main()
