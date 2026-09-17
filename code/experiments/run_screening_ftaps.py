"""Screen the post-hoc F_TAPS candidate (threat-weighted objective + pooled
parameter-shared backbone, dropping exact-leader and twin-critic/attention
which destabilized D/E in stage2) using the same protocol/seeds as Stage 2,
and append its results to results/screening/stage2_validation_screening.csv
and stage2_aggregated.csv."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
from code.experiments.run_screening import (
    stage2_screening, OUT_DIR, K_CONFIGS, VALID_SEEDS_STAGE2, run_one
)

if __name__ == "__main__":
    old_path = os.path.join(OUT_DIR, "stage2_validation_screening.csv")
    agg_path = os.path.join(OUT_DIR, "stage2_aggregated.csv")
    df_old = pd.read_csv(old_path) if os.path.exists(old_path) else None
    agg_old = pd.read_csv(agg_path) if os.path.exists(agg_path) else None

    df_new, agg_new = stage2_screening(["F_TAPS"])  # this overwrites old_path/agg_path internally

    df_all = pd.concat([df_old, df_new], ignore_index=True) if df_old is not None else df_new
    df_all.to_csv(old_path, index=False)
    agg_all = pd.concat([agg_old, agg_new], ignore_index=True) if agg_old is not None else agg_new
    agg_all.to_csv(agg_path, index=False)
    print(agg_all.to_string(index=False))
