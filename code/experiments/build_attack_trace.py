"""Derives a real, non-stationary attack-intensity trace from CIC-IDS-2017
(Friday-WorkingHours-Afternoon-DDoS capture) to replace the synthetic
bounded-random-walk attack-rate process for a real-data-grounded
generalization check (see run_real_attack_experiment.py).

The capture file has no absolute timestamp column, but its rows are written
in packet-capture (chronological) order, so binning consecutive rows into
fixed-size windows and computing the fraction of non-BENIGN flows per window
yields a genuine empirical attack-intensity time series with the capture's
real burst structure (a benign period followed by a sustained DDoS window),
rather than an author-chosen synthetic process.
"""
import os
import numpy as np
import pandas as pd

RAW_PATH = r"C:\Users\MMASSAOUDI\Desktop\Data\CIC-IDS- 2017\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "data", "real_attack_intensity_ddos.npy")
WINDOW = 2000


def build():
    chunks = pd.read_csv(RAW_PATH, usecols=lambda c: c.strip() == "Label", chunksize=200_000)
    labels = np.concatenate([ch[ch.columns[0]].astype(str).str.strip().values for ch in chunks])
    is_attack = (labels != "BENIGN").astype(np.float64)
    n_windows = len(is_attack) // WINDOW
    binned = is_attack[: n_windows * WINDOW].reshape(n_windows, WINDOW).mean(axis=1)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.save(OUT_PATH, binned.astype(np.float32))
    print(f"Built trace: {n_windows} windows of {WINDOW} flows each, "
          f"attack-fraction range [{binned.min():.3f}, {binned.max():.3f}], mean {binned.mean():.3f}")
    print(f"Source: CIC-IDS-2017 Friday-Afternoon-DDoS capture "
          f"({(labels != 'BENIGN').sum()} attack / {(labels == 'BENIGN').sum()} benign flows total)")
    return binned


if __name__ == "__main__":
    build()
