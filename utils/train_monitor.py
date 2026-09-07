# -*- coding: utf-8 -*-
import csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class CSVLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = self.path.exists() and self.path.stat().st_size > 0

    def log(self, row: dict):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)

def load_or_create_fixed_val_indices(path, dataset_len, num_samples=8, seed=2026):
    import random
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    n = min(num_samples, dataset_len)
    idx = sorted(rng.sample(range(dataset_len), n)) if dataset_len > n else list(range(dataset_len))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    return idx

def plot_curves(log_csv, out_png):
    import pandas as pd
    log_csv = Path(log_csv)
    if not log_csv.exists():
        return
    df = pd.read_csv(log_csv)
    if len(df) == 0 or "step" not in df.columns:
        return
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for col, label in [("train_loss", "Train loss"), ("val_loss", "Val loss"), ("val_mse", "Val MSE")]:
        if col in df.columns:
            sub = df.dropna(subset=[col])
            if len(sub):
                plt.plot(sub["step"], sub[col], label=label)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Curves")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
