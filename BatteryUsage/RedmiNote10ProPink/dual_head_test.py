"""
dual_head_test.py (v4)
================================

Inference / evaluation helper for the dual-head energy model with separate CPU and Screen models.
Uses meta-features for CPU portability across devices.
Produces:
  • CSV with per-sample estimates of E_CPU, E_Screen, E_TotalPred, E_TotalTrue
  • Overall MAE & MAPE printed to console.

Compatible with dual_head_energy_model.py (v3).

Usage
-----
python dual_head_test.py --csv new_log.csv \
                         --cpu_model cpu_model.pt \
                         --screen_model screen_model.pt \
                         --scalers feature_scalers.pkl \
                         --meta_params meta_feature_params.pkl \
                         --out predictions.csv
"""

import argparse, pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error

###############################################
# Utility funcs
###############################################

def compute_energy_mj(df: pd.DataFrame) -> np.ndarray:
    return (
        df["batt_current_mA"].values *
        df["batt_voltage_V"].values *
        df["sample_ms"].values / 1_000.0
    )

def make_feature_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cpu_df, screen_df) using simple column masks."""
    cpu_mask = df.columns.str.contains(r"core\d|cluster|cpu_.*|batt_temp_C", case=False)
    screen_mask = df.columns.str.contains(r"display_|brightness|batt_temp_C", case=False)
    cpu_df, screen_df = df.loc[:, cpu_mask].fillna(0.0), df.loc[:, screen_mask].fillna(0.0)
    return cpu_df, screen_df

def make_cpu_meta_features(cpu_df: pd.DataFrame, freqs: list) -> np.ndarray:
    """Transform CPU features into device-agnostic meta-features.
       Returns: [idle_time, dyn_power_c0, dyn_power_c1, active_time_c0, active_time_c1]
    """
    core_cols = [col for col in cpu_df.columns if col.startswith("core") and "idle" in col]
    idle_time = cpu_df[core_cols].sum(axis=1).values

    cluster0_cols = [col for col in cpu_df.columns if col.startswith("cluster0_") and col.endswith("_ms")]
    cluster1_cols = [col for col in cpu_df.columns if col.startswith("cluster1_") and col.endswith("_ms")]

    def parse_freq(col: str) -> float:
        return float(col.split("_")[1].replace("_ms", ""))

    c0_freqs = [parse_freq(col) / 1e6 for col in cluster0_cols]
    c1_freqs = [parse_freq(col) / 1e6 for col in cluster1_cols]

    dyn_power_c0 = np.zeros(len(cpu_df))
    active_time_c0 = np.zeros(len(cpu_df))
    for col, f in zip(cluster0_cols, c0_freqs):
        dyn_power_c0 += cpu_df[col].values * f
        active_time_c0 += cpu_df[col].values

    dyn_power_c1 = np.zeros(len(cpu_df))
    active_time_c1 = np.zeros(len(cpu_df))
    for col, f in zip(cluster1_cols, c1_freqs):
        dyn_power_c1 += cpu_df[col].values * f
        active_time_c1 += cpu_df[col].values

    battery_temp = cpu_df["batt_temp_C"].values if "batt_temp_C" in cpu_df.columns else np.zeros(len(cpu_df))
    return np.vstack([
        idle_time,
        dyn_power_c0,
        dyn_power_c1,
        active_time_c0,
        active_time_c1,
        battery_temp
    ]).T.astype("float32")

def create_cpu_head(in_cpu: int = 6, hidden: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_cpu, hidden), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden//2, 1), nn.Softplus()
    )

def create_screen_head(in_scr: int = 4, hidden: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_scr, hidden//2), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden//2, hidden//4), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden//4, 1), nn.Softplus()
    )

###############################################
# Main
###############################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Telemetry CSV to evaluate")
    ap.add_argument("--cpu_model", default="cpu_model.pt", help="Path to CPU model weights")
    ap.add_argument("--screen_model", default="screen_model.pt", help="Path to Screen model weights")
    ap.add_argument("--scalers", default="feature_scalers.pkl", help="Path to feature scalers")
    ap.add_argument("--meta_params", default="meta_feature_params.pkl", help="Path to meta-feature parameters")
    ap.add_argument("--hidden", type=int, default=128, help="Hidden size used at training")
    ap.add_argument("--out", default="predictions.csv", help="Where to save per-sample predictions")
    args = ap.parse_args()

    # Load data
    df = pd.read_csv(args.csv)
    y_true = compute_energy_mj(df)

    cpu_df, scr_df = make_feature_split(df)

    # Load preprocessors
    with open(args.scalers, "rb") as f:
        scalers = pickle.load(f)
    with open(args.meta_params, "rb") as f:
        meta_params = pickle.load(f)

    # Transform CPU features
    X_cpu = make_cpu_meta_features(cpu_df, meta_params["cpu_freqs"])
    X_cpu = scalers["cpu"].transform(X_cpu)
    X_scr = scalers["screen"].transform(scr_df.values).astype("float32")

    # Build models & load weights
    cpu_net = create_cpu_head(in_cpu=X_cpu.shape[1], hidden=args.hidden)
    cpu_net.load_state_dict(torch.load(args.cpu_model, map_location="cpu"))
    cpu_net.eval()

    screen_net = create_screen_head(in_scr=X_scr.shape[1], hidden=args.hidden)
    screen_net.load_state_dict(torch.load(args.screen_model, map_location="cpu"))
    screen_net.eval()

    # Inference
    with torch.no_grad():
        tc = torch.from_numpy(X_cpu)
        ts = torch.from_numpy(X_scr)
        e_cpu = cpu_net(tc).squeeze().numpy()
        e_scr = screen_net(ts).squeeze().numpy()
        e_tot = e_cpu + e_scr

    # Metrics
    mae = mean_absolute_error(y_true, e_tot)
    mape = np.mean(np.abs(e_tot - y_true) / (y_true + 1e-3))
    print(f"Samples: {len(df):,}\nMAE   : {mae:.2f} mJ\nMAPE  : {mape*100:.2f}%")

    # Save CSV
    out_df = df[["timestamp"]].copy() if "timestamp" in df.columns else pd.DataFrame(index=df.index)
    out_df["E_CPU_pred_mJ"] = e_cpu
    out_df["E_Screen_pred_mJ"] = e_scr
    out_df["E_Total_pred_mJ"] = e_tot
    out_df["E_Total_true_mJ"] = y_true
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Per-sample predictions written to {args.out}")

if __name__ == "__main__":
    main()