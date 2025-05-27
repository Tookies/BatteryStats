"""
dual_head_energy_model.py (v3)
================================================

Train two jointly-optimized neural sub-models that output separate positive
energy estimates for CPU and Screen given **only** the aggregate ground-truth
energy (smartphone battery readings).

* Works with the single CSV file you already log.
* Automatically computes target energy in milli-joules:
      batt_current_mA × batt_voltage_V × sample_ms / 1000.
* Robust scaling of CPU and Screen features with RobustScaler.
* Early stopping on validation MAPE, patience = 25 epochs.
* Saves artifacts: cpu_model.pt, screen_model.pt, feature_scalers.pkl, meta_feature_params.pkl.
* Supports portability with meta-features for CPU (fixed input size).


Dependencies
------------
```
pip install pandas scikit-learn torch tqdm
```

Usage example
-------------
Train on Device 1:
```
python dual_head_energy_model.py --csv energy_log.csv --epochs 400 --hidden 128
```

Inference example:
```python
from joblib import load
import torch
import pandas as pd
scalers, meta_params = load('feature_scalers.pkl'), load('meta_feature_params.pkl')
cpu_net, scr_net = torch.load('cpu_model.pt'), torch.load('screen_model.pt')
df = pd.read_csv('new_log.csv')
cpu_df, scr_df = make_feature_split(df)
X_cpu = make_cpu_meta_features(cpu_df, meta_params['cpu_freqs'], meta_params['cpu_voltages'])
X_cpu = scalers['cpu'].transform(X_cpu)
X_scr = scalers['screen'].transform(scr_df.values.astype('float32'))
E_cpu = cpu_net(torch.from_numpy(X_cpu)).detach().numpy()
E_scr = scr_net(torch.from_numpy(X_scr)).detach().numpy()
```
"""

import argparse, os, pickle, math
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from tqdm import trange

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

###############################################
# Helpers
###############################################

def compute_energy_mj(df: pd.DataFrame) -> np.ndarray:
    """Convert battery telemetry to energy in milli-joules (mJ).
           E [mJ] = I[mA] * V[V] * dt[ms] / 1e3"""
    return (
        df["batt_current_mA"].values *
        df["batt_voltage_V"].values *
        df["sample_ms"].values / 1_000.0
    )

def make_feature_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (cpu_df, screen_df) using simple column masks."""
    cpu_mask = df.columns.str.contains(r"core\d|cluster|cpu_.*|batt_temp_C", case=False)
    screen_mask = df.columns.str.contains(r"display_|brightness|batt_temp_C", case=False)
    cpu_df, screen_df = df.loc[:, cpu_mask], df.loc[:, screen_mask]
    return cpu_df.fillna(0.0), screen_df.fillna(0.0)

def make_cpu_meta_features(cpu_df: pd.DataFrame, freqs: list = None) -> np.ndarray:
    """Transform CPU features into device-agnostic meta-features.
       Returns: [idle_time, dyn_power_c0, dyn_power_c1, active_time_c0, active_time_c1, cpu_load_pct, cpu_temp_C]
    """
    # Sum idle times across all cores
    core_cols = [col for col in cpu_df.columns if col.startswith("core") and "idle" in col]
    idle_time = cpu_df[core_cols].sum(axis=1).values

    # Extract cluster frequency columns
    cluster0_cols = [col for col in cpu_df.columns if col.startswith("cluster0_") and col.endswith("_ms")]
    cluster1_cols = [col for col in cpu_df.columns if col.startswith("cluster1_") and col.endswith("_ms")]

    # Parse frequencies from column names (e.g., cluster0_300000_ms -> 300000 Hz)
    def parse_freq(col: str) -> float:
        return float(col.split("_")[1].replace("_ms", ""))

    if freqs is None:
        c0_freqs = [parse_freq(col) / 1e6 for col in cluster0_cols]  # MHz
        c1_freqs = [parse_freq(col) / 1e6 for col in cluster1_cols]  # MHz
    else:
        c0_freqs = [f / 1e6 for f in freqs[:len(cluster0_cols)]]
        c1_freqs = [f / 1e6 for f in freqs[len(cluster0_cols):]]

    # Compute dynamic power: Σ (t_k * f_k * V_k^2) for each cluster
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
    # Combine meta-features
    meta_features = np.vstack([
        idle_time,
        dyn_power_c0,
        dyn_power_c1,
        active_time_c0,
        active_time_c1,
        battery_temp
    ]).T
    return meta_features.astype("float32"), c0_freqs + c1_freqs

class TwoHeadNet(nn.Module):
    def __init__(self, in_cpu: int = 6, in_scr: int = 4, hidden: int = 128):
        super().__init__()
        self.cpu_head = nn.Sequential(
            nn.Linear(in_cpu, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden//2, 1), nn.Softplus()
        )
        self.scr_head = nn.Sequential(
            nn.Linear(in_scr, hidden//2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden//2, hidden//4), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden//4, 1), nn.Softplus()
        )

    def forward(self, x_cpu: torch.Tensor, x_scr: torch.Tensor):
        e_cpu = self.cpu_head(x_cpu)
        e_scr = self.scr_head(x_scr)
        return e_cpu, e_scr, e_cpu + e_scr

###############################################
# Training
###############################################

def train_model(df: pd.DataFrame, args):
    # Target
    y = compute_energy_mj(df).astype("float32")

    # Split features
    cpu_df, scr_df = make_feature_split(df)

    # Transform CPU features to meta-features
    X_cpu, freqs = make_cpu_meta_features(cpu_df)
    scaler_cpu = RobustScaler().fit(X_cpu)
    X_cpu = scaler_cpu.transform(X_cpu)

    # Scale screen features
    scaler_scr = RobustScaler().fit(scr_df.values)
    X_scr = scaler_scr.transform(scr_df.values).astype("float32")

    # Save meta-feature parameters
    meta_params = {"cpu_freqs": freqs}

    Xc_train, Xc_val, Xs_train, Xs_val, y_train, y_val = train_test_split(
        X_cpu, X_scr, y, test_size=0.20, random_state=42, shuffle=True)

    # Build loaders
    def make_loader(Xc, Xs, y, bs):
        ds = TensorDataset(torch.from_numpy(Xc), torch.from_numpy(Xs), torch.from_numpy(y.reshape(-1,1)))
        return DataLoader(ds, batch_size=bs, shuffle=True)

    train_loader = make_loader(Xc_train, Xs_train, y_train, args.batch)
    val_loader = make_loader(Xc_val, Xs_val, y_val, args.batch)

    # Model & optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = TwoHeadNet(in_cpu=X_cpu.shape[1], in_scr=X_scr.shape[1], hidden=args.hidden).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.L1Loss()

    best_val_mape, patience_cnt = math.inf, 0
    for epoch in trange(args.epochs, desc="Training"):
        net.train()
        train_losses = []
        for Xc_b, Xs_b, y_b in train_loader:
            Xc_b, Xs_b, y_b = Xc_b.to(device), Xs_b.to(device), y_b.to(device)
            opt.zero_grad()
            e_cpu, e_scr, y_hat = net(Xc_b, Xs_b)
            loss = criterion(y_hat, y_b)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        # Validation
        net.eval()
        val_losses, val_mape = [], []
        with torch.no_grad():
            for Xc_b, Xs_b, y_b in val_loader:
                Xc_b, Xs_b, y_b = Xc_b.to(device), Xs_b.to(device), y_b.to(device)
                _, _, y_hat = net(Xc_b, Xs_b)
                val_losses.append(criterion(y_hat, y_b).item())
                val_mape.append((torch.abs(y_hat - y_b) / (y_b + 1e-3)).mean().item())
        mean_train = np.mean(train_losses)
        mean_val = np.mean(val_losses)
        mean_mape = np.mean(val_mape)
        print(f"Epoch {epoch:03d} | train {mean_train:.4f} | val {mean_val:.4f} | val MAPE {mean_mape*100:.2f}%")

        # Early stop
        if mean_mape < best_val_mape - 1e-4:
            best_val_mape = mean_mape
            patience_cnt = 0
            torch.save(net.state_dict(), "best_net.pt")
        else:
            patience_cnt += 1
            if patience_cnt > args.patience:
                print("Early stopping")
                break

    # Restore best
    net.load_state_dict(torch.load("best_net.pt"))

    # Final evaluation on full data
    net.eval()
    with torch.no_grad():
        X_cpu_tensor = torch.from_numpy(X_cpu).to(device)
        X_scr_tensor = torch.from_numpy(X_scr).to(device)
        _, _, y_hat_full = net(X_cpu_tensor, X_scr_tensor)
        y_hat_full = y_hat_full.cpu().numpy().squeeze()
        full_mape = np.mean(np.abs(y_hat_full - y) / (y + 1e-3))
        print(f"Full-data MAPE: {full_mape*100:.2f}%")

    # Save artifacts
    torch.save(net.cpu_head.state_dict(), "cpu_model.pt")
    torch.save(net.scr_head.state_dict(), "screen_model.pt")
    with open("feature_scalers.pkl", "wb") as f:
        pickle.dump({"cpu": scaler_cpu, "screen": scaler_scr}, f)
    with open("meta_feature_params.pkl", "wb") as f:
        pickle.dump(meta_params, f)
    print("Artifacts saved: cpu_model.pt, screen_model.pt, feature_scalers.pkl, meta_feature_params.pkl")

###############################################
# CLI
###############################################
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to telemetry CSV")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=25, help="Early-stop patience on val MAPE")
    p.add_argument("--cpu_model", default=None, help="Path to pre-trained CPU model")
    p.add_argument("--screen_model", default=None, help="Path to pre-trained screen model")
    p.add_argument("--scalers", default=None, help="Path to pre-trained scalers")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df):,} rows × {len(df.columns)} cols from {args.csv}")

    train_model(df, args)