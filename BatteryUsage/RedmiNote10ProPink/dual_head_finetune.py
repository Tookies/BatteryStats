#!/usr/bin/env python
"""
dual_head_finetune.py
=====================
Fine-tune both heads of the dual-energy model on a *subset* of data
for one target device.

Saves:
  cpu_model_ft_<tag>.pt
  screen_model_ft_<tag>.pt
  (tag encodes device, fraction and epochs)

Example
-------
python dual_head_finetune.py \
    --train_csv ../RedmiNote10ProBlue/power_log_cleaned_.csv \
    --cpu_model cpu_model.pt \
    --screen_model screen_model.pt \
    --scalers feature_scalers.pkl \
    --meta_params meta_feature_params.pkl \
    --fraction 0.2 --epochs 20 --hidden 256 \
    --lr 5e-4 --out_dir finetuned_models
"""
import argparse, os, random, pickle, math, pathlib
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from tqdm import trange
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
# ------------------------------------------------------------
# ---------   общие функции (срезаны до нужного)   ------------
# ------------------------------------------------------------
def compute_energy_mj(df: pd.DataFrame) -> np.ndarray:
    return df["batt_current_mA"].values * df["batt_voltage_V"].values * df["sample_ms"].values / 1_000.0

def make_feature_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cpu_mask = df.columns.str.contains(r"core\d|cluster|cpu_.*|batt_temp_C", case=False)
    scr_mask = df.columns.str.contains(r"display_|brightness|batt_temp_C", case=False)
    return df.loc[:, cpu_mask].fillna(0.0), df.loc[:, scr_mask].fillna(0.0)

def make_cpu_meta_features(cpu_df: pd.DataFrame) -> np.ndarray:
    core_cols   = [c for c in cpu_df.columns if c.startswith("core") and "idle" in c]
    cluster0_ms = [c for c in cpu_df.columns if c.startswith("cluster0_") and c.endswith("_ms")]
    cluster1_ms = [c for c in cpu_df.columns if c.startswith("cluster1_") and c.endswith("_ms")]

    idle_time = cpu_df[core_cols].sum(axis=1).values
    dyn_p0 = cpu_df[cluster0_ms].sum(axis=1).values
    dyn_p1 = cpu_df[cluster1_ms].sum(axis=1).values
    act0   = dyn_p0.copy()
    act1   = dyn_p1.copy()
    batt_T = cpu_df["batt_temp_C"].values if "batt_temp_C" in cpu_df.columns else np.zeros(len(cpu_df))

    return np.vstack([idle_time, dyn_p0, dyn_p1, act0, act1, batt_T]).T.astype("float32")

def create_cpu_head(in_cpu=6, hidden=128):
    return nn.Sequential(
        nn.Linear(in_cpu, hidden), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden//2, 1), nn.Softplus()
    )
def create_screen_head(in_scr, hidden=128):
    return nn.Sequential(
        nn.Linear(in_scr, hidden//2), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden//2, hidden//4), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden//4, 1), nn.Softplus()
    )
# ------------------------------------------------------------
def finetune(train_csv:str, cpu_model_path:str, scr_model_path:str,
             scalers_path:str, meta_path:str,
             fraction:float, epochs:int, lr:float, batch:int,
             hidden:int, seed:int, out_dir:str)->Dict[str,str]:

    device_tag   = pathlib.Path(train_csv).parent.name  # RedmiNote10ProBlue …
    tag          = f"{device_tag}_f{int(fraction*100)}_e{epochs}"
    os.makedirs(out_dir, exist_ok=True)

    # --- Load data & sample subset --------------------------
    df_full = pd.read_csv(train_csv)
    n_sub   = int(len(df_full)*fraction)
    random.seed(seed)
    subset_idx = random.sample(range(len(df_full)), n_sub)
    df = df_full.iloc[subset_idx].reset_index(drop=True)

    y = compute_energy_mj(df).astype("float32")
    cpu_df, scr_df = make_feature_split(df)
    with open(scalers_path,"rb") as f: scalers = pickle.load(f)

    X_cpu = make_cpu_meta_features(cpu_df)
    X_cpu = scalers["cpu"].transform(X_cpu)
    X_scr = scalers["screen"].transform(scr_df.values).astype("float32")

    Xc_tr, Xc_val, Xs_tr, Xs_val, y_tr, y_val = train_test_split(
        X_cpu, X_scr, y, test_size=0.2, random_state=seed, shuffle=True)

    def make_loader(Xc,Xs,y,bs):
        ds = TensorDataset(torch.from_numpy(Xc), torch.from_numpy(Xs), torch.from_numpy(y.reshape(-1,1)))
        return DataLoader(ds,batch_size=bs,shuffle=True)

    train_loader = make_loader(Xc_tr,Xs_tr,y_tr,batch)
    val_loader   = make_loader(Xc_val,Xs_val,y_val,batch)

    # --- Build nets -----------------------------------------
    cpu_net = create_cpu_head(in_cpu=X_cpu.shape[1], hidden=hidden)
    scr_net = create_screen_head(in_scr=X_scr.shape[1], hidden=hidden)

    cpu_net.load_state_dict(torch.load(cpu_model_path, map_location="cpu"))
    scr_net.load_state_dict(torch.load(scr_model_path, map_location="cpu"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cpu_net, scr_net = cpu_net.to(device), scr_net.to(device)

    params = list(cpu_net.parameters())+list(scr_net.parameters())
    opt    = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    crit   = nn.L1Loss()

    best_val, patience, PATIENCE=1e9,0,5
    for ep in trange(epochs, desc=f"FT {tag}"):
        cpu_net.train(); scr_net.train()
        for Xc_b,Xs_b,y_b in train_loader:
            Xc_b,Xs_b,y_b = Xc_b.to(device),Xs_b.to(device),y_b.to(device)
            opt.zero_grad()
            y_hat = cpu_net(Xc_b)+scr_net(Xs_b)
            loss  = crit(y_hat, y_b)
            loss.backward(); opt.step()
        # --- quick val
        cpu_net.eval(); scr_net.eval()
        with torch.no_grad():
            val_l=[]
            for Xc_b,Xs_b,y_b in val_loader:
                Xc_b,Xs_b,y_b = Xc_b.to(device),Xs_b.to(device),y_b.to(device)
                y_hat = cpu_net(Xc_b)+scr_net(Xs_b)
                val_l.append(crit(y_hat,y_b).item())
            val_mean = np.mean(val_l)
        if val_mean<best_val-1e-4:
            best_val=val_mean; patience=0
        else:
            patience+=1
            if patience>PATIENCE: break

    # --- save ------------------------------------------------
    cpu_out = os.path.join(out_dir,f"cpu_model_ft_{tag}.pt")
    scr_out = os.path.join(out_dir,f"screen_model_ft_{tag}.pt")
    torch.save(cpu_net.cpu().state_dict(), cpu_out)
    torch.save(scr_net.cpu().state_dict(), scr_out)
    print(f"Saved {cpu_out}  &  {scr_out}")
    return {"cpu":cpu_out,"scr":scr_out,"tag":tag}

# -------------------- CLI -----------------------------------
if __name__=="__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--train_csv",required=True)
    pa.add_argument("--cpu_model",required=True)
    pa.add_argument("--screen_model",required=True)
    pa.add_argument("--scalers",required=True)
    pa.add_argument("--meta_params",required=True)
    pa.add_argument("--fraction",type=float,required=True,help="0.1 … 0.8")
    pa.add_argument("--epochs",type=int,required=True)
    pa.add_argument("--hidden",type=int,default=256)
    pa.add_argument("--batch",type=int,default=128)
    pa.add_argument("--lr",type=float,default=5e-4)
    pa.add_argument("--seed",type=int,default=42)
    pa.add_argument("--out_dir",default="finetuned_models")
    args = pa.parse_args()
    finetune(args.train_csv,args.cpu_model,args.screen_model,
             args.scalers,args.meta_params,
             args.fraction,args.epochs,args.lr,args.batch,
             args.hidden,args.seed,args.out_dir)
