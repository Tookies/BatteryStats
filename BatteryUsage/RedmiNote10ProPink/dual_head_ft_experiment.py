#!/usr/bin/env python
"""
dual_head_ft_experiment.py
==========================
Grid-runs fine-tuning on *one* target device with different
data fractions & epochs **and** instantly evaluates the new model
on *any* number of test CSVs (other devices).

Results → CSV table (fraction, epochs, test_device, MAE, MAPE).

Example
-------
python dual_head_ft_experiment.py \
    --target_csv ../RedmiNote10ProBlue/power_log_cleaned_.csv \
    --test_csvs ../RedmiNote10ProBlue/power_log_cleaned_.csv \
                ../RedmiNote10ProPink/power_log_cleaned_.csv \
                ../RedmiNote9/power_log_cleaned_.csv \
    --cpu_model cpu_model.pt --screen_model screen_model.pt \
    --scalers feature_scalers.pkl --meta_params meta_feature_params.pkl \
    --fractions 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 \
    --epochs_list 5 10 20 30 40 50 \
    --hidden 256 --out results_blue.csv
"""
import argparse, subprocess, json, csv, pathlib, os, sys, tempfile, pickle
import numpy as np, pandas as pd, torch, torch.nn as nn
from dual_head_finetune import finetune, compute_energy_mj, make_feature_split,\
                               make_cpu_meta_features, create_cpu_head, create_screen_head
# ----------------- evaluation helper ------------------------
def evaluate(csv_path:str, cpu_w:str, scr_w:str, scalers:str, hidden:int)->dict:
    df = pd.read_csv(csv_path)
    y_true = compute_energy_mj(df)
    cpu_df, scr_df = make_feature_split(df)
    with open(scalers,"rb") as f: scalers_obj = pickle.load(f)
    Xc = scalers_obj["cpu"].transform(make_cpu_meta_features(cpu_df))
    Xs = scalers_obj["screen"].transform(scr_df.values).astype("float32")
    cpu_net = create_cpu_head(in_cpu=Xc.shape[1], hidden=hidden)
    cpu_net.load_state_dict(torch.load(cpu_w, map_location="cpu")); cpu_net.eval()
    scr_net = create_screen_head(in_scr=Xs.shape[1], hidden=hidden)
    scr_net.load_state_dict(torch.load(scr_w, map_location="cpu")); scr_net.eval()
    with torch.no_grad():
        y_hat = (cpu_net(torch.from_numpy(Xc))+scr_net(torch.from_numpy(Xs))).numpy().squeeze()
    mae  = np.mean(np.abs(y_hat - y_true))
    mape = np.mean(np.abs(y_hat - y_true)/(y_true+1e-3))
    return {"MAE":mae,"MAPE":mape}

# ----------------- main grid loop ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_csv",required=True)
    ap.add_argument("--test_csvs",nargs="+",required=True)
    ap.add_argument("--cpu_model",required=True)
    ap.add_argument("--screen_model",required=True)
    ap.add_argument("--scalers",required=True)
    ap.add_argument("--meta_params",required=True)
    ap.add_argument("--fractions",nargs="+",type=float,default=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
    ap.add_argument("--epochs_list",nargs="+",type=int,default=[5,10,20,30,40,50])
    ap.add_argument("--hidden",type=int,default=128)
    ap.add_argument("--out",required=True)
    ap.add_argument("--lr",type=float,default=5e-4)
    args = ap.parse_args()

    results=[]
    for frac in args.fractions:
        for ep in args.epochs_list:
            ft_paths = finetune(args.target_csv,args.cpu_model,args.screen_model,
                                args.scalers,args.meta_params,
                                fraction=frac,epochs=ep,lr=args.lr,batch=128,
                                hidden=args.hidden,seed=42,out_dir="finetuned_models")
            for test_csv in args.test_csvs:
                metrics = evaluate(test_csv, ft_paths["cpu"], ft_paths["scr"],
                                   args.scalers,args.hidden)
                results.append({
                    "train_device": pathlib.Path(args.target_csv).parent.name,
                    "test_device" : pathlib.Path(test_csv).parent.name,
                    "fraction"    : frac,
                    "epochs"      : ep,
                    "MAE_mJ"      : round(metrics["MAE"],2),
                    "MAPE_pct"    : round(metrics["MAPE"]*100,2)
                })
                print(f"[{ft_paths['tag']}] → {metrics['MAPE']*100:.2f}% on {test_csv}")

    pd.DataFrame(results).to_csv(args.out,index=False)
    print(f"\nFull grid saved to {args.out}")

if __name__=="__main__":
    main()
