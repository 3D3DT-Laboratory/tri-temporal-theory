# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
# All rights reserved.
# Unauthorized copying, modification, distribution prohibited without prior written consent.

from __future__ import annotations
from pathlib import Path
import json
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from ..utils import ensure_dir

def _fft_one(radius_kpc: np.ndarray, v_resid: np.ndarray):
    # semplice PSD vs lambda
    r = np.asarray(radius_kpc, float)
    v = np.asarray(v_resid, float)
    if len(r) < 6:
        return None
    # resample su grid uniforme in r per FFT semplice
    rmin, rmax = np.nanmin(r), np.nanmax(r)
    grid = np.linspace(rmin, rmax, 256)
    v_grid = np.interp(grid, r, v)
    V = np.fft.rfft(v_grid - np.nanmean(v_grid))
    k = np.fft.rfftfreq(len(grid), d=(grid[1]-grid[0]))  # cicli/kpc
    # Evita k=0; converti in lambda = 1/k  (in kpc)
    mask = k > 0
    k = k[mask]
    psd = (np.abs(V[mask])**2)
    lam = 1.0 / k
    return lam, psd

def fft_residuals(rar_csv: str, outdir: str, lambda_max_kpc: float = 30.0):
    outdir = ensure_dir(outdir)
    df = pd.read_csv(rar_csv)
    # residuo semplice: V_obs - V_bar (se disponibile)
    if "V_bar" in df.columns:
        df["V_resid"] = df["V_obs"] - df["V_bar"]
    else:
        # Se non c'è V_bar, togliamo media mobile per rimuovere trend
        df = df.sort_values(["galaxy_id","R_kpc"])
        df["V_resid"] = df.groupby("galaxy_id")["V_obs"].transform(lambda x: x - x.rolling(7, min_periods=1, center=True).mean())

    peaks_all = []
    for gid, g in df.groupby("galaxy_id"):
        res = _fft_one(g["R_kpc"].values, g["V_resid"].values)
        if res is None:
            continue
        lam, psd = res
        m = lam <= lambda_max_kpc
        lam, psd = lam[m], psd[m]
        if len(lam) == 0:
            continue
        # salva top-3 picchi
        idx = np.argsort(psd)[::-1][:3]
        for i in idx:
            peaks_all.append({"galaxy_id": gid, "lambda_kpc": float(lam[i]), "power": float(psd[i])})

    if not peaks_all:
        print("[WARN] Nessun picco FFT rilevato; controlla input.")
        # Non alziamo eccezione per compatibilità
    out_json = Path(outdir) / "fft_peaks.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(peaks_all, f, indent=2)
    print(f"[OK] Salvato: {out_json}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV da rar_relation.py (rar_data.csv)")
    ap.add_argument("--outdir", default="outputs/fft")
    ap.add_argument("--lambda-max", type=float, default=30.0)

    # Flag storici (ignorati in sicurezza se passati da vecchi script)
    ap.add_argument("--min-peak-snr", type=float, default=None)
    ap.add_argument("--allow-empty", action="store_true")

    args = ap.parse_args()
    fft_residuals(args.input, outdir=args.outdir, lambda_max_kpc=args.lambda_max)
