# ===========================================================
#  TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
#  Unauthorized copying of this file, via any medium, is strictly prohibited.
#  Proprietary and confidential. All rights reserved.
# ===========================================================

from __future__ import annotations
import argparse, pathlib, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from src.data_io import load_rotmod_zip, rotcurves_to_dataframe
from src.utils import KPC, KM, G, safe_log10

def mond_like(g_bar, g0):
    return g_bar / (1.0 - np.exp(-np.sqrt(np.maximum(g_bar, 0) / g0)))

def compute_rar(rotmod_zip: str, outdir: str = "outputs/rar"):
    outpath = pathlib.Path(outdir)
    outpath.mkdir(parents=True, exist_ok=True)

    curves = load_rotmod_zip(rotmod_zip)
    df = rotcurves_to_dataframe(curves)

    R_m = df["R_kpc"].to_numpy() * KPC
    V_m_s = df["V_kms"].to_numpy() * KM
    g_obs = (V_m_s ** 2) / R_m

    M_bar = (V_m_s**2 * R_m) / G
    g_bar = G * M_bar / (R_m ** 2)

    mask = np.isfinite(g_obs) & np.isfinite(g_bar) & (g_obs > 0) & (g_bar > 0)
    g_obs = g_obs[mask]; g_bar = g_bar[mask]

    log_g_obs = np.log10(g_obs)
    log_g_bar = np.log10(g_bar)

    popt, pcov = curve_fit(mond_like, g_bar, g_obs, p0=[1.2e-10], maxfev=20000)
    g0 = float(popt[0])

    g_fit = mond_like(g_bar, g0)
    residuals = np.log10(g_obs) - np.log10(g_fit)
    r2 = float(1 - np.var(residuals) / np.var(np.log10(g_obs)))
    corr = float(np.corrcoef(log_g_bar, log_g_obs)[0, 1])

    summary = {
        "n_points": int(len(g_obs)),
        "g0_bestfit_m_s2": g0,
        "r2_log": r2,
        "corr_log10": corr
    }

    df_out = pd.DataFrame({
        "g_bar_m_s2": g_bar,
        "g_obs_m_s2": g_obs,
        "log_g_bar": log_g_bar,
        "log_g_obs": log_g_obs,
        "residuals_log10": residuals
    })

    out_csv = outpath / "rar_data.csv"
    out_json = outpath / "rar_summary.json"
    out_png  = outpath / "rar_plot.png"

    df_out.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plt.figure(figsize=(6,5))
    plt.scatter(log_g_bar, log_g_obs, s=8, alpha=0.4, label="SPARC data")
    plt.plot(np.log10(g_bar), np.log10(g_fit), lw=2, label=f"Fit MOND-like, g₀={g0:.2e}")
    plt.xlabel(r"log$_{10}$(g$_{bar}$) [m/s²]")
    plt.ylabel(r"log$_{10}$(g$_{obs}$) [m/s²]")
    plt.title("Radial Acceleration Relation (RAR)")
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    return summary, str(out_csv), str(out_json), str(out_png)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rotmod", type=str, default="data/Rotmod_LTG.zip")
    ap.add_argument("--outdir", type=str, default="outputs/rar")
    args = ap.parse_args()
    info, csv, js, png = compute_rar(args.rotmod, args.outdir)
    print("RAR computed successfully:\n", json.dumps(info, indent=2))
    print("Outputs:\n ", csv, "\n ", js, "\n ", png)

if __name__ == "__main__":
    main()
