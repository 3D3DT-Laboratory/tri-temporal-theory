# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
# All rights reserved.
# Unauthorized copying, modification, distribution prohibited without prior written consent.

from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from ..data_io_sparc import load_rotmod_archive, peek_rotmod_archive
from ..models.baselines import newtonian_g
from ..utils import ensure_dir, save_json

def compute_rar(df: pd.DataFrame) -> pd.DataFrame:
    g_obs = newtonian_g(df["V_obs"].to_numpy(), df["R_kpc"].to_numpy())
    out = pd.DataFrame({
        "galaxy_id": df["galaxy_id"],
        "R_kpc": df["R_kpc"],
        "V_obs": df["V_obs"],
        "g_obs": g_obs,
    })
    if "V_bar" in df:
        out["V_bar"] = df["V_bar"]
        out["g_bar"] = newtonian_g(df["V_bar"].to_numpy(), df["R_kpc"].to_numpy())
    return out

def plot_rar(df_rar: pd.DataFrame, out_png: Path):
    plt.figure(figsize=(6,5))
    x = df_rar["g_bar"] if "g_bar" in df_rar else df_rar["g_obs"]
    y = df_rar["g_obs"]
    plt.loglog(x, y, "o", ms=2, alpha=0.5)
    xs = np.logspace(-13, -8, 200)
    plt.loglog(xs, xs, "--", label="g_obs=g_bar")
    a0 = 1.2e-10
    plt.loglog(xs, np.sqrt(a0*xs), "-", label="MOND-like")
    plt.xlabel("g_bar  [m s⁻²]" if "g_bar" in df_rar else "g_obs  [m s⁻²]")
    plt.ylabel("g_obs  [m s⁻²]")
    plt.legend(fontsize=8)
    plt.title("Radial Acceleration Relation (SPARC-like)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def summarize_rar(df_rar: pd.DataFrame) -> dict:
    if "g_bar" in df_rar:
        x = np.log10(df_rar["g_bar"])
    else:
        x = np.log10(df_rar["g_obs"])
    y = np.log10(df_rar["g_obs"])
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    r = np.corrcoef(x, y)[0,1]
    return {"points": int(len(df_rar)), "log10_slope": float(slope), "log10_intercept": float(intercept), "pearson_r": float(r)}

def main(rotmod_archive: str, outdir: str = "outputs/rar"):
    outdir = ensure_dir(outdir)
    print(f"[*] Analisi archivio: {rotmod_archive}")
    peek_rotmod_archive(rotmod_archive)
    df = load_rotmod_archive(rotmod_archive)
    rar = compute_rar(df)
    rar_csv = Path(outdir) / "rar_data.csv"
    rar_png = Path(outdir) / "rar_plot.png"
    rar_json = Path(outdir) / "rar_summary.json"
    rar.to_csv(rar_csv, index=False)
    plot_rar(rar, rar_png)
    save_json(summarize_rar(rar), rar_json)
    print(f"[OK] Salvati: {rar_csv}, {rar_png}, {rar_json}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rotmod", required=True)
    ap.add_argument("--outdir", default="outputs/rar")
    args = ap.parse_args()
    main(args.rotmod, outdir=args.outdir)

