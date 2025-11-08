# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
# All rights reserved.
# Unauthorized copying, modification, distribution prohibited without prior written consent.

"""
fft_rar.py — Analisi armoniche/FFT dei residui RAR o curve di rotazione.

Obiettivo
---------
Rilevare scale armoniche (es. λ_b ≈ 4.3 kpc e sotto/sovra-armoniche)
previste dal framework 3D+3D, tramite FFT dei residui vs r per galassia.

Input atteso
------------
CSV creato dalla tua pipeline (es. outputs/rar/rar_data.csv) con colonne:
    - galaxy (str)             [opzionale ma consigliato]
    - r_kpc (float)            [consigliato per λ in kpc]
    - gbar, gobs (float, SI)   [RAR in m/s²]
    - (opz.) residual/resid    [se già calcolati]

Uso (esempi)
------------
python -m src.analysis.fft_rar --input outputs/rar/rar_data.csv --per-galaxy
python -m src.analysis.fft_rar --input outputs/rar/rar_data.csv --outdir outputs/harmonics --lambda-max 30.0 --min-peak-snr 3.0 --per-galaxy

Output
------
- <outdir>/harmonic_summary.csv
- <outdir>/fft_<GALAXY>.png       (se --per-galaxy)
- <outdir>/fft_global.png
- <outdir>/diagnostic_log.txt

Note
----
* Usiamo residui in log10(g) per stabilizzare la dinamica:
    resid = log10(gobs) - log10(gbar)  (se non forniti)
* Se r_kpc è presente e quasi uniforme, stimiamo una λ fisica (kpc) via passo medio.
"""

from __future__ import annotations
import argparse, os
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq

plt.rcParams.update({
    "figure.figsize": (7, 4),
    "figure.dpi": 110,
    "axes.grid": True,
    "font.size": 10,
})

# -----------------------------
# Utility
# -----------------------------
def safe_log10(x: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    return np.log10(np.clip(x, eps, None))

def compute_residuals(df: pd.DataFrame) -> np.ndarray:
    """Preferisce colonne 'residual'/'resid' se presenti, altrimenti log10(gobs)-log10(gbar)."""
    for col in ("residual", "resid", "log_residual"):
        if col in df.columns:
            arr = df[col].astype(float).values
            return arr[np.isfinite(arr)]
    if not {"gbar", "gobs"}.issubset(df.columns):
        raise RuntimeError("CSV deve contenere 'gbar' e 'gobs' o una colonna residuo ('residual'/'resid').")
    gbar = df["gbar"].astype(float).values
    gobs = df["gobs"].astype(float).values
    res = safe_log10(gobs) - safe_log10(gbar)
    res = res[np.isfinite(res)]
    return res

def fft_series(y: np.ndarray, r_vec: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[float]]:
    """FFT (ortho). Restituisce (freq_indice, power, dr_medio_se_r_kpc_uniforme)."""
    x = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 8:
        return np.array([]), np.array([]), None

    # r spacing (approssimazione: medio su punti ordinati)
    dr = None
    if r_vec is not None:
        rr = np.asarray(r_vec, dtype=float)
        rr = rr[np.isfinite(rr)]
        if len(rr) >= len(x):
            rr = np.sort(rr)[:len(x)]
        if len(rr) == len(x) and len(rr) > 3:
            diffs = np.diff(np.sort(rr))
            if np.all(np.isfinite(diffs)) and np.nanmean(diffs) > 0:
                dr = float(np.nanmean(diffs))

    x = x - np.nanmean(x)
    spec = rfft(x, norm="ortho")
    power = np.abs(spec) ** 2
    freqs = rfftfreq(len(x), d=1.0)  # frequenze in unità indice
    return freqs, power, dr

def pick_peaks(freqs: np.ndarray, power: np.ndarray, min_snr: float = 3.0) -> pd.DataFrame:
    """Selezione massimi locali con SNR (MAD-based)."""
    if len(freqs) == 0:
        return pd.DataFrame(columns=["index","freq_index","power","snr"])
    med = np.median(power)
    mad = np.median(np.abs(power - med)) + 1e-12
    snr = (power - med) / mad
    idxs = []
    for i in range(1, len(power) - 1):
        if power[i] > power[i-1] and power[i] > power[i+1] and snr[i] >= min_snr:
            idxs.append(i)
    rows = [{
        "index": int(i),
        "freq_index": float(freqs[i]),
        "power": float(power[i]),
        "snr": float(snr[i]),
    } for i in idxs]
    return pd.DataFrame(rows).sort_values("snr", ascending=False).reset_index(drop=True)

def estimate_lambda_kpc(top_index: int, n_samples: int, dr_kpc: Optional[float]) -> float:
    """Stima λ (kpc) dal picco dominante se il passo r (dr_kpc) è disponibile."""
    if dr_kpc is None or top_index <= 0 or n_samples <= 0:
        return float("nan")
    # Periodo in 'indici' ~ n / i  (approssimazione semplice)
    index_period = n_samples / top_index
    return float(index_period * dr_kpc)

def plot_spectrum(out_png: str, freqs: np.ndarray, power: np.ndarray, title: str):
    plt.figure()
    plt.plot(freqs, power)
    plt.xlabel("Index frequency (a.u.)")
    plt.ylabel("Power")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

# -----------------------------
# Main
# -----------------------------
def run(input_csv: str, outdir: str, lambda_max_kpc: float, min_peak_snr: float, per_galaxy: bool):
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, "diagnostic_log.txt")
    summ_path = os.path.join(outdir, "harmonic_summary.csv")
    glob_png = os.path.join(outdir, "fft_global.png")

    with open(log_path, "w", encoding="utf-8") as LOG:
        LOG.write("[*] Avvio FFT/Harmonics\n")
        df = pd.read_csv(input_csv)
        needed = {"gbar", "gobs"}
        if not needed.issubset(df.columns):
            raise RuntimeError("Input deve avere colonne: gbar, gobs (e idealmente galaxy, r_kpc).")

        # Spettro globale (sanity check)
        freqs, power, _ = fft_series(
            compute_residuals(df),
            r_vec=df["r_kpc"].values if "r_kpc" in df.columns else None
        )
        if len(freqs) > 0:
            plot_spectrum(glob_png, freqs, power, title="FFT — Global residuals")
        else:
            LOG.write("[WARN] Campioni insufficienti per FFT globale.\n")

        rows = []
        if per_galaxy and "galaxy" in df.columns:
            for name, g in df.groupby("galaxy"):
                try:
                    res = compute_residuals(g)
                    if len(res) < 8:
                        continue
                    r_vec = g["r_kpc"].values if "r_kpc" in g.columns else None
                    f, P, dr = fft_series(res, r_vec=r_vec)
                    if len(f) == 0:
                        continue
                    peaks = pick_peaks(f, P, min_snr=min_peak_snr)
                    # salva grafico
                    out_png = os.path.join(outdir, f"fft_{str(name).replace('/', '_')}.png")
                    plot_spectrum(out_png, f, P, title=f"FFT — {name}")

                    lam_kpc = float("nan")
                    if len(peaks) > 0:
                        idx = int(peaks.iloc[0]["index"])
                        lam_kpc = estimate_lambda_kpc(idx, n_samples=len(res), dr_kpc=dr)

                    top_snr = float(peaks.iloc[0]["snr"]) if len(peaks) > 0 else float("nan")
                    rows.append({
                        "galaxy": name,
                        "n_points": len(res),
                        "top_index": int(peaks.iloc[0]["index"]) if len(peaks) > 0 else -1,
                        "top_snr": top_snr,
                        "lambda_kpc_est": lam_kpc,
                    })
                except Exception as e:
                    LOG.write(f"[WARN] {name}: {e}\n")

        pd.DataFrame(rows).sort_values("top_snr", ascending=False).to_csv(summ_path, index=False)
        LOG.write(f"[OK] Scritto: {summ_path}\n")
        LOG.write(f"[OK] Grafici in: {outdir}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV con gbar,gobs e opz. galaxy,r_kpc")
    ap.add_argument("--outdir", default="outputs/harmonics", help="Cartella output")
    ap.add_argument("--lambda-max", type=float, default=30.0, help="Max λ kpc (solo annotazioni future)")
    ap.add_argument("--min-peak-snr", type=float, default=3.0, help="SNR minima per accettare il picco")
    ap.add_argument("--per-galaxy", action="store_true", help="FFT per galassia se 'galaxy' esiste")
    args = ap.parse_args()
    run(args.input, args.outdir, args.lambda_max, args.min_peak_snr, args.per_galaxy)

if __name__ == "__main__":
    main()
