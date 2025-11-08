# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
# All rights reserved. Unauthorized copying, modification, or distribution is prohibited.

import argparse, json, os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import probplot

# ----------------------------
# Costanti/Default
# ----------------------------
G0_DEFAULT = 1.20e-10  # m s^-2  (pivot RAR)
RNG = np.random.default_rng(42)

# ----------------------------
# Utilità
# ----------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def safe_log10(x):
    x = np.asarray(x, float)
    x = np.clip(x, 1e-40, None)
    return np.log10(x)

def mcgaugh_sigma_dex(loggbar):
    """
    Stima eteroschedastica delle incertezze in dex (≈ McGaugh16):
    ~0.09 dex a g_bar alto → ~0.20 dex a g_bar basso.
    """
    # più scatter sotto g0
    scatter = 0.12 + 0.08 * np.clip((np.log10(G0_DEFAULT) - loggbar) / 2.5, 0, 1)
    return np.clip(scatter, 0.09, 0.20)

def weighted_metrics(y, yhat, w):
    resid = y - yhat
    # χ^2_red con dof = N - k (k assegnato dal chiamante)
    chi2 = np.sum(w * resid**2)
    # R^2 pesato (definizione con media pesata)
    ybar = np.sum(w * y) / np.sum(w)
    ss_res = np.sum(w * (y - yhat) ** 2)
    ss_tot = np.sum(w * (y - ybar) ** 2)
    r2w = 1.0 - ss_res / max(ss_tot, 1e-30)
    # RMS (non pesato in dex, giusto per confronto)
    rms = np.sqrt(np.mean(resid**2))
    return chi2, r2w, rms, resid

def aic_bic(chi2, n, k):
    # log-likelihood in gaussiano: lnL = -0.5 * chi2 + const → costanti si cancellano nei Δ
    aic = chi2 + 2 * k
    bic = chi2 + k * np.log(n)
    return float(aic), float(bic)

# ----------------------------
# Modelli in log-spazio
# ----------------------------
def lcdm_log_model(x_log, params):
    # y = A + B x  (power-law in log-log)
    A, B = params
    return A + B * x_log

def mond_gobs(gbar, a0):
    # Forma standard: g = (gbar/2)*(1 + sqrt(1 + 4 a0 / gbar))
    return 0.5 * gbar * (1.0 + np.sqrt(1.0 + 4.0 * a0 / np.clip(gbar, 1e-40, None)))

def threeD3D_log_model(x_log, g0, gamma):
    """
    3D+3D power-law interpolation in log-space:
    
    log g_obs = (1 - gamma) * log g0 + gamma * log g_bar
    
    Equivalent to: g_obs = g0^(1-gamma) * g_bar^gamma
    
    This smoothly interpolates between:
    - Low g_bar: g_obs ~ g0 * (g_bar/g0)^gamma (MOND-like)
    - High g_bar: g_obs ~ g_bar (Newtonian)
    
    Expected gamma ~ 0.6-0.7 for best RAR fit.
    """
    logg0 = np.log10(g0)
    return (1.0 - gamma) * logg0 + gamma * x_log

# ----------------------------
# Fit pesati (log-spazio)
# ----------------------------
def fit_lcdm(x_log, y_log, w):
    # chi2(A,B) = sum w (y - (A + Bx))^2 → soluzione WLS chiusa
    W = np.diag(w)
    X = np.vstack([np.ones_like(x_log), x_log]).T
    try:
        beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y_log)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y_log)
    A, B = beta
    yhat = lcdm_log_model(x_log, (A, B))
    return (A, B), yhat, 2  # k=2

def fit_mond(x_log, y_log, w):
    x_lin = 10 ** x_log

    def obj(theta):
        a0 = 10 ** theta[0]
        yhat = safe_log10(mond_gobs(x_lin, a0))
        res = y_log - yhat
        return np.sum(w * res * res)

    theta0 = np.array([np.log10(1.2e-10)])
    out = minimize(obj, theta0, method="Nelder-Mead")
    a0 = 10 ** out.x[0]
    yhat = safe_log10(mond_gobs(x_lin, a0))
    return (a0,), yhat, 1  # k=1

def fit_3d3d(x_log, y_log, w, fit_g0=False):
    # prior morbida su g0 se fit_g0: log10 g0 ~ N(log10(G0_DEFAULT), 0.15^2)
    def obj(theta):
        if fit_g0:
            logg0, gamma = theta
            g0 = 10 ** logg0
            yhat = threeD3D_log_model(x_log, g0, gamma)
            res = y_log - yhat
            chi2 = np.sum(w * res * res)
            # prior su logg0 (σ=0.15 dex)
            chi2_prior = ((logg0 - np.log10(G0_DEFAULT)) / 0.15) ** 2
            return chi2 + chi2_prior
        else:
            gamma = theta[0]
            g0 = G0_DEFAULT
            yhat = threeD3D_log_model(x_log, g0, gamma)
            res = y_log - yhat
            return np.sum(w * res * res)

    if fit_g0:
        x0 = np.array([np.log10(G0_DEFAULT), 0.68])
    else:
        x0 = np.array([0.68])

    out = minimize(obj, x0, method="Nelder-Mead")
    if fit_g0:
        logg0, gamma = out.x
        g0 = 10 ** logg0
    else:
        g0, gamma = G0_DEFAULT, out.x[0]

    yhat = threeD3D_log_model(x_log, g0, gamma)
    k = 2 if fit_g0 else 1
    return (g0, gamma), yhat, k

# ----------------------------
# Plot
# ----------------------------
def plot_main(x_log, y_log, yhat_lcdm, yhat_mond, yhat_3d3d, params_lcdm, params_mond, params_3d3d, out_png):
    plt.figure(figsize=(9, 6))
    plt.scatter(x_log, y_log, s=8, alpha=0.35)
    A, B = params_lcdm
    a0 = params_mond[0]
    g0, gamma = params_3d3d

    xs = np.linspace(np.min(x_log)-0.2, np.max(x_log)+0.2, 400)
    plt.plot(xs, lcdm_log_model(xs, (A, B)), label=rf"ΛCDM: $y={A:.3f}+{B:.3f}\,x$", lw=2)
    # MOND curve su xs: calcolo esatto in lin-space per la linea liscia
    plt.plot(xs, safe_log10(mond_gobs(10**xs, a0)), label=rf"MOND: $a_0={a0:.2e}\,{{\rm m\,s^{{-2}}}}$", lw=2)
    plt.plot(xs, threeD3D_log_model(xs, g0, gamma), label=rf"3D+3D: $g_0={g0:.2e}$, $\gamma={gamma:.3f}$", lw=2)

    plt.xlabel(r"$\log_{10}\, g_{\rm bar}\,\,({\rm m\,s^{-2}})$")
    plt.ylabel(r"$\log_{10}\, g_{\rm obs}\,\,({\rm m\,s^{-2}})$")
    plt.title("RAR — log-space robust weighted fits")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_residuals_binned(x_log, resid, out_png, nbins=12):
    bins = np.linspace(np.min(x_log), np.max(x_log), nbins+1)
    idx = np.digitize(x_log, bins) - 1
    centers, med, p16, p84 = [], [], [], []
    for b in range(nbins):
        m = (idx == b)
        if not np.any(m): 
            continue
        r = resid[m]
        centers.append(0.5*(bins[b]+bins[b+1]))
        med.append(np.median(r))
        p16.append(np.percentile(r, 16))
        p84.append(np.percentile(r, 84))
    centers, med, p16, p84 = map(np.asarray, (centers, med, p16, p84))
    plt.figure(figsize=(8.8, 5.6))
    plt.axhline(0, color='k', lw=1, ls='--', alpha=0.6)
    plt.errorbar(centers, med, yerr=[med-p16, p84-med], fmt='o-', lw=1.5, ms=4)
    plt.xlabel(r"$\log_{10}\, g_{\rm bar}$")
    plt.ylabel("resid (dex)")
    plt.title("RAR residuals (median ±68%)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_residuals_qq(resid, out_png):
    plt.figure(figsize=(6.2, 6.2))
    probplot(resid, dist="norm", plot=plt)
    plt.title("Residuals QQ-plot")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

# ----------------------------
# Main
# ----------------------------
def main(rar_csv, outdir, sigma_int=0.0, fit_g0=False):
    ensure_dir(outdir)
    df = pd.read_csv(rar_csv)
    # compatibilità vecchi nomi
    gbar = df['g_bar'].values if 'g_bar' in df.columns else df['gbar'].values
    gobs = df['g_obs'].values if 'g_obs' in df.columns else df['gobs'].values

    x_log = safe_log10(gbar)
    y_log = safe_log10(gobs)

    # pesi eteroschedastici (dex) + scatter intrinseco
    sigma_dex = mcgaugh_sigma_dex(x_log)
    sigma_eff = np.sqrt(sigma_dex**2 + float(sigma_int)**2)
    w = 1.0 / np.clip(sigma_eff, 1e-6, None)**2

    # Fit
    (A, B), yhat_lcdm, k_lcdm = fit_lcdm(x_log, y_log, w)
    (a0,), yhat_mond, k_mond = fit_mond(x_log, y_log, w)
    (g0, gamma), yhat_3d3d, k_3d3d = fit_3d3d(x_log, y_log, w, fit_g0=fit_g0)

    # Metriche
    n = len(x_log)
    chi2_l, r2w_l, rms_l, resid_l = weighted_metrics(y_log, yhat_lcdm, w)
    chi2_m, r2w_m, rms_m, resid_m = weighted_metrics(y_log, yhat_mond, w)
    chi2_t, r2w_t, rms_t, resid_t = weighted_metrics(y_log, yhat_3d3d, w)

    # χ2_red
    chi2red_l = chi2_l / max(n - k_lcdm, 1)
    chi2red_m = chi2_m / max(n - k_mond, 1)
    chi2red_t = chi2_t / max(n - k_3d3d, 1)

    # AIC/BIC
    aic_l, bic_l = aic_bic(chi2_l, n, k_lcdm)
    aic_m, bic_m = aic_bic(chi2_m, n, k_mond)
    aic_t, bic_t = aic_bic(chi2_t, n, k_3d3d)

    # Plot principale + diagnostiche
    plot_main(x_log, y_log, yhat_lcdm, yhat_mond, yhat_3d3d,
              (A, B), (a0,), (g0, gamma),
              os.path.join(outdir, "rar_fit_logspace.png"))
    plot_residuals_binned(x_log, resid_t, os.path.join(outdir, "rar_fit_logspace_residuals_binned.png"))
    plot_residuals_qq(resid_t, os.path.join(outdir, "rar_fit_logspace_residuals_qq.png"))

    # JSON
    out = {
        "n_points": int(n),
        "sigma_int_dex": float(sigma_int),
        "models": {
            "LCDM": {
                "params": {"A": float(A), "B": float(B)},
                "chi2": float(chi2_l), "chi2_red": float(chi2red_l),
                "R2_w": float(r2w_l), "RMS_dex": float(rms_l),
                "k": k_lcdm, "AIC": aic_l, "BIC": bic_l
            },
            "MOND": {
                "params": {"a0": float(a0)},
                "chi2": float(chi2_m), "chi2_red": float(chi2red_m),
                "R2_w": float(r2w_m), "RMS_dex": float(rms_m),
                "k": k_mond, "AIC": aic_m, "BIC": bic_m
            },
            "3D3D": {
                "params": {"g0": float(g0), "gamma": float(gamma), "fit_g0": bool(fit_g0)},
                "chi2": float(chi2_t), "chi2_red": float(chi2red_t),
                "R2_w": float(r2w_t), "RMS_dex": float(rms_t),
                "k": k_3d3d, "AIC": aic_t, "BIC": bic_t
            }
        }
    }
    with open(os.path.join(outdir, "comparison_logspace.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # Report sintetico a console
    print("\n======================================================================")
    print("RESULTS (weighted log-space)")
    print("======================================================================")
    print(f"σ_int (dex): {sigma_int:.3f}   N: {n}")
    print("Model         χ²_red    R²_w     RMS   AIC      BIC")
    print("---------------------------------------------------------------")
    print(f"ΛCDM      {chi2red_l:8.3f}  {r2w_l:6.3f}  {rms_l:5.3f}  {aic_l:7.1f}  {bic_l:7.1f}")
    print(f"MOND      {chi2red_m:8.3f}  {r2w_m:6.3f}  {rms_m:5.3f}  {aic_m:7.1f}  {bic_m:7.1f}")
    print(f"3D+3D     {chi2red_t:8.3f}  {r2w_t:6.3f}  {rms_t:5.3f}  {aic_t:7.1f}  {bic_t:7.1f}")
    print("---------------------------------------------------------------")
    print(f"Saved: {os.path.join(outdir, 'comparison_logspace.json')}")
    print(f"Plot : {os.path.join(outdir, 'rar_fit_logspace.png')}")
    print(f"Diag : {os.path.join(outdir, 'rar_fit_logspace_residuals_binned.png')}")
    print(f"Diag : {os.path.join(outdir, 'rar_fit_logspace_residuals_qq.png')}")
    print("======================================================================")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RAR log-space weighted fits (ΛCDM, MOND, 3D+3D) con AIC/BIC e diagnostiche.")
    ap.add_argument("--rar-csv", required=True, help="CSV con colonne g_bar/gbar e g_obs/gobs (m s^-2).")
    ap.add_argument("--outdir", required=True, help="Cartella output.")
    ap.add_argument("--sigma-int", type=float, default=0.0, help="Scatter intrinseco (dex) sommato in quadratura ai pesi.")
    ap.add_argument("--fit-g0", action="store_true", help="Se presente, g0 è lasciato libero (con prior morbida su log10 g0).")
    args = ap.parse_args()
    main(args.rar_csv, ensure_dir(args.outdir), sigma_int=args.sigma_int, fit_g0=args.fit_g0)
