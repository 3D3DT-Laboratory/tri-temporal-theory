# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
# All rights reserved.
# Unauthorized copying, modification, distribution prohibited without prior written consent.

import numpy as np

def newtonian_g(V_kms: np.ndarray, R_kpc: np.ndarray) -> np.ndarray:
    """
    g = V^2 / R  con V in km/s, R in kpc → g in m/s^2
    """
    V = np.asarray(V_kms, float) * 1e3      # km/s → m/s
    R = np.asarray(R_kpc, float) * 3.085677581e19  # kpc → m
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (V**2) / R
    return g


def v_circ_newtonian(M_enclosed, R_m, G=6.67430e-11):
    return np.sqrt(G * M_enclosed / np.clip(R_m, 1e-12, None))

def mond_mu(x):
    # Simple standard interpolation function
    return x / (1.0 + x)

def v_circ_mond(g_bar, a0=1.2e-10):
    # g_obs = g_bar / mu(g_obs/a0) -> approximate closed form for plotting
    g = 0.5 * (g_bar + np.sqrt(g_bar**2 + 4*g_bar*a0))
    return g
