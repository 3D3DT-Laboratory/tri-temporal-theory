# ===========================================================
#  TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
#  Unauthorized copying of this file, via any medium, is strictly prohibited.
#  Proprietary and confidential. All rights reserved.
# ===========================================================

from __future__ import annotations
import numpy as np

def v_circ_newtonian(M_enclosed, R_m, G=6.67430e-11):
    return np.sqrt(G * M_enclosed / np.clip(R_m, 1e-12, None))

def mond_mu(x):
    # Simple standard interpolation function
    return x / (1.0 + x)

def v_circ_mond(g_bar, a0=1.2e-10):
    # g_obs = g_bar / mu(g_obs/a0) -> approximate closed form for plotting
    g = 0.5 * (g_bar + np.sqrt(g_bar**2 + 4*g_bar*a0))
    return g
