# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
# All rights reserved.
# Unauthorized copying, modification, distribution prohibited without prior written consent.


# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
import numpy as np
from scipy.optimize import differential_evolution

class HarmonicModel:
    """
    v_obs^2 = v_bary^2 + sum_i [ (Q_i * v_bary * sin(2π r / λ_i))^2 ]
    """
    def __init__(self, lambdas_kpc):
        self.lambdas = np.asarray(lambdas_kpc, dtype=float)

    def predict(self, r_kpc, v_bary, Q):
        v2 = v_bary**2
        for lam, qi in zip(self.lambdas, Q):
            phase = 2*np.pi*r_kpc/lam
            v2 += (qi * v_bary * np.sin(phase))**2
        return np.sqrt(np.maximum(v2, 0.0))

    def fit_single_galaxy(self, gal, bounds=(0,1.5)):
        r, vb = gal.r, gal.v_bary
        vo, ve = gal.v_obs, np.clip(gal.v_err, 1e-3, None)

        def chi2(qvec):
            v = self.predict(r, vb, qvec)
            return np.sum(((vo - v)/ve)**2)

        res = differential_evolution(
            chi2, bounds=[bounds]*len(self.lambdas),
            seed=42, maxiter=500, atol=1e-6, tol=1e-6
        )
        chi2_fit = float(res.fun)
        chi2_base = float(np.sum(((vo - vb)/ve)**2))
        imp = 100.0*(chi2_base - chi2_fit)/max(chi2_base, 1e-9)
        return dict(Q=res.x.tolist(), chi2_fit=chi2_fit,
                    chi2_base=chi2_base, improvement_pct=imp)
