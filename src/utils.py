"""
Utility functions and constants for 3D+3D analysis.
"""

import numpy as np
from astropy import units as u
from astropy import constants as const

# Physical constants
G = const.G.to(u.kpc * u.km**2 / u.s**2 / u.Msun).value
C = const.c.to(u.km / u.s).value

# 3D+3D Theory parameters (from paper)
LAMBDA_B = 4.30  # kpc
LAMBDA_B_ERR = 0.15  # kpc
M_CRIT = 2.43e10  # M_sun
M_CRIT_ERR = 0.31e10  # M_sun
ALPHA = 0.34
ALPHA_ERR = 0.05
BETA = 0.28  # pulsar exponent
BETA_ERR = 0.09
TAU_B = 28.4  # years
TAU_B_ERR = 6.2  # years

# Q-field parameters (from M-theory derivation)
Q2 = 0.476
Q2_ERR = 0.034
Q3 = 0.184
Q3_ERR = 0.027


def newtonian_velocity(radius, mass_profile):
    """
    Compute Newtonian rotation velocity.
    
    Parameters:
    -----------
    radius : array-like
        Galactocentric radius [kpc]
    mass_profile : array-like
        Enclosed mass M(<r) [M_sun]
    
    Returns:
    --------
    velocity : array-like
        Rotation velocity [km/s]
    """
    radius = np.asarray(radius)
    mass_profile = np.asarray(mass_profile)
    
    # V = sqrt(G * M(<r) / r)
    velocity = np.sqrt(G * mass_profile / radius)
    
    return velocity


def q_field_correction(radius, mass_total, lambda_b=LAMBDA_B, Q2=Q2):
    """
    Compute Q-field correction to rotation velocity.
    
    Parameters:
    -----------
    radius : array-like
        Galactocentric radius [kpc]
    mass_total : float
        Total baryonic mass [M_sun]
    lambda_b : float
        Breathing scale [kpc]
    Q2 : float
        Q-field coupling constant
    
    Returns:
    --------
    delta_v : array-like
        Velocity correction [km/s]
    """
    radius = np.asarray(radius)
    
    # Amplitude scaling with mass
    A = amplitude_from_mass(mass_total)
    
    # Harmonic oscillation
    k_b = 2 * np.pi / lambda_b
    delta_v = A * np.sin(k_b * radius) * Q2
    
    return delta_v


def amplitude_from_mass(mass, alpha=ALPHA, M0=1e10):
    """
    Compute oscillation amplitude from mass.
    
    Parameters:
    -----------
    mass : float or array-like
        Baryonic mass [M_sun]
    alpha : float
        Power-law exponent
    M0 : float
        Normalization mass [M_sun]
    
    Returns:
    --------
    amplitude : float or array-like
        Oscillation amplitude [km/s]
    """
    sigma_0 = 10.0  # km/s at M0
    amplitude = sigma_0 * (mass / M0)**alpha
    
    return amplitude


def mass_suppression_factor(mass, M_crit=M_CRIT):
    """
    Suppression factor for M > M_crit.
    
    Parameters:
    -----------
    mass : float or array-like
        Baryonic mass [M_sun]
    M_crit : float
        Critical mass [M_sun]
    
    Returns:
    --------
    factor : float or array-like
        Suppression factor (0 to 1)
    """
    mass = np.asarray(mass)
    factor = 1.0 / (1.0 + (mass / M_crit)**2)
    
    return factor


def velocity_3D3D(radius, mass_profile, mass_total, 
                  lambda_b=LAMBDA_B, Q2=Q2):
    """
    Full 3D+3D velocity prediction.
    
    Parameters:
    -----------
    radius : array-like
        Galactocentric radius [kpc]
    mass_profile : array-like
        Enclosed mass M(<r) [M_sun]
    mass_total : float
        Total baryonic mass [M_sun]
    lambda_b : float
        Breathing scale [kpc]
    Q2 : float
        Q-field coupling
    
    Returns:
    --------
    velocity : array-like
        Total rotation velocity [km/s]
    """
    # Newtonian component
    v_newton = newtonian_velocity(radius, mass_profile)
    
    # Q-field correction
    delta_v_q = q_field_correction(radius, mass_total, lambda_b, Q2)
    
    # Mass suppression
    suppression = mass_suppression_factor(mass_total)
    
    # Total velocity (quadrature sum)
    velocity = np.sqrt(v_newton**2 + (suppression * delta_v_q)**2)
    
    return velocity


def chi_squared(observed, predicted, uncertainty):
    """
    Compute chi-squared statistic.
    
    Parameters:
    -----------
    observed : array-like
        Observed values
    predicted : array-like
        Model predictions
    uncertainty : array-like
        Uncertainties on observed values
    
    Returns:
    --------
    chi2 : float
        Chi-squared value
    """
    residuals = (observed - predicted) / uncertainty
    chi2 = np.sum(residuals**2)
    
    return chi2


def reduced_chi_squared(observed, predicted, uncertainty, n_params):
    """
    Compute reduced chi-squared.
    
    Parameters:
    -----------
    observed : array-like
        Observed values
    predicted : array-like
        Model predictions
    uncertainty : array-like
        Uncertainties
    n_params : int
        Number of model parameters
    
    Returns:
    --------
    chi2_red : float
        Reduced chi-squared
    """
    chi2 = chi_squared(observed, predicted, uncertainty)
    dof = len(observed) - n_params
    chi2_red = chi2 / dof
    
    return chi2_red


def bayes_factor(lnL1, lnL2):
    """
    Compute Bayes Factor.
    
    Parameters:
    -----------
    lnL1 : float
        Log-likelihood of model 1
    lnL2 : float
        Log-likelihood of model 2
    
    Returns:
    --------
    BF : float
        Bayes Factor (model 1 vs model 2)
    """
    delta_lnL = lnL1 - lnL2
    BF = np.exp(delta_lnL)
    
    return BF


def jeffreys_scale(BF):
    """
    Interpret Bayes Factor using Jeffreys' scale.
    
    Parameters:
    -----------
    BF : float
        Bayes Factor
    
    Returns:
    --------
    interpretation : str
        Verbal interpretation
    """
    log10_BF = np.log10(BF)
    
    if log10_BF < 0:
        return "Negative (favors model 2)"
    elif log10_BF < 0.5:
        return "Barely worth mentioning"
    elif log10_BF < 1.0:
        return "Substantial"
    elif log10_BF < 1.5:
        return "Strong"
    elif log10_BF < 2.0:
        return "Very strong"
    else:
        return "Decisive"


def read_sparc_galaxy(filename):
    """
    Read SPARC galaxy data file.
    
    Parameters:
    -----------
    filename : str
        Path to galaxy file
    
    Returns:
    --------
    data : dict
        Dictionary with keys: 'radius', 'vobs', 'verr', 
        'vgas', 'vdisk', 'vbul'
    """
    data = np.loadtxt(filename, skiprows=3)
    
    return {
        'radius': data[:, 0],  # kpc
        'vobs': data[:, 1],    # km/s
        'verr': data[:, 2],    # km/s
        'vgas': data[:, 3],    # km/s
        'vdisk': data[:, 4],   # km/s
        'vbul': data[:, 5] if data.shape[1] > 5 else np.zeros_like(data[:, 0])
    }


def baryonic_mass_enclosed(radius, vgas, vdisk, vbul):
    """
    Compute enclosed baryonic mass from velocity components.
    
    Parameters:
    -----------
    radius : array-like
        Radius [kpc]
    vgas, vdisk, vbul : array-like
        Velocity components [km/s]
    
    Returns:
    --------
    mass_enclosed : array-like
        M(<r) [M_sun]
    """
    # V^2 = G*M/r → M = V^2 * r / G
    v_total_sq = vgas**2 + vdisk**2 + vbul**2
    mass_enclosed = v_total_sq * radius / G
    
    return mass_enclosed


class ProgressBar:
    """Simple progress bar for loops."""
    
    def __init__(self, total, desc="Processing"):
        self.total = total
        self.desc = desc
        self.current = 0
    
    def update(self, n=1):
        self.current += n
        percent = 100 * self.current / self.total
        bar_length = 50
        filled = int(bar_length * self.current / self.total)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f'\r{self.desc}: |{bar}| {percent:.1f}%', end='', flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete


# Logging setup
import logging

def setup_logger(name, level=logging.INFO):
    """Setup logger for analysis."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger
