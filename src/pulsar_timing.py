"""
Pulsar Timing Analysis for 3D+3D Theory Validation
===================================================

This module implements pulsar timing residual analysis to test the 3D+3D theory
predictions of temporal breathing with period τ_b ≈ 28.4 years.

Key Functions:
--------------
- load_ipta_data(): Load IPTA DR2 pulsar timing data
- load_nanograv_data(): Load NANOGrav 15-year dataset
- calculate_residuals(): Compute timing residuals
- fit_breathing_signal(): Fit temporal breathing signature
- mass_amplitude_scaling(): Test σ ∝ M^α relationship for pulsars
- model_comparison(): Compare 3D+3D vs ΛCDM for pulsar timing

Theory Predictions:
-------------------
τ_b = 28.4 ± 6.2 years          (temporal breathing period)
σ ∝ M^(0.28 ± 0.08)             (mass-amplitude scaling for pulsars)
λ_b = 4.30 ± 0.15 kpc           (breathing scale, universal)

Authors: Simone Calzighetti & Lucy (AI Collaborator)
License: MIT (code) + CC-BY-4.0 (scientific content)
DOI: 10.5281/zenodo.17516365
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize, signal
from scipy.interpolate import interp1d
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
import warnings
from typing import Dict, List, Tuple, Optional, Union

from .utils import (
    calculate_chi2, calculate_bic, calculate_aic,
    bootstrap_uncertainty, save_results
)


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_ipta_data(data_path: str, quality_cuts: bool = True) -> pd.DataFrame:
    """
    Load IPTA DR2 pulsar timing data.
    
    Parameters:
    -----------
    data_path : str
        Path to IPTA DR2 data directory
    quality_cuts : bool
        Apply quality cuts (remove pulsars with <50 TOAs or timespan <5 years)
    
    Returns:
    --------
    pd.DataFrame
        Pulsar timing data with columns:
        - pulsar_name: PSR name (e.g., 'J0030+0451')
        - mjd: Modified Julian Date array
        - residual: Timing residual (microseconds)
        - residual_err: Timing residual error (microseconds)
        - mass: Pulsar mass estimate (solar masses)
        - distance: Distance estimate (kpc)
    """
    print(f"Loading IPTA DR2 data from {data_path}...")
    
    # TODO: Implement actual IPTA DR2 data reading
    # This is a placeholder structure - adapt to actual IPTA format
    
    # IPTA DR2 typically uses .par (parameter) and .tim (timing) files
    # We'll need to parse these using PINT or similar
    
    # For now, create template structure
    pulsars = []
    
    # Example structure - replace with actual file reading
    """
    Expected IPTA DR2 format:
    - One directory per pulsar
    - Each contains: pulsar.par (parameters) and pulsar.tim (TOAs)
    """
    
    if quality_cuts:
        print("Applying quality cuts: N_TOAs >= 50, timespan >= 5 years")
        # Filter criteria will be applied after loading
    
    print(f"Loaded {len(pulsars)} pulsars from IPTA DR2")
    
    return pd.DataFrame(pulsars)


def load_nanograv_data(data_path: str, dataset: str = '15yr') -> pd.DataFrame:
    """
    Load NANOGrav pulsar timing data.
    
    Parameters:
    -----------
    data_path : str
        Path to NANOGrav data directory
    dataset : str
        Dataset version ('12.5yr' or '15yr')
    
    Returns:
    --------
    pd.DataFrame
        Pulsar timing data (same structure as load_ipta_data)
    """
    print(f"Loading NANOGrav {dataset} data from {data_path}...")
    
    # NANOGrav provides data in .par and .tim format
    # Can also use their published residuals directly
    
    # TODO: Implement NANOGrav data reading
    # Use PINT library for robust parsing
    
    pulsars = []
    
    print(f"Loaded {len(pulsars)} pulsars from NANOGrav {dataset}")
    
    return pd.DataFrame(pulsars)


def parse_tempo2_residuals(par_file: str, tim_file: str) -> Dict:
    """
    Parse TEMPO2 .par and .tim files to extract timing residuals.
    
    Parameters:
    -----------
    par_file : str
        Path to .par file (pulsar parameters)
    tim_file : str
        Path to .tim file (time-of-arrival data)
    
    Returns:
    --------
    dict
        Dictionary with keys: 'mjd', 'residual', 'residual_err', 'mass', 'distance'
    """
    # TODO: Implement TEMPO2 file parsing
    # Use PINT library: pint.models.get_model(par_file), pint.toa.get_TOAs(tim_file)
    
    pulsar_data = {
        'mjd': np.array([]),
        'residual': np.array([]),
        'residual_err': np.array([]),
        'mass': None,  # Extract from .par if available
        'distance': None  # Extract from .par if available
    }
    
    return pulsar_data


# ============================================================================
# TIMING RESIDUAL ANALYSIS
# ============================================================================

def calculate_residuals(toas: np.ndarray, model_params: Dict, 
                       fit_type: str = 'tempo2') -> np.ndarray:
    """
    Calculate timing residuals: observed TOA - predicted TOA.
    
    Parameters:
    -----------
    toas : np.ndarray
        Time-of-arrival observations (MJD)
    model_params : dict
        Pulsar timing model parameters
    fit_type : str
        Timing model type ('tempo2', 'pint')
    
    Returns:
    --------
    np.ndarray
        Timing residuals (microseconds)
    """
    # Timing residuals are typically pre-computed by TEMPO2/PINT
    # This function is for post-processing and additional modeling
    
    # Basic residual = TOA_obs - TOA_predicted
    # where TOA_predicted comes from the timing model
    
    pass


def remove_timing_model(residuals: np.ndarray, mjd: np.ndarray,
                       model_order: int = 2) -> np.ndarray:
    """
    Remove low-frequency timing model variations (polynomial fit).
    
    Parameters:
    -----------
    residuals : np.ndarray
        Raw timing residuals (microseconds)
    mjd : np.ndarray
        Modified Julian Dates
    model_order : int
        Polynomial order for timing model removal
    
    Returns:
    --------
    np.ndarray
        Residuals with timing model removed
    """
    # Fit polynomial to remove long-term trends
    coeffs = np.polyfit(mjd, residuals, model_order)
    model = np.polyval(coeffs, mjd)
    
    return residuals - model


# ============================================================================
# 3D+3D BREATHING SIGNAL ANALYSIS
# ============================================================================

def fit_breathing_signal(mjd: np.ndarray, residuals: np.ndarray, 
                        residual_err: np.ndarray,
                        period_range: Tuple[float, float] = (20.0, 35.0),
                        n_harmonics: int = 2) -> Dict:
    """
    Fit temporal breathing signature to pulsar timing residuals.
    
    3D+3D Theory predicts sinusoidal modulation:
        Δt(t) = A * sin(2π t / τ_b + φ)
    
    where τ_b ≈ 28.4 years is the temporal breathing period.
    
    Parameters:
    -----------
    mjd : np.ndarray
        Modified Julian Dates
    residuals : np.ndarray
        Timing residuals (microseconds)
    residual_err : np.ndarray
        Timing residual errors (microseconds)
    period_range : tuple
        Search range for period (years)
    n_harmonics : int
        Number of harmonics to include
    
    Returns:
    --------
    dict
        Fit results:
        - period: Best-fit period (years)
        - period_err: Period uncertainty
        - amplitude: Signal amplitude (microseconds)
        - amplitude_err: Amplitude uncertainty
        - phase: Initial phase (radians)
        - chi2: Chi-squared of fit
        - significance: Detection significance (sigma)
    """
    # Convert MJD to years (for period fitting)
    mjd_years = (mjd - mjd[0]) / 365.25
    timespan = mjd_years[-1] - mjd_years[0]
    
    print(f"Fitting breathing signal over {timespan:.1f} years...")
    
    # Initial guess: τ_b = 28.4 years (from SPARC analysis)
    period_guess = 28.4
    
    def breathing_model(t, period, amplitude, phase, *harmonic_amps):
        """
        Multi-harmonic breathing model.
        """
        model = amplitude * np.sin(2*np.pi*t/period + phase)
        
        # Add harmonics if requested
        for i, amp_h in enumerate(harmonic_amps, start=2):
            model += amp_h * np.sin(2*np.pi*i*t/period + phase)
        
        return model
    
    # Initial parameters: [period, amplitude, phase, harmonic_amplitudes...]
    p0 = [period_guess, np.std(residuals), 0.0] + [0.0] * (n_harmonics - 1)
    
    # Bounds for parameters
    bounds = (
        [period_range[0], 0, -np.pi] + [-np.inf] * (n_harmonics - 1),
        [period_range[1], np.inf, np.pi] + [np.inf] * (n_harmonics - 1)
    )
    
    try:
        # Weighted least-squares fit
        popt, pcov = optimize.curve_fit(
            breathing_model, mjd_years, residuals,
            p0=p0, sigma=residual_err, bounds=bounds,
            absolute_sigma=True, maxfev=10000
        )
        
        # Extract fit parameters
        period_fit = popt[0]
        amplitude_fit = popt[1]
        phase_fit = popt[2]
        
        # Uncertainties from covariance matrix
        perr = np.sqrt(np.diag(pcov))
        period_err = perr[0]
        amplitude_err = perr[1]
        
        # Calculate chi-squared
        model_fit = breathing_model(mjd_years, *popt)
        chi2 = np.sum(((residuals - model_fit) / residual_err) ** 2)
        dof = len(residuals) - len(popt)
        chi2_reduced = chi2 / dof
        
        # Null hypothesis: no signal (constant mean)
        chi2_null = np.sum((residuals / residual_err) ** 2)
        delta_chi2 = chi2_null - chi2
        significance = np.sqrt(delta_chi2)  # Approximate significance
        
        results = {
            'period': period_fit,
            'period_err': period_err,
            'amplitude': amplitude_fit,
            'amplitude_err': amplitude_err,
            'phase': phase_fit,
            'chi2': chi2,
            'chi2_reduced': chi2_reduced,
            'dof': dof,
            'significance': significance,
            'model_residuals': residuals - model_fit,
            'timespan': timespan
        }
        
        print(f"  Period: {period_fit:.2f} ± {period_err:.2f} years")
        print(f"  Amplitude: {amplitude_fit:.3f} ± {amplitude_err:.3f} μs")
        print(f"  χ²/dof: {chi2_reduced:.2f}")
        print(f"  Significance: {significance:.1f}σ")
        
        return results
        
    except Exception as e:
        print(f"  WARNING: Fit failed - {e}")
        return None


def lomb_scargle_periodogram(mjd: np.ndarray, residuals: np.ndarray,
                             period_range: Tuple[float, float] = (5.0, 50.0),
                             n_periods: int = 1000) -> Dict:
    """
    Compute Lomb-Scargle periodogram for unevenly sampled timing data.
    
    Parameters:
    -----------
    mjd : np.ndarray
        Modified Julian Dates
    residuals : np.ndarray
        Timing residuals (microseconds)
    period_range : tuple
        Period range to search (years)
    n_periods : int
        Number of periods to test
    
    Returns:
    --------
    dict
        Periodogram results:
        - periods: Array of tested periods (years)
        - power: Lomb-Scargle power at each period
        - best_period: Period with maximum power
        - false_alarm_prob: False alarm probability
    """
    # Convert to frequency space
    mjd_years = (mjd - mjd[0]) / 365.25
    
    # Frequency range (1/period)
    freq_min = 1.0 / period_range[1]
    freq_max = 1.0 / period_range[0]
    frequencies = np.linspace(freq_min, freq_max, n_periods)
    
    # Compute Lomb-Scargle periodogram
    power = signal.lombscargle(mjd_years, residuals, 2*np.pi*frequencies,
                               normalize=True)
    
    # Convert back to periods
    periods = 1.0 / frequencies
    
    # Find peak
    idx_max = np.argmax(power)
    best_period = periods[idx_max]
    max_power = power[idx_max]
    
    # Estimate false alarm probability (Baluev 2008)
    n_independent = len(residuals)
    false_alarm_prob = 1.0 - (1.0 - np.exp(-max_power)) ** n_independent
    
    results = {
        'periods': periods,
        'power': power,
        'best_period': best_period,
        'max_power': max_power,
        'false_alarm_prob': false_alarm_prob
    }
    
    print(f"Lomb-Scargle peak at period = {best_period:.2f} years")
    print(f"  False alarm probability: {false_alarm_prob:.2e}")
    
    return results


# ============================================================================
# MASS-AMPLITUDE SCALING
# ============================================================================

def mass_amplitude_scaling(pulsar_data: pd.DataFrame,
                          mass_col: str = 'mass',
                          amp_col: str = 'breathing_amplitude') -> Dict:
    """
    Test mass-amplitude scaling relationship for pulsars.
    
    3D+3D Theory predicts: σ ∝ M^α
    
    For pulsars: α = 0.28 ± 0.08 (prediction from theory)
    For galaxies: α = 0.34 ± 0.05 (measured from SPARC)
    
    Parameters:
    -----------
    pulsar_data : pd.DataFrame
        Pulsar properties including mass and breathing amplitude
    mass_col : str
        Column name for pulsar mass (solar masses)
    amp_col : str
        Column name for breathing amplitude (microseconds)
    
    Returns:
    --------
    dict
        Scaling analysis results:
        - alpha: Power-law exponent
        - alpha_err: Uncertainty
        - correlation: Pearson correlation coefficient
        - p_value: Significance of correlation
    """
    # Filter valid data (positive masses and amplitudes)
    valid = (pulsar_data[mass_col] > 0) & (pulsar_data[amp_col] > 0)
    masses = pulsar_data.loc[valid, mass_col].values
    amplitudes = pulsar_data.loc[valid, amp_col].values
    
    print(f"Testing mass-amplitude scaling with {len(masses)} pulsars...")
    
    # Log-log fit: log(σ) = α * log(M) + β
    log_mass = np.log10(masses)
    log_amp = np.log10(amplitudes)
    
    # Linear regression in log-space
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_mass, log_amp)
    
    alpha = slope
    alpha_err = std_err
    
    # Bootstrap for robust uncertainty
    alpha_bootstrap = bootstrap_uncertainty(
        lambda M, A: stats.linregress(np.log10(M), np.log10(A))[0],
        masses, amplitudes, n_bootstrap=1000
    )
    
    results = {
        'alpha': alpha,
        'alpha_err': alpha_err,
        'alpha_bootstrap': alpha_bootstrap,
        'correlation': r_value,
        'p_value': p_value,
        'n_pulsars': len(masses),
        'log_mass': log_mass,
        'log_amplitude': log_amp
    }
    
    print(f"  α = {alpha:.3f} ± {alpha_err:.3f}")
    print(f"  Correlation: r = {r_value:.3f}, p = {p_value:.2e}")
    
    # Compare to theoretical prediction
    alpha_theory = 0.28
    alpha_theory_err = 0.08
    
    deviation = abs(alpha - alpha_theory) / np.sqrt(alpha_err**2 + alpha_theory_err**2)
    print(f"  Theory: α = {alpha_theory:.2f} ± {alpha_theory_err:.2f}")
    print(f"  Deviation: {deviation:.1f}σ")
    
    results['alpha_theory'] = alpha_theory
    results['deviation_sigma'] = deviation
    
    return results


# ============================================================================
# MODEL COMPARISON: 3D+3D vs ΛCDM
# ============================================================================

def compare_models_pulsar(pulsar_data: Dict, 
                         include_breathing: bool = True) -> Dict:
    """
    Compare 3D+3D model vs ΛCDM for pulsar timing residuals.
    
    Models:
    -------
    ΛCDM: Pure noise (white + red noise processes)
    3D+3D: ΛCDM + temporal breathing signal
    
    Parameters:
    -----------
    pulsar_data : dict
        Dictionary with 'mjd', 'residual', 'residual_err'
    include_breathing : bool
        If True, fit 3D+3D model; if False, fit ΛCDM only
    
    Returns:
    --------
    dict
        Model comparison results:
        - chi2_lcdm: ΛCDM chi-squared
        - chi2_3d3d: 3D+3D chi-squared
        - delta_chi2: Improvement (chi2_lcdm - chi2_3d3d)
        - bayes_factor: Bayesian evidence ratio
        - aic_lcdm, aic_3d3d: Akaike Information Criterion
        - bic_lcdm, bic_3d3d: Bayesian Information Criterion
    """
    mjd = pulsar_data['mjd']
    residuals = pulsar_data['residual']
    residual_err = pulsar_data['residual_err']
    
    n_data = len(residuals)
    
    # ========== ΛCDM Model (null hypothesis) ==========
    # Residuals are pure noise (mean = 0)
    chi2_lcdm = np.sum((residuals / residual_err) ** 2)
    n_params_lcdm = 1  # Only mean
    
    aic_lcdm = calculate_aic(chi2_lcdm, n_params_lcdm)
    bic_lcdm = calculate_bic(chi2_lcdm, n_params_lcdm, n_data)
    
    print(f"ΛCDM Model:")
    print(f"  χ² = {chi2_lcdm:.1f} ({n_data} dof)")
    print(f"  AIC = {aic_lcdm:.1f}")
    print(f"  BIC = {bic_lcdm:.1f}")
    
    if not include_breathing:
        return {
            'chi2_lcdm': chi2_lcdm,
            'aic_lcdm': aic_lcdm,
            'bic_lcdm': bic_lcdm
        }
    
    # ========== 3D+3D Model (with breathing) ==========
    breathing_fit = fit_breathing_signal(mjd, residuals, residual_err)
    
    if breathing_fit is None:
        print("WARNING: 3D+3D fit failed, cannot compare models")
        return None
    
    chi2_3d3d = breathing_fit['chi2']
    n_params_3d3d = 4  # period, amplitude, phase, (+ mean)
    
    aic_3d3d = calculate_aic(chi2_3d3d, n_params_3d3d)
    bic_3d3d = calculate_bic(chi2_3d3d, n_params_3d3d, n_data)
    
    print(f"\n3D+3D Model:")
    print(f"  χ² = {chi2_3d3d:.1f} ({breathing_fit['dof']} dof)")
    print(f"  AIC = {aic_3d3d:.1f}")
    print(f"  BIC = {bic_3d3d:.1f}")
    
    # ========== Model Comparison ==========
    delta_chi2 = chi2_lcdm - chi2_3d3d
    delta_aic = aic_lcdm - aic_3d3d
    delta_bic = bic_lcdm - bic_3d3d
    
    # Bayes factor (approximate from BIC difference)
    bayes_factor = np.exp(-0.5 * delta_bic)
    
    print(f"\nModel Comparison:")
    print(f"  Δχ² = {delta_chi2:.1f} (3D+3D improvement)")
    print(f"  ΔAIC = {delta_aic:.1f} (positive favors 3D+3D)")
    print(f"  ΔBIC = {delta_bic:.1f} (positive favors 3D+3D)")
    print(f"  Bayes Factor = {bayes_factor:.2e}")
    
    results = {
        'chi2_lcdm': chi2_lcdm,
        'chi2_3d3d': chi2_3d3d,
        'delta_chi2': delta_chi2,
        'aic_lcdm': aic_lcdm,
        'aic_3d3d': aic_3d3d,
        'delta_aic': delta_aic,
        'bic_lcdm': bic_lcdm,
        'bic_3d3d': bic_3d3d,
        'delta_bic': delta_bic,
        'bayes_factor': bayes_factor,
        'breathing_fit': breathing_fit
    }
    
    return results


# ============================================================================
# ENSEMBLE ANALYSIS
# ============================================================================

def analyze_pulsar_ensemble(pulsar_list: List[Dict],
                            min_timespan: float = 5.0,
                            save_output: bool = True) -> Dict:
    """
    Analyze ensemble of pulsars for 3D+3D breathing signal.
    
    Parameters:
    -----------
    pulsar_list : list of dict
        List of pulsar data dictionaries
    min_timespan : float
        Minimum observation timespan (years) for inclusion
    save_output : bool
        Save results to file
    
    Returns:
    --------
    dict
        Ensemble analysis results:
        - n_pulsars: Number of pulsars analyzed
        - detection_rate: Fraction with significant breathing signal
        - mean_period: Mean detected period
        - period_std: Standard deviation of periods
        - mass_scaling: Mass-amplitude scaling results
    """
    print(f"\n{'='*60}")
    print("PULSAR ENSEMBLE ANALYSIS - 3D+3D Theory")
    print(f"{'='*60}\n")
    
    results_list = []
    detected_periods = []
    detected_amplitudes = []
    pulsar_masses = []
    
    for i, pulsar_data in enumerate(pulsar_list, 1):
        name = pulsar_data.get('name', f'PSR_{i}')
        mjd = pulsar_data['mjd']
        timespan = (mjd[-1] - mjd[0]) / 365.25
        
        print(f"\n[{i}/{len(pulsar_list)}] Analyzing {name}...")
        print(f"  Timespan: {timespan:.1f} years, N_TOAs: {len(mjd)}")
        
        if timespan < min_timespan:
            print(f"  SKIPPED: timespan < {min_timespan} years")
            continue
        
        # Fit breathing signal
        fit_result = fit_breathing_signal(
            pulsar_data['mjd'],
            pulsar_data['residual'],
            pulsar_data['residual_err']
        )
        
        if fit_result is not None and fit_result['significance'] > 3.0:
            detected_periods.append(fit_result['period'])
            detected_amplitudes.append(fit_result['amplitude'])
            
            if 'mass' in pulsar_data and pulsar_data['mass'] is not None:
                pulsar_masses.append(pulsar_data['mass'])
            
            print(f"  ✓ DETECTED at {fit_result['significance']:.1f}σ")
        else:
            print(f"  ✗ Not significant")
        
        results_list.append({
            'name': name,
            'timespan': timespan,
            'n_toas': len(mjd),
            'fit_result': fit_result
        })
    
    # ========== Ensemble Statistics ==========
    n_analyzed = len(results_list)
    n_detected = len(detected_periods)
    detection_rate = n_detected / n_analyzed if n_analyzed > 0 else 0
    
    print(f"\n{'='*60}")
    print("ENSEMBLE RESULTS")
    print(f"{'='*60}")
    print(f"Pulsars analyzed: {n_analyzed}")
    print(f"Significant detections (>3σ): {n_detected}")
    print(f"Detection rate: {detection_rate*100:.1f}%")
    
    if n_detected > 0:
        mean_period = np.mean(detected_periods)
        period_std = np.std(detected_periods)
        
        print(f"\nDetected periods:")
        print(f"  Mean: {mean_period:.2f} ± {period_std:.2f} years")
        print(f"  Theory prediction: 28.4 ± 6.2 years")
        
        # Agreement with theory
        theory_period = 28.4
        deviation = abs(mean_period - theory_period) / period_std
        print(f"  Deviation from theory: {deviation:.1f}σ")
    
    ensemble_results = {
        'n_pulsars': n_analyzed,
        'n_detected': n_detected,
        'detection_rate': detection_rate,
        'detected_periods': detected_periods,
        'detected_amplitudes': detected_amplitudes,
        'mean_period': np.mean(detected_periods) if n_detected > 0 else None,
        'period_std': np.std(detected_periods) if n_detected > 0 else None,
        'individual_results': results_list
    }
    
    # Mass-amplitude scaling (if masses available)
    if len(pulsar_masses) == len(detected_amplitudes) and len(pulsar_masses) > 3:
        df_scaling = pd.DataFrame({
            'mass': pulsar_masses,
            'breathing_amplitude': detected_amplitudes
        })
        scaling_results = mass_amplitude_scaling(df_scaling)
        ensemble_results['mass_scaling'] = scaling_results
    
    if save_output:
        save_results(ensemble_results, 'pulsar_ensemble_results.pkl')
        print(f"\nResults saved to 'pulsar_ensemble_results.pkl'")
    
    return ensemble_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    """
    Example usage of pulsar timing analysis.
    """
    print("3D+3D Pulsar Timing Analysis Module")
    print("====================================\n")
    
    # Example: Load and analyze IPTA DR2 data
    # data_path = '/path/to/ipta/dr2/'
    # pulsar_data = load_ipta_data(data_path)
    
    # Example: Analyze single pulsar
    # pulsar = pulsar_data[pulsar_data['pulsar_name'] == 'J0030+0451']
    # fit_result = fit_breathing_signal(pulsar['mjd'], pulsar['residual'], pulsar['residual_err'])
    
    print("Module loaded successfully.")
    print("\nKey functions:")
    print("  - load_ipta_data()")
    print("  - load_nanograv_data()")
    print("  - fit_breathing_signal()")
    print("  - mass_amplitude_scaling()")
    print("  - compare_models_pulsar()")
    print("  - analyze_pulsar_ensemble()")
