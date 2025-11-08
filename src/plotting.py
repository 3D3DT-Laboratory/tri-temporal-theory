#  TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
#  Unauthorized copying of this file, via any medium, is strictly prohibited.
#  Proprietary and confidential. All rights reserved.
"""
Plotting and Visualization for 3D+3D Theory
============================================

This module creates publication-quality figures for the 3D+3D theory paper,
including rotation curves, scaling relationships, model comparisons, and
statistical diagnostics.

Key Plotting Functions:
-----------------------
- plot_rotation_curve(): Individual galaxy rotation curves with 3D+3D fit
- plot_mass_amplitude_scaling(): M^α scaling relationship
- plot_model_comparison(): χ² comparison between ΛCDM and 3D+3D
- plot_residual_analysis(): Residual distributions and diagnostics
- plot_breathing_detection(): Temporal breathing signal visualization
- plot_ensemble_summary(): Multi-panel summary figure

Figure Specifications:
----------------------
All figures follow publication standards:
- Size: 3.5" (single column) or 7" (double column) width
- DPI: 300 for raster, vector for line plots
- Fonts: 8-10pt for labels, 6-8pt for tick labels
- Color scheme: Colorblind-friendly (Wong 2011 palette)
- Format: PDF (vector) or PNG (high-res raster)

Authors: Simone Calzighetti & Lucy (AI Collaborator)
License: MIT (code) + CC-BY-4.0 (scientific content)
DOI: 10.5281/zenodo.17516365
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from typing import Dict, List, Tuple, Optional
import warnings

# ============================================================================
# PUBLICATION STYLE SETUP
# ============================================================================

def setup_publication_style():
    """
    Configure matplotlib for publication-quality figures.
    
    Follows guidelines from:
    - Nature journals
    - AAS journals (ApJ, AJ)
    - Physical Review
    """
    # Font settings
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times', 'Times New Roman', 'DejaVu Serif']
    rcParams['font.size'] = 9
    rcParams['axes.labelsize'] = 10
    rcParams['axes.titlesize'] = 10
    rcParams['xtick.labelsize'] = 8
    rcParams['ytick.labelsize'] = 8
    rcParams['legend.fontsize'] = 8
    
    # Figure settings
    rcParams['figure.dpi'] = 100  # Screen display
    rcParams['savefig.dpi'] = 300  # High-res output
    rcParams['savefig.format'] = 'pdf'
    rcParams['savefig.bbox'] = 'tight'
    
    # Line and marker settings
    rcParams['lines.linewidth'] = 1.5
    rcParams['lines.markersize'] = 4
    rcParams['patch.linewidth'] = 1.0
    
    # Axis settings
    rcParams['axes.linewidth'] = 0.8
    rcParams['xtick.major.width'] = 0.8
    rcParams['ytick.major.width'] = 0.8
    rcParams['xtick.minor.width'] = 0.5
    rcParams['ytick.minor.width'] = 0.5
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['xtick.top'] = True
    rcParams['ytick.right'] = True
    
    # Grid
    rcParams['grid.alpha'] = 0.3
    rcParams['grid.linewidth'] = 0.5
    
    # Legend
    rcParams['legend.frameon'] = True
    rcParams['legend.framealpha'] = 0.9
    rcParams['legend.fancybox'] = False
    rcParams['legend.edgecolor'] = 'black'


# Colorblind-friendly palette (Wong 2011)
COLORS = {
    'blue': '#0173B2',
    'orange': '#DE8F05',
    'green': '#029E73',
    'red': '#CC78BC',
    'purple': '#CA9161',
    'brown': '#949494',
    'pink': '#ECE133',
    'gray': '#56B4E9'
}


# ============================================================================
# ROTATION CURVE PLOTTING
# ============================================================================

def plot_rotation_curve(radius: np.ndarray, 
                       velocity_obs: np.ndarray,
                       velocity_err: np.ndarray,
                       velocity_lcdm: np.ndarray,
                       velocity_3d3d: np.ndarray,
                       galaxy_name: str = 'Galaxy',
                       mass: Optional[float] = None,
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot galaxy rotation curve with ΛCDM and 3D+3D model fits.
    
    Parameters:
    -----------
    radius : np.ndarray
        Radial distances (kpc)
    velocity_obs : np.ndarray
        Observed rotation velocities (km/s)
    velocity_err : np.ndarray
        Velocity uncertainties (km/s)
    velocity_lcdm : np.ndarray
        ΛCDM model predictions (km/s)
    velocity_3d3d : np.ndarray
        3D+3D model predictions (km/s)
    galaxy_name : str
        Galaxy identifier
    mass : float, optional
        Total stellar mass (solar masses)
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    matplotlib.Figure
    """
    setup_publication_style()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.5), 
                                    gridspec_kw={'height_ratios': [3, 1]},
                                    sharex=True)
    
    # ========== Main Panel: Rotation Curve ==========
    
    # Observations with error bars
    ax1.errorbar(radius, velocity_obs, yerr=velocity_err,
                fmt='o', color='black', markersize=3, 
                capsize=2, capthick=0.5, linewidth=0.5,
                label='Observed', zorder=3)
    
    # ΛCDM model
    ax1.plot(radius, velocity_lcdm, '-', color=COLORS['orange'],
            linewidth=1.5, label='ΛCDM', zorder=2)
    
    # 3D+3D model
    ax1.plot(radius, velocity_3d3d, '-', color=COLORS['blue'],
            linewidth=2.0, label='3D+3D', zorder=2)
    
    # Labels and formatting
    ax1.set_ylabel('Rotation Velocity (km/s)', fontsize=10)
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(alpha=0.3, linewidth=0.5)
    
    # Title with mass if provided
    if mass is not None:
        title = f'{galaxy_name}\n' + r'$M_* = $' + f'{mass:.2e}' + r' $M_\odot$'
    else:
        title = galaxy_name
    ax1.set_title(title, fontsize=10, pad=10)
    
    # ========== Bottom Panel: Residuals ==========
    
    residuals_lcdm = velocity_obs - velocity_lcdm
    residuals_3d3d = velocity_obs - velocity_3d3d
    
    # Zero line
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, zorder=1)
    
    # Residuals
    ax2.plot(radius, residuals_lcdm, 'o', color=COLORS['orange'],
            markersize=3, label='ΛCDM', alpha=0.6)
    ax2.plot(radius, residuals_3d3d, 's', color=COLORS['blue'],
            markersize=3, label='3D+3D', alpha=0.8)
    
    # Labels
    ax2.set_xlabel('Radius (kpc)', fontsize=10)
    ax2.set_ylabel('Residuals\n(km/s)', fontsize=9)
    ax2.legend(loc='best', fontsize=7)
    ax2.grid(alpha=0.3, linewidth=0.5)
    
    # Calculate χ²
    chi2_lcdm = np.sum((residuals_lcdm / velocity_err)**2)
    chi2_3d3d = np.sum((residuals_3d3d / velocity_err)**2)
    
    # Add χ² values as text
    chi2_text = f'χ²/N: ΛCDM={chi2_lcdm/len(radius):.2f}, 3D+3D={chi2_3d3d/len(radius):.2f}'
    ax2.text(0.05, 0.95, chi2_text, transform=ax2.transAxes,
            fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# MASS-AMPLITUDE SCALING
# ============================================================================

def plot_mass_amplitude_scaling(masses: np.ndarray,
                               amplitudes: np.ndarray,
                               amplitude_err: Optional[np.ndarray] = None,
                               fit_params: Optional[Dict] = None,
                               dataset_label: str = 'Galaxies',
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot mass-amplitude scaling relationship: σ ∝ M^α
    
    Parameters:
    -----------
    masses : np.ndarray
        Stellar masses (solar masses)
    amplitudes : np.ndarray
        Breathing amplitudes (km/s for galaxies, μs for pulsars)
    amplitude_err : np.ndarray, optional
        Amplitude uncertainties
    fit_params : dict, optional
        Fit results with keys: 'alpha', 'alpha_err', 'correlation', 'p_value'
    dataset_label : str
        Label for dataset
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    matplotlib.Figure
    """
    setup_publication_style()
    
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    
    # Log-log plot
    log_mass = np.log10(masses)
    log_amp = np.log10(amplitudes)
    
    # Error bars if provided
    if amplitude_err is not None:
        log_amp_err = amplitude_err / (amplitudes * np.log(10))
        ax.errorbar(log_mass, log_amp, yerr=log_amp_err,
                   fmt='o', color=COLORS['blue'], markersize=4,
                   capsize=2, capthick=0.5, linewidth=0.5,
                   alpha=0.6, label=dataset_label, zorder=2)
    else:
        ax.plot(log_mass, log_amp, 'o', color=COLORS['blue'],
               markersize=4, alpha=0.6, label=dataset_label, zorder=2)
    
    # Best-fit line
    if fit_params is not None:
        alpha = fit_params['alpha']
        alpha_err = fit_params.get('alpha_err', 0)
        
        # Fit line
        mass_range = np.linspace(log_mass.min(), log_mass.max(), 100)
        intercept = np.mean(log_amp - alpha * log_mass)
        fit_line = alpha * mass_range + intercept
        
        ax.plot(mass_range, fit_line, '-', color=COLORS['orange'],
               linewidth=2.0, label=f'α = {alpha:.2f} ± {alpha_err:.2f}',
               zorder=3)
        
        # Confidence band (±1σ)
        fit_line_upper = (alpha + alpha_err) * mass_range + intercept
        fit_line_lower = (alpha - alpha_err) * mass_range + intercept
        ax.fill_between(mass_range, fit_line_lower, fit_line_upper,
                       color=COLORS['orange'], alpha=0.2, zorder=1)
        
        # Add correlation info
        if 'correlation' in fit_params and 'p_value' in fit_params:
            r = fit_params['correlation']
            p = fit_params['p_value']
            
            stats_text = f'r = {r:.3f}\np = {p:.2e}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Labels
    ax.set_xlabel(r'$\log_{10}(M_* / M_\odot)$', fontsize=11)
    ax.set_ylabel(r'$\log_{10}(\sigma)$', fontsize=11)
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(alpha=0.3, linewidth=0.5)
    
    # Theory prediction (if galaxies)
    if 'Galaxies' in dataset_label or 'SPARC' in dataset_label:
        theory_alpha = 0.34
        ax.axhline(0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.text(0.95, 0.05, f'Theory: α = {theory_alpha:.2f}',
               transform=ax.transAxes, fontsize=8,
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# MODEL COMPARISON
# ============================================================================

def plot_model_comparison(comparison_results: Dict,
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot comprehensive model comparison: χ², AIC, BIC, Bayes Factor.
    
    Parameters:
    -----------
    comparison_results : dict
        Results from model_comparison module
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    matplotlib.Figure
    """
    setup_publication_style()
    
    fig = plt.figure(figsize=(7.0, 5.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # ========== Panel 1: χ² Comparison ==========
    ax1 = fig.add_subplot(gs[0, 0])
    
    models = ['ΛCDM', '3D+3D']
    chi2_values = [
        comparison_results['chi2_lcdm'],
        comparison_results['chi2_3d3d']
    ]
    
    bars = ax1.bar(models, chi2_values, color=[COLORS['orange'], COLORS['blue']],
                  alpha=0.7, edgecolor='black', linewidth=1)
    
    ax1.set_ylabel(r'$\chi^2$', fontsize=11)
    ax1.set_title('Chi-Squared', fontsize=10, pad=10)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add values on bars
    for bar, val in zip(bars, chi2_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=8)
    
    # ========== Panel 2: Information Criteria ==========
    ax2 = fig.add_subplot(gs[0, 1])
    
    criteria = ['AIC', 'BIC']
    lcdm_values = [
        comparison_results['aic_lcdm'],
        comparison_results['bic_lcdm']
    ]
    d3d3_values = [
        comparison_results['aic_3d3d'],
        comparison_results['bic_3d3d']
    ]
    
    x = np.arange(len(criteria))
    width = 0.35
    
    ax2.bar(x - width/2, lcdm_values, width, label='ΛCDM',
           color=COLORS['orange'], alpha=0.7, edgecolor='black', linewidth=1)
    ax2.bar(x + width/2, d3d3_values, width, label='3D+3D',
           color=COLORS['blue'], alpha=0.7, edgecolor='black'], linewidth=1)
    
    ax2.set_ylabel('Value', fontsize=11)
    ax2.set_title('Information Criteria', fontsize=10, pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(criteria)
    ax2.legend(loc='best', framealpha=0.9, fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    
    # ========== Panel 3: Δχ², ΔAIC, ΔBIC ==========
    ax3 = fig.add_subplot(gs[1, 0])
    
    delta_metrics = ['Δχ²', 'ΔAIC', 'ΔBIC']
    delta_values = [
        comparison_results['delta_chi2'],
        comparison_results['delta_aic'],
        comparison_results['delta_bic']
    ]
    
    colors_delta = [COLORS['green'] if v > 0 else COLORS['red'] 
                    for v in delta_values]
    
    bars = ax3.barh(delta_metrics, delta_values, color=colors_delta,
                   alpha=0.7, edgecolor='black', linewidth=1)
    
    ax3.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax3.set_xlabel('Improvement (3D+3D - ΛCDM)', fontsize=10)
    ax3.set_title('Model Improvements', fontsize=10, pad=10)
    ax3.grid(axis='x', alpha=0.3)
    
    # Add values
    for i, (metric, val) in enumerate(zip(delta_metrics, delta_values)):
        ax3.text(val, i, f' {val:.1f}',
                va='center', ha='left' if val > 0 else 'right',
                fontsize=8)
    
    # ========== Panel 4: Bayes Factor ==========
    ax4 = fig.add_subplot(gs[1, 1])
    
    bf = comparison_results.get('bayes_factor', 1.0)
    log_bf = np.log10(bf) if bf > 0 else 0
    
    # Jeffreys' scale
    categories = ['Decisive\n(>100)', 'V. Strong\n(30-100)', 
                 'Strong\n(10-30)', 'Substantial\n(3-10)']
    thresholds = [2, 1.48, 1, 0.48]  # log10 thresholds
    
    ax4.barh(categories, thresholds, color='lightgray', alpha=0.3,
            edgecolor='black', linewidth=0.8)
    
    # Actual BF
    if log_bf >= 2:
        bar_height = 0
        color_bf = COLORS['green']
    elif log_bf >= 1.48:
        bar_height = 1
        color_bf = COLORS['blue']
    elif log_bf >= 1:
        bar_height = 2
        color_bf = COLORS['orange']
    else:
        bar_height = 3
        color_bf = COLORS['red']
    
    ax4.barh(categories[bar_height], log_bf, color=color_bf, alpha=0.7,
            edgecolor='black', linewidth=1.5)
    
    ax4.set_xlabel(r'$\log_{10}$(Bayes Factor)', fontsize=10)
    ax4.set_title('Bayesian Evidence', fontsize=10, pad=10)
    ax4.grid(axis='x', alpha=0.3)
    
    # Add BF value
    ax4.text(0.95, 0.95, f'BF = {bf:.2e}',
            transform=ax4.transAxes, fontsize=8,
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# RESIDUAL ANALYSIS
# ============================================================================

def plot_residual_analysis(residuals: np.ndarray,
                          model_name: str = 'Model',
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot residual diagnostics: histogram, Q-Q plot, autocorrelation.
    
    Parameters:
    -----------
    residuals : np.ndarray
        Model residuals
    model_name : str
        Name of model
    save_path : str, optional
        Path to save figure
    
    Returns:
    --------
    matplotlib.Figure
    """
    setup_publication_style()
    
    fig = plt.figure(figsize=(7.0, 3.0))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)
    
    # ========== Panel 1: Histogram ==========
    ax1 = fig.add_subplot(gs[0, 0])
    
    ax1.hist(residuals, bins=20, color=COLORS['blue'], alpha=0.6,
            edgecolor='black', linewidth=0.8, density=True)
    
    # Gaussian overlay
    mu, sigma = np.mean(residuals), np.std(residuals)
    x_gauss = np.linspace(residuals.min(), residuals.max(), 100)
    y_gauss = (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x_gauss - mu)/sigma)**2)
    ax1.plot(x_gauss, y_gauss, 'r-', linewidth=2, label='Gaussian')
    
    ax1.set_xlabel('Residuals', fontsize=10)
    ax1.set_ylabel('Density', fontsize=10)
    ax1.set_title(f'{model_name} Residuals', fontsize=10, pad=10)
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(alpha=0.3)
    
    # ========== Panel 2: Q-Q Plot ==========
    ax2 = fig.add_subplot(gs[0, 1])
    
    from scipy import stats as sp_stats
    sp_stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.get_lines()[0].set_color(COLORS['blue'])
    ax2.get_lines()[0].set_markersize(3)
    ax2.get_lines()[1].set_color('red')
    ax2.get_lines()[1].set_linewidth(1.5)
    
    ax2.set_title('Q-Q Plot', fontsize=10, pad=10)
    ax2.grid(alpha=0.3)
    
    # ========== Panel 3: Autocorrelation ==========
    ax3 = fig.add_subplot(gs[0, 2])
    
    from matplotlib import pyplot
    # Simple autocorrelation
    autocorr = np.correlate(residuals - np.mean(residuals), 
                           residuals - np.mean(residuals), mode='full')
    autocorr = autocorr[autocorr.size // 2:] / autocorr[autocorr.size // 2]
    
    lags = np.arange(len(autocorr))
    ax3.stem(lags[:20], autocorr[:20], basefmt=' ', linefmt=COLORS['blue'],
            markerfmt='o')
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax3.axhline(0.2, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax3.axhline(-0.2, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    
    ax3.set_xlabel('Lag', fontsize=10)
    ax3.set_ylabel('Autocorrelation', fontsize=10)
    ax3.set_title('Autocorrelation', fontsize=10, pad=10)
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    """
    Example usage of plotting module.
    """
    print("3D+3D Plotting Module")
    print("=====================\n")
    
    setup_publication_style()
    print("Publication style configured.")
    
    print("\nKey functions:")
    print("  - plot_rotation_curve()")
    print("  - plot_mass_amplitude_scaling()")
    print("  - plot_model_comparison()")
    print("  - plot_residual_analysis()")
    print("  - setup_publication_style()")
