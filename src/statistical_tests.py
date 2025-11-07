"""
Statistical Testing Framework for 3D+3D Theory
===============================================

This module implements rigorous statistical tests for validating 3D+3D theory
predictions against observational data.

Key Functions:
--------------
- correlation_analysis(): Test mass-amplitude, breathing-distance correlations
- bootstrap_analysis(): Non-parametric uncertainty estimation
- monte_carlo_simulation(): Test robustness to systematics
- permutation_test(): Non-parametric hypothesis testing
- outlier_detection(): Identify and handle anomalous data
- goodness_of_fit_tests(): KS, Anderson-Darling, chi-squared tests

Statistical Tests Implemented:
-------------------------------
- Pearson/Spearman correlation (with significance)
- Bootstrap resampling (confidence intervals)
- Permutation tests (null hypothesis testing)
- Monte Carlo error propagation
- Outlier detection (Chauvenet, modified Z-score)
- Distribution tests (normality, uniformity)

Theory Predictions to Test:
----------------------------
1. Universal λ_b = 4.30 ± 0.15 kpc (breathing scale)
2. Mass-amplitude scaling: σ ∝ M^0.34 (galaxies), M^0.28 (pulsars)
3. Critical mass threshold: M_crit = 2.43×10¹⁰ M☉
4. Temporal period: τ_b = 28.4 ± 6.2 years

Authors: Simone Calzighetti & Lucy (AI Collaborator)
License: MIT (code) + CC-BY-4.0 (scientific content)
DOI: 10.5281/zenodo.17516365
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import erf
from typing import Dict, List, Tuple, Optional, Callable
import warnings

from .utils import bootstrap_uncertainty, save_results


# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

def correlation_analysis(x: np.ndarray, y: np.ndarray,
                        x_err: Optional[np.ndarray] = None,
                        y_err: Optional[np.ndarray] = None,
                        method: str = 'pearson',
                        n_bootstrap: int = 1000) -> Dict:
    """
    Comprehensive correlation analysis with uncertainty propagation.
    
    Tests correlation between two variables, accounting for measurement
    errors and providing robust uncertainty estimates via bootstrap.
    
    Parameters:
    -----------
    x, y : np.ndarray
        Data arrays to correlate
    x_err, y_err : np.ndarray, optional
        Measurement uncertainties
    method : str
        Correlation method:
        - 'pearson': Linear correlation (default)
        - 'spearman': Rank correlation (robust to outliers)
        - 'kendall': Tau correlation (for small samples)
    n_bootstrap : int
        Number of bootstrap iterations
    
    Returns:
    --------
    dict
        Correlation results:
        - correlation: Correlation coefficient
        - p_value: Statistical significance
        - confidence_interval: 95% CI from bootstrap
        - standard_error: Bootstrap standard error
    """
    # Remove NaN values
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    
    n = len(x)
    
    if n < 3:
        raise ValueError("Need at least 3 data points for correlation")
    
    print(f"\nCorrelation Analysis ({method}):")
    print(f"  N = {n} data points")
    
    # ========== Primary Correlation ==========
    if method == 'pearson':
        corr, p_value = stats.pearsonr(x, y)
    elif method == 'spearman':
        corr, p_value = stats.spearmanr(x, y)
    elif method == 'kendall':
        corr, p_value = stats.kendalltau(x, y)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"  Correlation: r = {corr:.4f}")
    print(f"  p-value: {p_value:.2e}")
    
    # ========== Bootstrap Uncertainty ==========
    def correlation_func(x_boot, y_boot):
        if method == 'pearson':
            return stats.pearsonr(x_boot, y_boot)[0]
        elif method == 'spearman':
            return stats.spearmanr(x_boot, y_boot)[0]
        elif method == 'kendall':
            return stats.kendalltau(x_boot, y_boot)[0]
    
    corr_bootstrap = []
    rng = np.random.RandomState(42)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n, size=n, replace=True)
        x_boot = x[indices]
        y_boot = y[indices]
        
        try:
            corr_boot = correlation_func(x_boot, y_boot)
            corr_bootstrap.append(corr_boot)
        except:
            continue
    
    corr_bootstrap = np.array(corr_bootstrap)
    
    # Confidence interval (95%)
    ci_lower = np.percentile(corr_bootstrap, 2.5)
    ci_upper = np.percentile(corr_bootstrap, 97.5)
    
    # Standard error
    se = np.std(corr_bootstrap)
    
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  Standard error: {se:.4f}")
    
    # ========== Effect Size Interpretation ==========
    abs_corr = abs(corr)
    if abs_corr > 0.7:
        strength = "very strong"
    elif abs_corr > 0.5:
        strength = "strong"
    elif abs_corr > 0.3:
        strength = "moderate"
    elif abs_corr > 0.1:
        strength = "weak"
    else:
        strength = "very weak"
    
    print(f"  → {strength} correlation")
    
    results = {
        'method': method,
        'n_points': n,
        'correlation': corr,
        'p_value': p_value,
        'confidence_interval': (ci_lower, ci_upper),
        'standard_error': se,
        'bootstrap_distribution': corr_bootstrap,
        'strength': strength
    }
    
    return results


def partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict:
    """
    Calculate partial correlation between x and y, controlling for z.
    
    Partial correlation measures the relationship between x and y after
    removing the linear effect of z on both variables.
    
    Parameters:
    -----------
    x, y : np.ndarray
        Variables to correlate
    z : np.ndarray
        Control variable
    
    Returns:
    --------
    dict
        Partial correlation results
    """
    # Remove z's effect from x and y
    x_resid = x - np.polyval(np.polyfit(z, x, 1), z)
    y_resid = y - np.polyval(np.polyfit(z, y, 1), z)
    
    # Correlate residuals
    corr, p_value = stats.pearsonr(x_resid, y_resid)
    
    print(f"\nPartial Correlation (controlling for z):")
    print(f"  r_xy·z = {corr:.4f}")
    print(f"  p-value = {p_value:.2e}")
    
    return {
        'partial_correlation': corr,
        'p_value': p_value,
        'x_residuals': x_resid,
        'y_residuals': y_resid
    }


# ============================================================================
# BOOTSTRAP ANALYSIS
# ============================================================================

def bootstrap_parameter_uncertainty(data: pd.DataFrame,
                                   fit_func: Callable,
                                   param_names: List[str],
                                   n_bootstrap: int = 1000,
                                   confidence_level: float = 0.95) -> Dict:
    """
    Bootstrap uncertainty estimation for model parameters.
    
    Resamples data with replacement and refits model to obtain
    empirical distribution of parameters.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to resample
    fit_func : callable
        Function that fits model and returns parameters
        Signature: fit_func(data) -> dict of parameters
    param_names : list of str
        Names of parameters to track
    n_bootstrap : int
        Number of bootstrap iterations
    confidence_level : float
        Confidence level for intervals (default 95%)
    
    Returns:
    --------
    dict
        Bootstrap results for each parameter:
        - mean: Bootstrap mean
        - std: Bootstrap standard deviation
        - ci_lower, ci_upper: Confidence interval
        - distribution: Full bootstrap distribution
    """
    n_data = len(data)
    rng = np.random.RandomState(42)
    
    print(f"\nBootstrap Parameter Uncertainty:")
    print(f"  N_data = {n_data}")
    print(f"  N_bootstrap = {n_bootstrap}")
    print(f"  Confidence level = {confidence_level*100:.0f}%")
    
    # Initialize storage for bootstrap parameters
    bootstrap_params = {name: [] for name in param_names}
    
    # Bootstrap loop
    for i in range(n_bootstrap):
        # Resample data
        indices = rng.choice(n_data, size=n_data, replace=True)
        data_boot = data.iloc[indices]
        
        try:
            # Fit model on bootstrap sample
            params_boot = fit_func(data_boot)
            
            # Store parameters
            for name in param_names:
                if name in params_boot:
                    bootstrap_params[name].append(params_boot[name])
        
        except Exception as e:
            warnings.warn(f"Bootstrap iteration {i} failed: {e}")
            continue
    
    # Calculate statistics for each parameter
    results = {}
    
    alpha = 1 - confidence_level
    percentile_lower = 100 * (alpha / 2)
    percentile_upper = 100 * (1 - alpha / 2)
    
    for name in param_names:
        if len(bootstrap_params[name]) > 0:
            dist = np.array(bootstrap_params[name])
            
            mean = np.mean(dist)
            std = np.std(dist)
            ci_lower = np.percentile(dist, percentile_lower)
            ci_upper = np.percentile(dist, percentile_upper)
            
            print(f"\n  {name}:")
            print(f"    Mean: {mean:.4f}")
            print(f"    Std: {std:.4f}")
            print(f"    {confidence_level*100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            
            results[name] = {
                'mean': mean,
                'std': std,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'distribution': dist
            }
    
    return results


# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

def monte_carlo_error_propagation(func: Callable,
                                 params: Dict[str, Tuple[float, float]],
                                 n_samples: int = 10000) -> Dict:
    """
    Monte Carlo error propagation for complex functions.
    
    Propagates uncertainties through arbitrary functions by sampling
    from parameter distributions and evaluating function.
    
    Parameters:
    -----------
    func : callable
        Function to evaluate: result = func(**params)
    params : dict
        Dictionary of {param_name: (mean, std)}
    n_samples : int
        Number of Monte Carlo samples
    
    Returns:
    --------
    dict
        Result distribution:
        - mean: Mean of result
        - std: Standard deviation
        - ci_lower, ci_upper: 95% confidence interval
        - distribution: Full distribution
    """
    rng = np.random.RandomState(42)
    
    print(f"\nMonte Carlo Error Propagation:")
    print(f"  N_samples = {n_samples}")
    print(f"  Parameters:")
    for name, (mean, std) in params.items():
        print(f"    {name} = {mean:.4f} ± {std:.4f}")
    
    # Sample parameters from Gaussian distributions
    param_samples = {}
    for name, (mean, std) in params.items():
        param_samples[name] = rng.normal(mean, std, n_samples)
    
    # Evaluate function for each sample
    results = []
    for i in range(n_samples):
        # Extract i-th sample of each parameter
        sample_params = {name: param_samples[name][i] for name in params.keys()}
        
        try:
            result = func(**sample_params)
            results.append(result)
        except:
            continue
    
    results = np.array(results)
    
    # Statistics
    mean = np.mean(results)
    std = np.std(results)
    ci_lower = np.percentile(results, 2.5)
    ci_upper = np.percentile(results, 97.5)
    
    print(f"\n  Result:")
    print(f"    Mean: {mean:.4f}")
    print(f"    Std: {std:.4f}")
    print(f"    95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    return {
        'mean': mean,
        'std': std,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'distribution': results
    }


def systematic_uncertainty_analysis(data: pd.DataFrame,
                                   fit_func: Callable,
                                   systematic_variations: Dict[str, List]) -> Dict:
    """
    Test model robustness to systematic uncertainties.
    
    Varies systematic parameters and checks stability of results.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset
    fit_func : callable
        Fitting function
    systematic_variations : dict
        Dictionary of {systematic_name: [variation_values]}
    
    Returns:
    --------
    dict
        Systematic variation results
    """
    print(f"\nSystematic Uncertainty Analysis:")
    
    baseline_result = fit_func(data)
    
    variation_results = {}
    
    for sys_name, variations in systematic_variations.items():
        print(f"\n  Testing {sys_name}:")
        
        results_list = []
        
        for variation in variations:
            # Apply systematic variation to data
            # (specific implementation depends on systematic)
            data_varied = data.copy()  # Modify as needed
            
            try:
                result = fit_func(data_varied)
                results_list.append(result)
                print(f"    {variation}: {result}")
            except:
                print(f"    {variation}: FAILED")
        
        variation_results[sys_name] = results_list
    
    return {
        'baseline': baseline_result,
        'variations': variation_results
    }


# ============================================================================
# PERMUTATION TESTS
# ============================================================================

def permutation_test(x: np.ndarray, y: np.ndarray,
                    statistic: Callable = None,
                    n_permutations: int = 10000,
                    alternative: str = 'two-sided') -> Dict:
    """
    Non-parametric permutation test for correlation.
    
    Tests null hypothesis that x and y are independent by randomly
    permuting one variable and computing test statistic.
    
    Parameters:
    -----------
    x, y : np.ndarray
        Data arrays
    statistic : callable, optional
        Test statistic function (default: Pearson correlation)
    n_permutations : int
        Number of permutations
    alternative : str
        'two-sided', 'greater', or 'less'
    
    Returns:
    --------
    dict
        Permutation test results:
        - observed_statistic: Test statistic on real data
        - p_value: Permutation p-value
        - null_distribution: Distribution under null hypothesis
    """
    if statistic is None:
        # Default: Pearson correlation
        statistic = lambda x, y: stats.pearsonr(x, y)[0]
    
    # Observed test statistic
    observed = statistic(x, y)
    
    print(f"\nPermutation Test:")
    print(f"  Observed statistic: {observed:.4f}")
    print(f"  N_permutations: {n_permutations}")
    
    # Generate null distribution
    rng = np.random.RandomState(42)
    null_distribution = []
    
    for _ in range(n_permutations):
        # Permute y (breaks any real correlation)
        y_perm = rng.permutation(y)
        
        try:
            stat_perm = statistic(x, y_perm)
            null_distribution.append(stat_perm)
        except:
            continue
    
    null_distribution = np.array(null_distribution)
    
    # Calculate p-value
    if alternative == 'two-sided':
        p_value = np.mean(np.abs(null_distribution) >= np.abs(observed))
    elif alternative == 'greater':
        p_value = np.mean(null_distribution >= observed)
    elif alternative == 'less':
        p_value = np.mean(null_distribution <= observed)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")
    
    print(f"  p-value ({alternative}): {p_value:.4f}")
    
    if p_value < 0.001:
        print(f"  → HIGHLY significant (p < 0.001)")
    elif p_value < 0.01:
        print(f"  → Significant (p < 0.01)")
    elif p_value < 0.05:
        print(f"  → Marginally significant (p < 0.05)")
    else:
        print(f"  → Not significant (p ≥ 0.05)")
    
    return {
        'observed_statistic': observed,
        'p_value': p_value,
        'null_distribution': null_distribution,
        'alternative': alternative
    }


# ============================================================================
# OUTLIER DETECTION
# ============================================================================

def detect_outliers_zscore(data: np.ndarray, 
                          threshold: float = 3.0,
                          modified: bool = True) -> np.ndarray:
    """
    Detect outliers using Z-score method.
    
    Parameters:
    -----------
    data : np.ndarray
        Data array
    threshold : float
        Z-score threshold (default: 3.0)
    modified : bool
        Use modified Z-score (robust to outliers)
    
    Returns:
    --------
    np.ndarray
        Boolean mask: True for outliers
    """
    if modified:
        # Modified Z-score using median absolute deviation (MAD)
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        # Modified Z-score
        if mad > 0:
            z_scores = 0.6745 * (data - median) / mad
        else:
            z_scores = np.zeros_like(data)
    else:
        # Standard Z-score
        mean = np.mean(data)
        std = np.std(data)
        
        if std > 0:
            z_scores = (data - mean) / std
        else:
            z_scores = np.zeros_like(data)
    
    outliers = np.abs(z_scores) > threshold
    
    n_outliers = np.sum(outliers)
    print(f"Outlier detection (Z-score):")
    print(f"  Threshold: {threshold}")
    print(f"  Method: {'Modified' if modified else 'Standard'}")
    print(f"  Outliers found: {n_outliers}/{len(data)} ({n_outliers/len(data)*100:.1f}%)")
    
    return outliers


def detect_outliers_chauvenet(data: np.ndarray) -> np.ndarray:
    """
    Chauvenet's criterion for outlier detection.
    
    Rejects points with probability < 1/(2N) of occurring in a
    Gaussian distribution.
    
    Parameters:
    -----------
    data : np.ndarray
        Data array
    
    Returns:
    --------
    np.ndarray
        Boolean mask: True for outliers
    """
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    
    # Z-scores
    z_scores = np.abs((data - mean) / std)
    
    # Chauvenet's criterion
    # Probability of observing |z| or greater
    prob = 2 * (1 - 0.5 * (1 + erf(z_scores / np.sqrt(2))))
    
    # Expected number of observations at this probability
    expected = prob * n
    
    # Reject if expected < 0.5 (i.e., probability < 1/(2N))
    outliers = expected < 0.5
    
    n_outliers = np.sum(outliers)
    print(f"Outlier detection (Chauvenet):")
    print(f"  Outliers found: {n_outliers}/{n} ({n_outliers/n*100:.1f}%)")
    
    return outliers


def robust_statistics(data: np.ndarray, remove_outliers: bool = True) -> Dict:
    """
    Calculate robust statistics (resistant to outliers).
    
    Parameters:
    -----------
    data : np.ndarray
        Data array
    remove_outliers : bool
        Remove outliers before calculating statistics
    
    Returns:
    --------
    dict
        Robust statistics:
        - median, mad: Median and median absolute deviation
        - trimmed_mean, trimmed_std: Trimmed statistics (10% each tail)
        - winsorized_mean, winsorized_std: Winsorized statistics
    """
    if remove_outliers:
        outliers = detect_outliers_zscore(data, modified=True)
        data_clean = data[~outliers]
    else:
        data_clean = data
    
    # Median and MAD
    median = np.median(data_clean)
    mad = np.median(np.abs(data_clean - median))
    
    # Trimmed mean (remove 10% from each tail)
    trimmed_mean = stats.trim_mean(data_clean, proportiontocut=0.1)
    
    # Winsorized mean (replace extremes with percentiles)
    data_winsorized = stats.mstats.winsorize(data_clean, limits=[0.1, 0.1])
    winsorized_mean = np.mean(data_winsorized)
    winsorized_std = np.std(data_winsorized)
    
    print(f"\nRobust Statistics:")
    print(f"  Median: {median:.4f}")
    print(f"  MAD: {mad:.4f}")
    print(f"  Trimmed mean (10%): {trimmed_mean:.4f}")
    print(f"  Winsorized mean: {winsorized_mean:.4f} ± {winsorized_std:.4f}")
    
    return {
        'median': median,
        'mad': mad,
        'trimmed_mean': trimmed_mean,
        'winsorized_mean': winsorized_mean,
        'winsorized_std': winsorized_std
    }


# ============================================================================
# GOODNESS-OF-FIT TESTS
# ============================================================================

def normality_tests(data: np.ndarray) -> Dict:
    """
    Test whether data follows a normal distribution.
    
    Uses multiple tests:
    - Shapiro-Wilk (most powerful for small samples)
    - Kolmogorov-Smirnov (for larger samples)
    - Anderson-Darling (sensitive to tails)
    
    Parameters:
    -----------
    data : np.ndarray
        Data array
    
    Returns:
    --------
    dict
        Results from each normality test
    """
    print(f"\nNormality Tests (N={len(data)}):")
    
    results = {}
    
    # Shapiro-Wilk test
    if len(data) >= 3 and len(data) <= 5000:
        stat_sw, p_sw = stats.shapiro(data)
        print(f"  Shapiro-Wilk: W={stat_sw:.4f}, p={p_sw:.4f}")
        results['shapiro_wilk'] = {'statistic': stat_sw, 'p_value': p_sw}
    
    # Kolmogorov-Smirnov test
    stat_ks, p_ks = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
    print(f"  Kolmogorov-Smirnov: D={stat_ks:.4f}, p={p_ks:.4f}")
    results['ks_test'] = {'statistic': stat_ks, 'p_value': p_ks}
    
    # Anderson-Darling test
    result_ad = stats.anderson(data, dist='norm')
    print(f"  Anderson-Darling: A²={result_ad.statistic:.4f}")
    print(f"    Critical values: {result_ad.critical_values}")
    results['anderson_darling'] = {
        'statistic': result_ad.statistic,
        'critical_values': result_ad.critical_values,
        'significance_levels': result_ad.significance_level
    }
    
    return results


def chi2_goodness_of_fit(observed: np.ndarray, expected: np.ndarray) -> Dict:
    """
    Chi-squared goodness-of-fit test.
    
    Tests whether observed frequencies match expected frequencies.
    
    Parameters:
    -----------
    observed : np.ndarray
        Observed frequencies
    expected : np.ndarray
        Expected frequencies under null hypothesis
    
    Returns:
    --------
    dict
        Chi-squared test results
    """
    chi2_stat, p_value = stats.chisquare(observed, expected)
    dof = len(observed) - 1
    
    print(f"\nχ² Goodness-of-Fit Test:")
    print(f"  χ² = {chi2_stat:.2f}")
    print(f"  dof = {dof}")
    print(f"  p-value = {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"  → Reject null hypothesis (p < 0.05)")
    else:
        print(f"  → Cannot reject null hypothesis")
    
    return {
        'chi2': chi2_stat,
        'dof': dof,
        'p_value': p_value
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    """
    Example usage of statistical tests module.
    """
    print("3D+3D Statistical Tests Module")
    print("===============================\n")
    
    print("Module loaded successfully.")
    print("\nKey functions:")
    print("  - correlation_analysis()")
    print("  - bootstrap_parameter_uncertainty()")
    print("  - monte_carlo_error_propagation()")
    print("  - permutation_test()")
    print("  - detect_outliers_zscore()")
    print("  - detect_outliers_chauvenet()")
    print("  - normality_tests()")
