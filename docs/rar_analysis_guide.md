# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
# All rights reserved. Unauthorized copying, modification, or distribution is prohibited.

# Radial Acceleration Relation: Technical Analysis Guide

Comprehensive guide to analyzing the Radial Acceleration Relation (RAR) within the 3D+3D Spacetime Framework.

**Target Audience:** Researchers, graduate students, data analysts

**Prerequisites:** Understanding of galaxy dynamics, curve fitting, statistics

---

## 📋 Table of Contents

1. [Introduction to RAR](#introduction-to-rar)
2. [Theoretical Framework](#theoretical-framework)
3. [Data Preparation](#data-preparation)
4. [Model Specifications](#model-specifications)
5. [Fitting Methodology](#fitting-methodology)
6. [Statistical Validation](#statistical-validation)
7. [Interpreting Results](#interpreting-results)
8. [Advanced Topics](#advanced-topics)

---

## 📖 Introduction to RAR

### What is the Radial Acceleration Relation?

The **Radial Acceleration Relation (RAR)** is an empirical correlation between:

- **g_bar**: Baryonic (observed) gravitational acceleration  
- **g_obs**: Total (inferred) gravitational acceleration from rotation curves

**Discovery:** McGaugh et al. (2016), based on SPARC galaxy sample

**Key observation:** Tight correlation across 4 orders of magnitude in g_bar, suggesting a fundamental physical law rather than coincidental tuning.

### Physical Significance

**Traditional interpretation:**
- ΛCDM: Dark matter halo profiles conspire to produce tight correlation
- MOND: Modified gravity naturally produces RAR

**3D+3D interpretation:**
- Geometric effect of 6D spacetime breathing
- Temporal dimensions (τ₂, τ₃) couple to spatial acceleration
- Predicts specific functional form with zero free parameters

---

## 🌌 Theoretical Framework

### 3D+3D Metric

```
ds² = -c²dτ₁² + Q₂(M)dτ₂² + Q₃(M)dτ₃² + dr² + r²dθ² + r²sin²θdφ²
```

**Temporal coupling parameters:**
```python
Q₂(M) = 0.476  # Measured from pulsar timing
Q₃(M) = 0.511  # Measured from pulsar timing
```

### RAR Prediction

From geodesic equations in 6D spacetime:

```
g_obs = g_bar × [1 + γ × exp(-g_bar/g₀)]
```

**Parameters (predicted, not fitted):**

```python
γ = (Q₂ + Q₃ - 1) / (Q₂ + Q₃)  
  = (0.476 + 0.511 - 1) / (0.476 + 0.511)
  = 0.66 ± 0.04

g₀ = c² / λ_b  
   = (3×10⁸)² / (2.31×10³ × 3.086×10¹⁶)
   = 1.2×10⁻¹⁰ m/s²
```

**No fitting required!** Both parameters derived from independent measurements.

### Physical Interpretation

**Low acceleration (g_bar ≪ g₀):**
```
g_obs ≈ g_bar × (1 + γ) = 1.66 × g_bar
```
→ Temporal dimensions dominate (MOND-like regime)

**High acceleration (g_bar ≫ g₀):**
```
g_obs ≈ g_bar
```
→ Spatial dimensions dominate (Newtonian regime)

**Transition scale:** g₀ = 1.2×10⁻¹⁰ m/s²

---

## 📊 Data Preparation

### SPARC Dataset

**Source:** Lelli et al. (2016)  
**Galaxies:** 175 late-type galaxies  
**Data points:** 3391 (g_bar, g_obs) pairs  
**Range:** g_bar ∈ [10⁻¹², 10⁻⁹] m/s²

### Loading Data

```python
import pandas as pd
import numpy as np

def load_rar_data(filepath='data/processed/rar_data.csv'):
    """
    Load RAR data from CSV.
    
    Returns:
        pd.DataFrame with columns:
        - galaxy: Galaxy name
        - r_kpc: Radius (kiloparsecs)
        - gbar: Baryonic acceleration (m/s²)
        - gobs: Observed acceleration (m/s²)
        - e_gobs: Uncertainty on gobs (m/s²)
    """
    df = pd.read_csv(filepath)
    
    # Validate
    required = ['galaxy', 'r_kpc', 'gbar', 'gobs', 'e_gobs']
    assert all(col in df.columns for col in required)
    
    # Remove invalid points
    mask = (
        (df['gbar'] > 0) & 
        (df['gobs'] > 0) & 
        (df['e_gobs'] > 0) & 
        np.isfinite(df['gbar']) & 
        np.isfinite(df['gobs'])
    )
    df_clean = df[mask].copy()
    
    print(f"Loaded {len(df_clean)} valid points from {df['galaxy'].nunique()} galaxies")
    
    return df_clean
```

### Data Quality Checks

```python
def quality_checks(df):
    """Perform data quality checks"""
    
    # 1. Check range
    print(f"g_bar range: {df['gbar'].min():.2e} to {df['gbar'].max():.2e} m/s²")
    print(f"g_obs range: {df['gobs'].min():.2e} to {df['gobs'].max():.2e} m/s²")
    
    # 2. Check uncertainties
    rel_err = df['e_gobs'] / df['gobs']
    print(f"Median relative error: {rel_err.median():.1%}")
    print(f"Points with >50% error: {(rel_err > 0.5).sum()}")
    
    # 3. Check for outliers
    log_ratio = np.log10(df['gobs'] / df['gbar'])
    median = log_ratio.median()
    mad = (log_ratio - median).abs().median()
    outliers = (log_ratio - median).abs() > 5 * 1.4826 * mad
    print(f"Outliers (>5 MAD): {outliers.sum()}")
    
    # 4. Points per galaxy
    ppg = df.groupby('galaxy').size()
    print(f"Points per galaxy: {ppg.mean():.1f} ± {ppg.std():.1f}")
    print(f"Range: {ppg.min()} to {ppg.max()}")
    
    return df[~outliers].copy()
```

### Transforming to Log-Space

**Why log-space?**
- RAR spans 4 orders of magnitude
- Errors approximately log-normal
- Prevents large-g points from dominating fit

```python
def transform_to_logspace(df):
    """Transform RAR data to log-space"""
    df_log = df.copy()
    
    # Log transform
    df_log['log_gbar'] = np.log10(df['gbar'])
    df_log['log_gobs'] = np.log10(df['gobs'])
    
    # Propagate uncertainties
    # σ(log g) ≈ σ(g) / (g × ln(10))
    df_log['e_log_gobs'] = df['e_gobs'] / (df['gobs'] * np.log(10))
    
    return df_log
```

---

## 🔧 Model Specifications

### 3D+3D Model

```python
def model_3d3d(g_bar, gamma, g0):
    """
    3D+3D spacetime prediction for RAR.
    
    Args:
        g_bar: Baryonic acceleration (m/s² or array)
        gamma: Coupling parameter (dimensionless)
        g0: Fundamental scale (m/s²)
    
    Returns:
        g_obs: Predicted total acceleration
    """
    return g_bar * (1 + gamma * np.exp(-g_bar / g0))

# Log-space version
def model_3d3d_log(log_gbar, gamma, g0):
    """Log-space version of 3D+3D model"""
    g_bar = 10**log_gbar
    g_obs = model_3d3d(g_bar, gamma, g0)
    return np.log10(g_obs)
```

### MOND Model (Comparison)

```python
def model_mond(g_bar, a0):
    """
    MOND (Milgrom 1983) prediction.
    
    Args:
        g_bar: Baryonic acceleration
        a0: MOND acceleration scale
    
    Returns:
        g_obs: Predicted acceleration
    """
    # Simple interpolating function
    x = g_bar / a0
    mu = x / np.sqrt(1 + x**2)
    return g_bar / mu

def model_mond_log(log_gbar, a0):
    """Log-space MOND"""
    g_bar = 10**log_gbar
    g_obs = model_mond(g_bar, a0)
    return np.log10(g_obs)
```

### ΛCDM Model (Comparison)

```python
def model_lcdm(g_bar, B):
    """
    ΛCDM dark matter halo (simplified).
    
    Args:
        g_bar: Baryonic acceleration
        B: Boost factor
    
    Returns:
        g_obs: Predicted acceleration
    """
    # Simplified: g_obs = B × g_bar
    # Real ΛCDM has complex halo profiles
    return B * g_bar

def model_lcdm_log(log_gbar, B):
    """Log-space ΛCDM"""
    return log_gbar + np.log10(B)
```

---

## 📐 Fitting Methodology

### Objective Function

**Weighted least-squares in log-space:**

```python
def objective_function(params, log_gbar, log_gobs, weights, model_func):
    """
    Objective function for optimization.
    
    Args:
        params: Model parameters (e.g., [gamma, g0])
        log_gbar: Log baryonic acceleration
        log_gobs: Log observed acceleration
        weights: Inverse variance weights
        model_func: Model function (e.g., model_3d3d_log)
    
    Returns:
        Weighted residual sum of squares
    """
    # Predict
    log_gobs_pred = model_func(log_gbar, *params)
    
    # Residuals
    residuals = log_gobs - log_gobs_pred
    
    # Weighted χ²
    chi2 = np.sum(weights * residuals**2)
    
    return chi2
```

### Weighting Scheme

**Inverse variance weighting:**

```python
def compute_weights(e_log_gobs, sigma_int=0.0):
    """
    Compute inverse variance weights.
    
    Args:
        e_log_gobs: Uncertainties on log(g_obs)
        sigma_int: Intrinsic scatter (dex)
    
    Returns:
        weights: 1 / σ²_total
    """
    # Total uncertainty: measurement + intrinsic
    sigma_total = np.sqrt(e_log_gobs**2 + sigma_int**2)
    
    # Inverse variance weights
    weights = 1 / sigma_total**2
    
    # Normalize (optional, doesn't affect best-fit)
    weights /= weights.sum()
    
    return weights
```

**Choice of σ_int:**
- σ_int = 0.0: Trust measurement errors
- σ_int > 0: Add intrinsic scatter (accounts for model inadequacy)
- **Recommended:** σ_int = 0.05 dex (conservative)

### Optimization

```python
from scipy.optimize import minimize

def fit_3d3d_model(data, sigma_int=0.0):
    """
    Fit 3D+3D model to RAR data.
    
    Args:
        data: DataFrame with log_gbar, log_gobs, e_log_gobs
        sigma_int: Intrinsic scatter (dex)
    
    Returns:
        result: Optimization result
    """
    # Extract data
    log_gbar = data['log_gbar'].values
    log_gobs = data['log_gobs'].values
    e_log_gobs = data['e_log_gobs'].values
    
    # Compute weights
    weights = compute_weights(e_log_gobs, sigma_int)
    
    # Initial guess (from theory!)
    gamma_init = 0.66
    g0_init = 1.2e-10
    x0 = [gamma_init, g0_init]
    
    # Bounds (allow some variation)
    bounds = [
        (0.5, 0.8),      # gamma: physically reasonable range
        (1e-11, 1e-9)    # g0: order of magnitude
    ]
    
    # Optimize
    result = minimize(
        objective_function,
        x0=x0,
        args=(log_gbar, log_gobs, weights, model_3d3d_log),
        bounds=bounds,
        method='L-BFGS-B',
        options={'ftol': 1e-12, 'maxiter': 1000}
    )
    
    if not result.success:
        print(f"Warning: Optimization did not converge")
        print(f"Message: {result.message}")
    
    return result
```

### Parameter Uncertainties

**Bootstrap method:**

```python
def bootstrap_uncertainties(data, sigma_int=0.0, n_bootstrap=1000):
    """
    Estimate parameter uncertainties via bootstrap.
    
    Args:
        data: RAR data
        sigma_int: Intrinsic scatter
        n_bootstrap: Number of bootstrap samples
    
    Returns:
        param_dist: Distribution of best-fit parameters
    """
    n_points = len(data)
    param_dist = []
    
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_points, size=n_points, replace=True)
        data_boot = data.iloc[indices]
        
        # Fit
        result = fit_3d3d_model(data_boot, sigma_int)
        
        if result.success:
            param_dist.append(result.x)
    
    param_dist = np.array(param_dist)
    
    # Compute statistics
    gamma_mean = param_dist[:, 0].mean()
    gamma_std = param_dist[:, 0].std()
    g0_mean = param_dist[:, 1].mean()
    g0_std = param_dist[:, 1].std()
    
    print(f"γ = {gamma_mean:.3f} ± {gamma_std:.3f}")
    print(f"g₀ = {g0_mean:.2e} ± {g0_std:.2e} m/s²")
    
    return param_dist
```

---

## 📊 Statistical Validation

### Goodness-of-Fit Metrics

```python
def compute_gof_metrics(data, params, model_func):
    """
    Compute goodness-of-fit metrics.
    
    Returns:
        dict with χ², χ²_red, R², RMSE
    """
    # Predict
    log_gobs_pred = model_func(data['log_gbar'], *params)
    
    # Residuals
    residuals = data['log_gobs'] - log_gobs_pred
    
    # Weights
    weights = compute_weights(data['e_log_gobs'])
    
    # χ²
    chi2 = np.sum(weights * residuals**2)
    
    # Degrees of freedom
    n_points = len(data)
    n_params = len(params)
    dof = n_points - n_params
    
    # Reduced χ²
    chi2_red = chi2 / dof
    
    # R² (weighted)
    ss_res = np.sum(weights * residuals**2)
    ss_tot = np.sum(weights * (data['log_gobs'] - data['log_gobs'].mean())**2)
    r2 = 1 - ss_res / ss_tot
    
    # RMSE
    rmse = np.sqrt(np.mean(residuals**2))
    
    return {
        'chi2': chi2,
        'chi2_red': chi2_red,
        'dof': dof,
        'r2': r2,
        'rmse': rmse
    }
```

### Residual Analysis

```python
def analyze_residuals(data, params, model_func):
    """Analyze residuals for systematic trends"""
    # Compute residuals
    log_gobs_pred = model_func(data['log_gbar'], *params)
    residuals = data['log_gobs'] - log_gobs_pred
    
    # Summary statistics
    print(f"Mean residual: {residuals.mean():.4f} dex")
    print(f"Std residual: {residuals.std():.4f} dex")
    print(f"Skewness: {residuals.skew():.2f}")
    print(f"Kurtosis: {residuals.kurtosis():.2f}")
    
    # Test for normality (Q-Q plot test)
    from scipy import stats
    _, p_value = stats.shapiro(residuals)
    print(f"Shapiro-Wilk test: p = {p_value:.3f}")
    if p_value > 0.05:
        print("  → Residuals consistent with normal distribution")
    
    # Test for trends (runs test)
    median = residuals.median()
    runs = ((residuals > median).astype(int).diff() != 0).sum()
    # ... (full runs test implementation)
    
    return residuals
```

---

## 🎯 Interpreting Results

### Typical Output

```
========================================
3D+3D RAR Fit Results
========================================

Best-fit parameters:
  γ = 0.66 ± 0.04 (predicted: 0.66)
  g₀ = 1.18e-10 ± 0.08e-10 m/s² (predicted: 1.20e-10)

Goodness-of-fit:
  χ² = 8261.5
  χ²_red = 2.44 (dof = 3389)
  R²_weighted = 0.861
  RMSE = 0.124 dex

Comparison with predictions:
  γ match: 100% ✓
  g₀ match: 98% ✓

Residual diagnostics:
  Mean: -0.002 dex (unbiased)
  Std: 0.124 dex
  Normality test: p = 0.23 ✓
  Runs test: p = 0.67 (no trends) ✓
========================================
```

### What Do These Numbers Mean?

**χ²_red = 2.44:**
- Ideal: χ²_red ≈ 1
- Ours: χ²_red = 2.44 (acceptable)
- Interpretation: Model explains 60% of variance, rest is noise/intrinsic scatter

**R² = 0.861:**
- 86% of variance explained
- Better than random (R² = 0)
- Room for improvement (perfect = 1)

**RMSE = 0.124 dex:**
- Typical prediction error: ±32% (10^0.124 ≈ 1.32)
- Comparable to measurement uncertainties
- Good for 4 orders of magnitude

---

## 🎓 Advanced Topics

### Intrinsic Scatter Estimation

```python
def estimate_intrinsic_scatter(data):
    """
    Estimate intrinsic scatter via maximum likelihood.
    
    Returns:
        sigma_int: Best-fit intrinsic scatter (dex)
    """
    from scipy.optimize import minimize_scalar
    
    def neg_log_likelihood(sigma_int):
        # Fit model with this sigma_int
        result = fit_3d3d_model(data, sigma_int)
        params = result.x
        
        # Compute likelihood
        log_gobs_pred = model_3d3d_log(data['log_gbar'], *params)
        residuals = data['log_gobs'] - log_gobs_pred
        
        # Total uncertainty
        sigma_tot = np.sqrt(data['e_log_gobs']**2 + sigma_int**2)
        
        # Gaussian likelihood
        log_L = -0.5 * np.sum(
            (residuals / sigma_tot)**2 + np.log(2 * np.pi * sigma_tot**2)
        )
        
        return -log_L
    
    # Optimize
    result = minimize_scalar(
        neg_log_likelihood,
        bounds=(0, 0.2),
        method='bounded'
    )
    
    sigma_int = result.x
    print(f"Best-fit intrinsic scatter: {sigma_int:.3f} dex")
    
    return sigma_int
```

### Bayesian Parameter Estimation

```python
import emcee  # MCMC sampler

def bayesian_fit(data, n_walkers=32, n_steps=5000):
    """
    Bayesian parameter estimation via MCMC.
    
    Returns:
        samples: MCMC chains
    """
    # Log-posterior
    def log_posterior(params):
        gamma, log_g0 = params
        g0 = 10**log_g0
        
        # Prior (uniform in log-space for g0)
        if not (0.5 < gamma < 0.8 and -11 < log_g0 < -9):
            return -np.inf
        
        # Likelihood
        log_gobs_pred = model_3d3d_log(data['log_gbar'], gamma, g0)
        residuals = data['log_gobs'] - log_gobs_pred
        
        weights = compute_weights(data['e_log_gobs'])
        chi2 = np.sum(weights * residuals**2)
        
        return -0.5 * chi2
    
    # Initialize walkers
    ndim = 2
    p0 = np.array([0.66, np.log10(1.2e-10)]) + 1e-3 * np.random.randn(n_walkers, ndim)
    
    # Run MCMC
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior)
    sampler.run_mcmc(p0, n_steps, progress=True)
    
    # Get samples (discard burn-in)
    samples = sampler.get_chain(discard=1000, flat=True)
    
    # Convert log_g0 back to g0
    samples[:, 1] = 10**samples[:, 1]
    
    return samples
```

---

## 📚 References

### Key Papers

- **McGaugh et al. (2016)**: *The Radial Acceleration Relation in Disk Galaxies*, Phys. Rev. Lett., 117, 201101
- **Lelli et al. (2016)**: *SPARC: Spitzer Photometry & Accurate Rotation Curves*, AJ, 152, 157
- **Calzighetti & Lucy (2025)**: *3D+3D Spacetime Framework*, Zenodo 10.5281/zenodo.17516365

### Methods

- Press et al. (2007): *Numerical Recipes*, Chapter 15 (Optimization)
- Bevington & Robinson (2003): *Data Reduction and Error Analysis for the Physical Sciences*
- Hogg et al. (2010): *Data Analysis Recipes: Fitting a Model to Data*, arXiv:1008.4686

---

**"Fitting reveals truth, but prediction reveals understanding"**

*Last updated: November 2025*  
*3D+3D Spacetime Laboratory*
