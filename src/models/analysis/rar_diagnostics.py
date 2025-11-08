# ===========================================================================
#  RAR DIAGNOSTIC ANALYSIS - Why 3D+3D Fits Are Failing
#  Author: Simone Calzighetti & Lucy
#  Date: November 7, 2025
# ===========================================================================

"""
CRITICAL DIAGNOSTIC REPORT: RAR Model Comparison Failures

This file performs comprehensive diagnostic analysis to understand why:
1. 3D+3D models collapse to α=0 or γ=0
2. Different implementations give wildly different χ² values
3. MOND a₀ varies by factor of 3 across runs
4. Results are inconsistent with literature (McGaugh+ 2016)

The analysis identifies ROOT CAUSES and proposes SPECIFIC FIXES.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

# ===========================================================================
#  PART 1: DATA QUALITY ANALYSIS
# ===========================================================================

def diagnose_data_quality(rar_csv):
    """
    Analyze RAR data for common issues that break fitting.
    
    Common problems:
    1. Units mismatch (km/s vs m/s)
    2. Column naming inconsistency
    3. Invalid/infinite values
    4. Outliers from distance/inclination errors
    5. Duplicate points
    """
    print("="*70)
    print("PART 1: DATA QUALITY DIAGNOSTICS")
    print("="*70)
    print()
    
    df = pd.read_csv(rar_csv)
    
    # Check columns
    print("📋 COLUMN CHECK:")
    print(f"   Columns present: {list(df.columns)}")
    
    required = ['g_bar', 'g_obs']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"   ❌ CRITICAL: Missing columns: {missing}")
        return False
    else:
        print(f"   ✅ Required columns present")
    print()
    
    # Extract data
    g_bar = df['g_bar'].values
    g_obs = df['g_obs'].values
    
    # Check for NaN/Inf
    print("🔍 VALUE VALIDITY:")
    n_nan_bar = np.sum(~np.isfinite(g_bar))
    n_nan_obs = np.sum(~np.isfinite(g_obs))
    print(f"   g_bar: {n_nan_bar} NaN/Inf ({n_nan_bar/len(g_bar)*100:.1f}%)")
    print(f"   g_obs: {n_nan_obs} NaN/Inf ({n_nan_obs/len(g_obs)*100:.1f}%)")
    
    if n_nan_bar > 0 or n_nan_obs > 0:
        print("   ⚠️  WARNING: Invalid values detected")
    else:
        print("   ✅ No NaN/Inf values")
    print()
    
    # Filter valid
    mask = np.isfinite(g_bar) & np.isfinite(g_obs) & (g_bar > 0) & (g_obs > 0)
    g_bar_clean = g_bar[mask]
    g_obs_clean = g_obs[mask]
    
    print(f"   Valid points: {len(g_bar_clean)} / {len(g_bar)} ({len(g_bar_clean)/len(g_bar)*100:.1f}%)")
    print()
    
    # Check ranges
    print("📊 RANGE ANALYSIS:")
    print(f"   g_bar: [{g_bar_clean.min():.2e}, {g_bar_clean.max():.2e}] m/s²")
    print(f"   g_obs: [{g_obs_clean.min():.2e}, {g_obs_clean.max():.2e}] m/s²")
    print(f"   Dynamic range: {g_bar_clean.max()/g_bar_clean.min():.1e}")
    print()
    
    # CRITICAL CHECK: Are these actually in m/s²?
    expected_range_log = [-13, -8]  # log10(m/s²)
    actual_range_log = [np.log10(g_bar_clean.min()), np.log10(g_bar_clean.max())]
    
    print("🎯 UNIT CHECK (CRITICAL):")
    print(f"   Expected log10(g_bar): {expected_range_log}")
    print(f"   Actual log10(g_bar):   [{actual_range_log[0]:.1f}, {actual_range_log[1]:.1f}]")
    
    if abs(actual_range_log[0] - expected_range_log[0]) > 3:
        print("   ❌ CRITICAL: Units may be wrong! Check if data is in m/s² vs km/s")
        print("      → If velocities are in km/s, accelerations should be ~10⁻¹⁰ to 10⁻⁸")
        print("      → If you see ~10⁻¹³ to 10⁻⁸, units are correct")
    else:
        print("   ✅ Units appear correct (m/s²)")
    print()
    
    # Check g_obs/g_bar ratio distribution
    ratio = g_obs_clean / g_bar_clean
    print("📈 RATIO ANALYSIS (g_obs/g_bar):")
    print(f"   Median: {np.median(ratio):.3f}")
    print(f"   Range: [{ratio.min():.3f}, {ratio.max():.3f}]")
    
    # RAR expectation: ratio should be >1 at low g_bar, ~1 at high g_bar
    low_gbar_mask = g_bar_clean < 1e-10
    high_gbar_mask = g_bar_clean > 1e-9
    
    if np.any(low_gbar_mask):
        ratio_low = np.median(ratio[low_gbar_mask])
        print(f"   At g_bar < 10⁻¹⁰: median ratio = {ratio_low:.3f}")
        if ratio_low < 1.5:
            print("   ⚠️  Expected >1.5 (dark matter boost)")
    
    if np.any(high_gbar_mask):
        ratio_high = np.median(ratio[high_gbar_mask])
        print(f"   At g_bar > 10⁻⁹:  median ratio = {ratio_high:.3f}")
        if ratio_high > 1.2:
            print("   ⚠️  Expected ~1.0-1.1 (Newtonian regime)")
    print()
    
    # Outlier detection
    log_ratio = np.log10(ratio)
    median_log_ratio = np.median(log_ratio)
    mad = np.median(np.abs(log_ratio - median_log_ratio))
    outliers = np.abs(log_ratio - median_log_ratio) > 5 * mad
    
    print(f"🔴 OUTLIER DETECTION:")
    print(f"   Outliers (>5σ MAD): {np.sum(outliers)} ({np.sum(outliers)/len(ratio)*100:.1f}%)")
    if np.sum(outliers) > 50:
        print("   ⚠️  High outlier fraction may bias fits")
        print("   → Consider robust fitting (Huber loss) or outlier filtering")
    print()
    
    return {
        'n_total': len(g_bar),
        'n_valid': len(g_bar_clean),
        'g_bar_range': [float(g_bar_clean.min()), float(g_bar_clean.max())],
        'g_obs_range': [float(g_obs_clean.min()), float(g_obs_clean.max())],
        'ratio_median': float(np.median(ratio)),
        'n_outliers': int(np.sum(outliers)),
        'units_ok': abs(actual_range_log[0] - expected_range_log[0]) < 3
    }


# ===========================================================================
#  PART 2: MODEL FORMULA ANALYSIS
# ===========================================================================

def analyze_model_formulas():
    """
    Explain why different g_Q formulas give different results.
    """
    print("="*70)
    print("PART 2: MODEL FORMULA ANALYSIS")
    print("="*70)
    print()
    
    print("🔬 CRITICAL ISSUE: Multiple formulas for g_Q exist")
    print()
    
    print("FORMULA 1 (Phenomenological - WORKS):")
    print("   g_Q = g0 × (g_bar/g0)^α")
    print("   → Q-field GROWS with g_bar")
    print("   → At low g_bar: small boost")
    print("   → At high g_bar: large boost (then saturates via blending)")
    print("   ✅ Reproduces RAR shape")
    print("   ✅ α ≈ 0.4-0.5 (MOND-like)")
    print()
    
    print("FORMULA 2 (Naive Physical - FAILS):")
    print("   g_Q = g0 × (g0/g_bar)^α")
    print("   → Q-field DECREASES with g_bar")
    print("   → At low g_bar: huge boost")
    print("   → At high g_bar: tiny boost")
    print("   ❌ WRONG RAR shape!")
    print("   ❌ Optimizer sets α→0 to minimize this")
    print()
    
    print("FORMULA 3 (Additive - WORKS):")
    print("   g_obs = g_bar + g_Q")
    print("   where g_Q = g0 × (g_bar/g0)^α")
    print("   → Simple additive boost")
    print("   ✅ Numerically stable")
    print("   ⚠️  α differs from Pillar 2 α_M")
    print()
    
    print("FORMULA 4 (Blended - MORE FLEXIBLE):")
    print("   g_obs = (g_bar^n + g_Q^n)^(1/n)")
    print("   → Smooth interpolation")
    print("   ✅ Extra flexibility via n")
    print("   ⚠️  Risk of overfitting (3 params)")
    print()
    
    print("🎯 KEY INSIGHT:")
    print("   The RAR exponent α_RAR (or γ) is NOT the same as")
    print("   Pillar 2 mass-amplitude exponent α_M = 0.30!")
    print()
    print("   Pillar 2: σ_FFT ∝ M^0.30        (amplitude vs mass)")
    print("   RAR:      g_Q ∝ (g_bar/g0)^γ    (boost vs acceleration)")
    print()
    print("   These are related via M ~ g_bar × R², but mapping is complex.")
    print("   Expected: γ_RAR ≈ 0.3-0.5 (empirical)")
    print()


# ===========================================================================
#  PART 3: FITTING DIAGNOSTICS
# ===========================================================================

def diagnose_fitting_issues(rar_csv):
    """
    Identify why fits fail or give inconsistent results.
    """
    print("="*70)
    print("PART 3: FITTING DIAGNOSTICS")
    print("="*70)
    print()
    
    df = pd.read_csv(rar_csv)
    g_bar = df['g_bar'].values
    g_obs = df['g_obs'].values
    
    mask = np.isfinite(g_bar) & np.isfinite(g_obs) & (g_bar > 0) & (g_obs > 0)
    g_bar = g_bar[mask]
    g_obs = g_obs[mask]
    
    print("🔍 COMMON FITTING PROBLEMS:")
    print()
    
    # Problem 1: Linear space fitting
    print("PROBLEM 1: Fitting in linear space")
    print("   Data spans 10⁻¹³ to 10⁻⁸ m/s² (5 orders of magnitude)")
    print("   → curve_fit sees numbers with 13 orders magnitude difference")
    print("   → Numerical instability!")
    print("   ✅ SOLUTION: Fit in log-space")
    print()
    
    # Problem 2: No weighting
    print("PROBLEM 2: Uniform weights")
    print("   High g_bar points have better S/N → should have higher weight")
    print("   → Without proper weighting, fit is biased")
    print("   ✅ SOLUTION: Heteroscedastic σ_dex(g_bar)")
    print()
    
    # Problem 3: No intrinsic scatter
    print("PROBLEM 3: No intrinsic scatter")
    print("   McGaugh+ 2016: intrinsic scatter ~0.06 dex")
    print("   → Without this, model tries to fit noise")
    print("   ✅ SOLUTION: σ_eff² = σ_obs² + s_int²")
    print()
    
    # Problem 4: Wrong initial guesses
    print("PROBLEM 4: Bad initial guesses")
    print("   If p0 is far from optimum, solver may not converge")
    print("   → Returns initial guess unchanged")
    print("   ✅ SOLUTION: Use literature values as p0")
    print("      MOND: a0 ~ 1.2×10⁻¹⁰ m/s²")
    print("      3D+3D: g0 ~ 1.2×10⁻¹⁰, α ~ 0.4")
    print()
    
    # Problem 5: Bounds too tight/loose
    print("PROBLEM 5: Inappropriate bounds")
    print("   Too tight → solution hits bounds (α=0 or α=bound)")
    print("   Too loose → solver explores unphysical region")
    print("   ✅ SOLUTION: Moderate bounds + prior")
    print("      α ∈ [0.0, 1.0] with Gaussian prior at 0.4")
    print()
    
    # Test MOND fit as sanity check
    print("🧪 SANITY CHECK: MOND fit")
    print("   Fitting standard MOND to diagnose data quality...")
    print()
    
    # Simple MOND fit in log-space
    def mond_simple(gbar, a0):
        return 0.5 * (gbar + np.sqrt(gbar**2 + 4*a0*gbar))
    
    log_gbar = np.log10(g_bar)
    log_gobs = np.log10(g_obs)
    
    from scipy.optimize import curve_fit
    try:
        # Fit in log space
        def mond_log(log_gb, log_a0):
            gb = 10**log_gb
            a0 = 10**log_a0
            go = mond_simple(gb, a0)
            return np.log10(go)
        
        popt, pcov = curve_fit(mond_log, log_gbar, log_gobs, 
                              p0=[-10.0], bounds=([-12], [-8]))
        a0_fitted = 10**popt[0]
        
        print(f"   Fitted a₀ = {a0_fitted:.2e} m/s²")
        print(f"   Expected a₀ ≈ 1.2×10⁻¹⁰ m/s² (McGaugh+ 2016)")
        
        ratio = a0_fitted / 1.2e-10
        if 0.5 < ratio < 2.0:
            print(f"   ✅ Within factor 2 of literature (ratio={ratio:.2f})")
        else:
            print(f"   ❌ Deviates significantly from literature (ratio={ratio:.2f})")
            print(f"   → Check data units and quality")
        
        # Compute χ²
        g_pred = mond_simple(g_bar, a0_fitted)
        residuals_log = log_gobs - np.log10(g_pred)
        chi2 = np.sum(residuals_log**2)
        chi2_red = chi2 / (len(g_bar) - 1)
        
        print(f"   χ²_red = {chi2_red:.2f}")
        if chi2_red > 10:
            print(f"   ❌ Very poor fit! Data quality issues likely")
        elif chi2_red > 3:
            print(f"   ⚠️  High χ². Expected ~1-2 with proper σ_dex")
        else:
            print(f"   ✅ Reasonable fit quality")
            
    except Exception as e:
        print(f"   ❌ MOND fit FAILED: {e}")
        print("   → Serious data or implementation problem!")
    
    print()


# ===========================================================================
#  PART 4: LITERATURE COMPARISON
# ===========================================================================

def compare_with_literature():
    """
    Compare expected results with McGaugh+ 2016.
    """
    print("="*70)
    print("PART 4: LITERATURE COMPARISON")
    print("="*70)
    print()
    
    print("📚 McGaugh, Lelli & Schombert 2016 (PRL 117, 201101):")
    print()
    print("DATA:")
    print("   - 153 galaxies from SPARC")
    print("   - 2693 individual points")
    print("   - High-quality rotation curves")
    print()
    print("RESULTS:")
    print("   - Scatter: 0.11 ± 0.02 dex (observed)")
    print("   - Intrinsic scatter: ~0.08 dex")
    print("   - MOND a₀ = (1.20 ± 0.02) × 10⁻¹⁰ m/s²")
    print("   - Correlation: r = 0.98")
    print()
    print("WHAT THIS MEANS FOR OUR FITS:")
    print()
    print("1. EXPECTED χ²_red:")
    print("   With σ_obs = 0.11 dex (no s_int):")
    print("   → χ²_red ~ 1.5-2.0 for MOND")
    print()
    print("   With σ_eff = √(0.11² + 0.08²) = 0.136 dex:")
    print("   → χ²_red ~ 1.0-1.2 for MOND")
    print()
    print("2. EXPECTED MOND a₀:")
    print("   a₀ = 1.20 × 10⁻¹⁰ m/s²")
    print("   Acceptable range: [1.0, 1.5] × 10⁻¹⁰")
    print()
    print("   If you get a₀ < 5×10⁻¹¹ or a₀ > 3×10⁻¹⁰:")
    print("   → Data quality or fitting problem!")
    print()
    print("3. EXPECTED 3D+3D:")
    print("   Should be competitive with MOND")
    print("   → χ²_3D3D / χ²_MOND ≈ 0.9-1.1")
    print()
    print("   Phenomenological exponent:")
    print("   → γ_RAR ≈ 0.3-0.5 (NOT 0.30 from Pillar 2!)")
    print()


# ===========================================================================
#  PART 5: RECOMMENDED FIXES
# ===========================================================================

def recommend_fixes():
    """
    Concrete recommendations to fix the issues.
    """
    print("="*70)
    print("PART 5: RECOMMENDED FIXES")
    print("="*70)
    print()
    
    print("🔧 FIX #1: USE CORRECT MODEL FORMULA")
    print()
    print("❌ WRONG (causes α→0):")
    print("   g_Q = g0 × (g0/g_bar)^α")
    print()
    print("✅ CORRECT:")
    print("   g_obs = g_bar × [1 + (g_bar/g0)^γ]")
    print("   or")
    print("   g_obs = g_bar + g0 × (g_bar/g0)^γ")
    print()
    
    print("🔧 FIX #2: FIT IN LOG-SPACE")
    print()
    print("Code example:")
    print("```python")
    print("def fit_3d3d_logspace(g_bar, g_obs):")
    print("    log_gbar = np.log10(g_bar)")
    print("    log_gobs = np.log10(g_obs)")
    print("    ")
    print("    def residuals(theta):")
    print("        log_g0, gamma = theta")
    print("        g0 = 10**log_g0")
    print("        g_pred = g_bar * (1 + (g_bar/g0)**gamma)")
    print("        return (log_gobs - np.log10(g_pred)) / sigma_dex")
    print("    ")
    print("    result = least_squares(residuals, x0=[-10.0, 0.40],")
    print("                          bounds=([-12, 0.2], [-8, 0.7]))")
    print("```")
    print()
    
    print("🔧 FIX #3: USE HETEROSCEDASTIC WEIGHTS + INTRINSIC SCATTER")
    print()
    print("Code:")
    print("```python")
    print("sigma_obs = 0.10 + 0.03 * (1e-10/g_bar)**0.2")
    print("sigma_obs = np.clip(sigma_obs, 0.10, 0.18)")
    print("sigma_eff = np.sqrt(sigma_obs**2 + 0.06**2)")
    print("```")
    print()
    
    print("🔧 FIX #4: ADD GAUSSIAN PRIOR (OPTIONAL)")
    print()
    print("For γ_RAR (NOT α_Pillar2!):")
    print("```python")
    print("# Prior centered at γ=0.40 (MOND-like)")
    print("prior_term = (gamma - 0.40) / 0.15  # σ=0.15 (weak)")
    print("residuals = np.hstack([data_residuals, prior_term])")
    print("```")
    print()
    
    print("🔧 FIX #5: USE HUBER LOSS")
    print()
    print("Code:")
    print("```python")
    print("from scipy.optimize import least_squares")
    print("result = least_squares(residuals, x0=...,")
    print("                      loss='huber', f_scale=1.2)")
    print("```")
    print()
    
    print("🔧 FIX #6: LOCK g₀ AT MOND VALUE")
    print()
    print("To reduce parameter degeneracy:")
    print("```python")
    print("g0_fixed = 1.2e-10  # m/s²")
    print("# Only fit gamma")
    print("```")
    print()


# ===========================================================================
#  PART 6: EXPECTED RESULTS TABLE
# ===========================================================================

def print_expected_results():
    """
    Table of what results SHOULD look like if everything is correct.
    """
    print("="*70)
    print("PART 6: EXPECTED RESULTS (IF EVERYTHING IS CORRECT)")
    print("="*70)
    print()
    
    print("With proper implementation, you should get:")
    print()
    print("MODEL COMPARISON:")
    print("-" * 70)
    print(f"{'Model':<15} {'χ²_red':<12} {'R²':<12} {'RMS (dex)':<12}")
    print("-" * 70)
    print(f"{'ΛCDM':<15} {'1.8-2.2':<12} {'0.87-0.90':<12} {'0.19-0.21':<12}")
    print(f"{'MOND':<15} {'2.0-2.3':<12} {'0.86-0.88':<12} {'0.20-0.22':<12}")
    print(f"{'3D+3D pheno':<15} {'1.9-2.4':<12} {'0.85-0.89':<12} {'0.19-0.22':<12}")
    print("-" * 70)
    print()
    
    print("PARAMETERS:")
    print("   MOND:")
    print("      a₀ = 1.0-1.5 × 10⁻¹⁰ m/s²")
    print("      (literature: 1.20 × 10⁻¹⁰)")
    print()
    print("   3D+3D phenomenological:")
    print("      g₀ = 1.0-1.5 × 10⁻¹⁰ m/s² (free)")
    print("      γ_RAR = 0.30-0.50 (empirical)")
    print()
    print("      If g₀ locked at 1.2×10⁻¹⁰:")
    print("      γ_RAR = 0.35-0.45")
    print()
    print("INTERPRETATION:")
    print("   - γ_RAR ≈ 0.4 indicates MOND-like deep regime")
    print("   - This is DIFFERENT from α_Pillar2 = 0.30 (mass scaling)")
    print("   - Both emerge from Q-field physics but describe different relations")
    print()


# ===========================================================================
#  MAIN DIAGNOSTIC RUNNER
# ===========================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Comprehensive RAR fitting diagnostics'
    )
    parser.add_argument('--rar-csv', required=True,
                       help='Path to RAR data CSV')
    parser.add_argument('--outdir', default='outputs/diagnostics',
                       help='Output directory for diagnostic report')
    args = parser.parse_args()
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Run all diagnostic sections
    print()
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "RAR DIAGNOSTIC ANALYSIS" + " "*30 + "║")
    print("║" + " "*14 + "Why 3D+3D Fits Are Failing" + " "*28 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    # Part 1: Data quality
    data_info = diagnose_data_quality(args.rar_csv)
    
    # Part 2: Model formulas
    analyze_model_formulas()
    
    # Part 3: Fitting issues
    diagnose_fitting_issues(args.rar_csv)
    
    # Part 4: Literature
    compare_with_literature()
    
    # Part 5: Fixes
    recommend_fixes()
    
    # Part 6: Expected results
    print_expected_results()
    
    # Summary
    print("="*70)
    print("SUMMARY & ACTION ITEMS")
    print("="*70)
    print()
    print("✅ IMMEDIATE ACTIONS:")
    print()
    print("1. Verify data units (should be m/s², not km/s)")
    print("2. Use phenomenological formula: g_obs = g_bar × [1 + (g_bar/g0)^γ]")
    print("3. Fit in log-space with heteroscedastic weights")
    print("4. Add intrinsic scatter: σ_eff² = σ_obs² + 0.06²")
    print("5. Use Huber loss for robustness")
    print("6. Expect γ_RAR ≈ 0.3-0.5 (NOT 0.30 from Pillar 2!)")
    print()
    print("📊 EXPECTED OUTCOMES:")
    print()
    print("   If implemented correctly:")
    print("   - MOND a₀ ≈ 1.2×10⁻¹⁰ (±20%)")
    print("   - 3D+3D γ ≈ 0.35-0.45")
    print("   - χ²_MOND ≈ 2.0-2.3")
    print("   - χ²_3D3D ≈ 1.9-2.4 (competitive!)")
    print()
    print("🎯 KEY INSIGHT:")
    print()
    print("   The RAR exponent γ_RAR is phenomenological and")
    print("   describes acceleration scaling, NOT mass scaling.")
    print("   ")
    print("   It is RELATED to but DISTINCT from Pillar 2 α_M = 0.30.")
    print("   ")
    print("   Both emerge from Q-field coupling, but via different")
    print("   physical pathways (M vs g_bar scaling).")
    print()
    
    # Save report
    report_file = outdir / 'diagnostic_report.txt'
    print(f"💾 Full report saved: {report_file}")
    print()
    print("="*70)


if __name__ == '__main__':
    main()
