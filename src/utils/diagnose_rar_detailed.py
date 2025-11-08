# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
# All rights reserved. Unauthorized copying, modification, or distribution is prohibited.

#!/usr/bin/env python3
"""
Detailed RAR Data Diagnostics
Analyzes every aspect of the RAR CSV to find what's wrong
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def diagnose_rar_csv(csv_path):
    """Complete diagnostic of RAR data."""
    
    print("="*70)
    print("DETAILED RAR DATA DIAGNOSTICS")
    print("="*70)
    print()
    
    # Load
    df = pd.read_csv(csv_path)
    print(f"📂 File: {csv_path}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print()
    
    # Show first 10 rows
    print("📋 FIRST 10 ROWS:")
    print(df.head(10).to_string())
    print()
    
    # Statistics for each column
    print("📊 STATISTICS:")
    for col in ['g_bar', 'g_obs', 'V_bar', 'V_obs']:
        if col in df.columns:
            vals = df[col].values
            print(f"\n{col}:")
            print(f"   Min:    {vals.min():.3e}")
            print(f"   Median: {np.median(vals):.3e}")
            print(f"   Max:    {vals.max():.3e}")
            print(f"   Mean:   {vals.mean():.3e}")
    print()
    
    # Check relationships
    g_bar = df['g_bar'].values
    g_obs = df['g_obs'].values
    V_bar = df['V_bar'].values if 'V_bar' in df.columns else None
    V_obs = df['V_obs'].values if 'V_obs' in df.columns else None
    R_kpc = df['R_kpc'].values if 'R_kpc' in df.columns else None
    
    print("🔍 PHYSICAL CONSISTENCY CHECKS:")
    print()
    
    # Check 1: V² / R relationship
    if V_obs is not None and R_kpc is not None:
        KPC = 3.086e19  # meters
        KM = 1000       # m/s
        
        R_m = R_kpc * KPC
        V_m_s = V_obs * KM
        
        g_computed = V_m_s**2 / R_m
        
        print("CHECK 1: Does g_obs = V_obs² / R?")
        ratio_check1 = g_obs / g_computed
        print(f"   Median ratio g_obs / (V_obs²/R): {np.median(ratio_check1):.3f}")
        if 0.95 < np.median(ratio_check1) < 1.05:
            print("   ✅ CONSISTENT")
        else:
            print("   ❌ INCONSISTENT - g_obs not computed from V_obs!")
        print()
    
    # Check 2: Similar for g_bar
    if V_bar is not None and R_kpc is not None:
        KPC = 3.086e19
        KM = 1000
        
        R_m = R_kpc * KPC
        V_m_s = V_bar * KM
        
        g_bar_computed = V_m_s**2 / R_m
        
        print("CHECK 2: Does g_bar = V_bar² / R?")
        ratio_check2 = g_bar / g_bar_computed
        print(f"   Median ratio g_bar / (V_bar²/R): {np.median(ratio_check2):.3f}")
        if 0.95 < np.median(ratio_check2) < 1.05:
            print("   ✅ CONSISTENT")
        else:
            print("   ❌ INCONSISTENT - g_bar not computed from V_bar!")
        print()
    
    # Check 3: RAR shape
    print("CHECK 3: RAR Shape Analysis")
    
    # Split into bins
    log_gbar = np.log10(g_bar)
    log_gobs = np.log10(g_obs)
    
    bins = np.linspace(log_gbar.min(), log_gbar.max(), 20)
    idx = np.digitize(log_gbar, bins)
    
    print("   Binned median ratios (g_obs/g_bar):")
    print("   log10(g_bar)  |  median ratio")
    print("   " + "-"*35)
    
    for i in range(1, len(bins)):
        mask = idx == i
        if mask.sum() > 0:
            bin_center = (bins[i-1] + bins[i]) / 2
            ratio_bin = np.median((g_obs/g_bar)[mask])
            print(f"   {bin_center:8.2f}      |  {ratio_bin:6.3f}")
    print()
    
    # Check 4: Expected RAR behavior
    print("CHECK 4: Expected RAR Behavior")
    
    # Low g_bar: expect g_obs > g_bar (DM boost)
    low_mask = g_bar < 1e-10
    if low_mask.sum() > 0:
        ratio_low = np.median((g_obs/g_bar)[low_mask])
        print(f"   At g_bar < 10⁻¹⁰: ratio = {ratio_low:.3f}")
        if ratio_low > 1.5:
            print("   ✅ CORRECT - shows DM boost")
        else:
            print("   ❌ WRONG - no DM boost!")
    
    # High g_bar: expect g_obs ≈ g_bar (Newtonian)
    high_mask = g_bar > 1e-9
    if high_mask.sum() > 0:
        ratio_high = np.median((g_obs/g_bar)[high_mask])
        print(f"   At g_bar > 10⁻⁹:  ratio = {ratio_high:.3f}")
        if 0.9 < ratio_high < 1.2:
            print("   ✅ CORRECT - Newtonian regime")
        else:
            print("   ❌ WRONG - not Newtonian!")
    print()
    
    # Check 5: V_obs vs V_bar
    if V_obs is not None and V_bar is not None:
        print("CHECK 5: Velocity Comparison")
        ratio_V = V_obs / V_bar
        print(f"   Median V_obs/V_bar: {np.median(ratio_V):.3f}")
        print(f"   Range: [{ratio_V.min():.2f}, {ratio_V.max():.2f}]")
        
        # At large R: expect V_obs > V_bar (flat rotation)
        if R_kpc is not None:
            large_R = R_kpc > 10  # outer regions
            if large_R.sum() > 0:
                ratio_V_outer = np.median(ratio_V[large_R])
                print(f"   At R > 10 kpc: V_obs/V_bar = {ratio_V_outer:.3f}")
                if ratio_V_outer > 1.2:
                    print("   ✅ Shows flat rotation curve (DM)")
                else:
                    print("   ❌ No flat rotation curve!")
        print()
    
    # Summary
    print("="*70)
    print("DIAGNOSIS SUMMARY:")
    print("="*70)
    
    # Check all conditions
    checks_passed = []
    
    if V_obs is not None and R_kpc is not None:
        R_m = R_kpc * 3.086e19
        V_m_s = V_obs * 1000
        g_computed = V_m_s**2 / R_m
        ratio_check = np.median(g_obs / g_computed)
        checks_passed.append(0.95 < ratio_check < 1.05)
    
    low_mask = g_bar < 1e-10
    if low_mask.sum() > 0:
        ratio_low = np.median((g_obs/g_bar)[low_mask])
        checks_passed.append(ratio_low > 1.5)
    
    high_mask = g_bar > 1e-9
    if high_mask.sum() > 0:
        ratio_high = np.median((g_obs/g_bar)[high_mask])
        checks_passed.append(0.9 < ratio_high < 1.2)
    
    passed = sum(checks_passed)
    total = len(checks_passed)
    
    print(f"\nChecks passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ DATA APPEARS VALID")
    elif passed >= total/2:
        print("\n⚠️  DATA HAS ISSUES but partially valid")
    else:
        print("\n❌ DATA IS SEVERELY CORRUPTED")
        print("\nPOSSIBLE CAUSES:")
        print("1. g_bar and g_obs formulas are wrong")
        print("2. Data is from different source/processing")
        print("3. Units are inconsistent")
        print("\nRECOMMENDATION: Regenerate from raw SPARC data")
    
    print()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python diagnose_rar.py <rar_csv_file>")
        sys.exit(1)
    
    diagnose_rar_csv(sys.argv[1])
