#!/usr/bin/env python3
"""
Filter RAR data to reliable region

Keeps only data where g_bar is in the reliable range.
McGaugh+ 2016 analysis focuses on g_bar > 10^-12 and g_bar < 10^-8.5
"""

import pandas as pd
import numpy as np
import argparse

def filter_rar_data(input_csv, output_csv, 
                   gbar_min=1e-12, 
                   gbar_max=3e-9):
    """
    Filter RAR data to reliable range.
    
    Default range: 10^-12 to 3×10^-9 m/s²
    (avoids extreme inner and outer regions)
    """
    
    print("="*70)
    print("RAR DATA FILTERING")
    print("="*70)
    print()
    
    # Load
    df = pd.read_csv(input_csv)
    print(f"📂 Input: {input_csv}")
    print(f"   Original rows: {len(df)}")
    print()
    
    # Filter
    g_bar = df['g_bar'].values
    
    mask = (g_bar >= gbar_min) & (g_bar <= gbar_max)
    df_filtered = df[mask].copy()
    
    print("🔍 FILTERING:")
    print(f"   g_bar range: [{gbar_min:.2e}, {gbar_max:.2e}] m/s²")
    print(f"   Points kept: {len(df_filtered)} / {len(df)} ({len(df_filtered)/len(df)*100:.1f}%)")
    print()
    
    # Check filtered data quality
    g_bar_f = df_filtered['g_bar'].values
    g_obs_f = df_filtered['g_obs'].values
    
    ratio_f = g_obs_f / g_bar_f
    
    print("📊 FILTERED DATA QUALITY:")
    print(f"   g_bar range: [{g_bar_f.min():.2e}, {g_bar_f.max():.2e}]")
    print(f"   g_obs range: [{g_obs_f.min():.2e}, {g_obs_f.max():.2e}]")
    print(f"   Median ratio: {np.median(ratio_f):.3f}")
    print()
    
    # Check by regime
    low_mask = g_bar_f < 1e-10
    mid_mask = (g_bar_f >= 1e-10) & (g_bar_f < 1e-9)
    high_mask = g_bar_f >= 1e-9
    
    if low_mask.sum() > 0:
        ratio_low = np.median(ratio_f[low_mask])
        print(f"   At g_bar < 10⁻¹⁰: ratio = {ratio_low:.3f} (expect >1.5) {'✅' if ratio_low > 1.5 else '⚠️'}")
    
    if mid_mask.sum() > 0:
        ratio_mid = np.median(ratio_f[mid_mask])
        print(f"   At 10⁻¹⁰ < g_bar < 10⁻⁹: ratio = {ratio_mid:.3f} (expect ~1.0-1.3)")
    
    if high_mask.sum() > 0:
        ratio_high = np.median(ratio_f[high_mask])
        print(f"   At g_bar > 10⁻⁹: ratio = {ratio_high:.3f} (expect ~1.0) {'✅' if 0.9 < ratio_high < 1.1 else '⚠️'}")
    
    print()
    
    # Save
    df_filtered.to_csv(output_csv, index=False)
    print(f"💾 Saved: {output_csv}")
    print()
    
    # Summary
    print("="*70)
    print("FILTERING COMPLETE")
    print("="*70)
    print()
    print("✅ Use this filtered CSV for RAR fitting")
    print("   (removes problematic inner/outer regions)")
    print()
    
    return df_filtered


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter RAR data to reliable region')
    parser.add_argument('--input', required=True, help='Input RAR CSV')
    parser.add_argument('--output', required=True, help='Output filtered CSV')
    parser.add_argument('--gbar-min', type=float, default=1e-12, 
                       help='Minimum g_bar (default: 1e-12)')
    parser.add_argument('--gbar-max', type=float, default=3e-9,
                       help='Maximum g_bar (default: 3e-9)')
    args = parser.parse_args()
    
    filter_rar_data(args.input, args.output, 
                   gbar_min=args.gbar_min,
                   gbar_max=args.gbar_max)
