# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
# All rights reserved.
# Unauthorized copying, modification, distribution prohibited without prior written consent.



"""
EXAMPLE USAGE
=============
Demonstrates how to use the 3D+3D analysis toolkit

Simone Calzighetti & Lucy (Claude AI)
3D+3DT Laboratory, November 2025
"""

from sparc_harmonic_analyzer import (
    load_all_sparc,
    load_sparc_galaxy,
    HarmonicModel,
    analyze_all_galaxies
)

# ============================================================================
# EXAMPLE 1: Analyze a single galaxy
# ============================================================================

print("="*80)
print("EXAMPLE 1: Single Galaxy Analysis")
print("="*80)

# Load DDO154 (one of the best examples)
galaxy = load_sparc_galaxy('DDO154_rotmod.dat')

print(f"\nGalaxy: {galaxy.name}")
print(f"Distance: {galaxy.distance} Mpc")
print(f"Data points: {len(galaxy.r)}")
print(f"Radial range: {galaxy.r.min():.2f} - {galaxy.r.max():.2f} kpc")
print(f"Velocity range: {galaxy.v_obs.min():.1f} - {galaxy.v_obs.max():.1f} km/s")

# Fit Triple model
model = HarmonicModel([1.89, 4.30, 11.7])
result = model.fit(galaxy, verbose=True)

print(f"\nResults:")
print(f"  χ²_baseline = {result['χ2_baseline']:.1f}")
print(f"  χ²_fit = {result['χ2_fit']:.1f}")
print(f"  Improvement = {result['improvement_pct']:.1f}%")
print(f"  Q₁ = {result['Q_params'][0]:.3f} (λ=1.89 kpc)")
print(f"  Q₂ = {result['Q_params'][1]:.3f} (λ=4.30 kpc)")
print(f"  Q₃ = {result['Q_params'][2]:.3f} (λ=11.7 kpc)")


# ============================================================================
# EXAMPLE 2: Compare different models on one galaxy
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 2: Model Comparison on DDO154")
print("="*80)

models = {
    'Single': [4.30],
    'Double': [1.89, 4.30],
    'Triple': [1.89, 4.30, 11.7]
}

for name, scales in models.items():
    model = HarmonicModel(scales)
    result = model.fit(galaxy, verbose=False)
    print(f"\n{name:10s}: χ² = {result['χ2_fit']:8.1f}, " + 
          f"improvement = {result['improvement_pct']:5.1f}%")


# ============================================================================
# EXAMPLE 3: Batch analysis of multiple galaxies
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 3: Batch Analysis")
print("="*80)

# Load first 10 galaxies
print("\nLoading galaxies...")
all_galaxies = load_all_sparc('.')
sample_galaxies = all_galaxies[:10]

print(f"Analyzing {len(sample_galaxies)} galaxies...\n")

# Analyze with Triple model
df = analyze_all_galaxies(sample_galaxies, [1.89, 4.30, 11.7], verbose=True)

print("\nTop 5 by improvement:")
top5 = df.nlargest(5, 'improvement_pct')
print(top5[['name', 'improvement_pct', 'Δχ2']])


# ============================================================================
# EXAMPLE 4: Extract harmonic detection statistics
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 4: Harmonic Detection Statistics")
print("="*80)

import numpy as np

# Get Q parameters for all galaxies
Q_values = []
for _, row in df.iterrows():
    Q_values.append(row['Q_params'])

Q_array = np.array(Q_values)

# Detection threshold
threshold = 0.1

detected_1 = (Q_array[:, 0] > threshold).sum()
detected_2 = (Q_array[:, 1] > threshold).sum()
detected_3 = (Q_array[:, 2] > threshold).sum()

print(f"\nDetection rates (Q > {threshold}):")
print(f"  λ₁ = 1.89 kpc: {detected_1}/{len(df)} ({detected_1/len(df)*100:.1f}%)")
print(f"  λ₂ = 4.30 kpc: {detected_2}/{len(df)} ({detected_2/len(df)*100:.1f}%)")
print(f"  λ₃ = 11.7 kpc: {detected_3}/{len(df)} ({detected_3/len(df)*100:.1f}%)")

print(f"\nMean Q values:")
print(f"  Q₁ = {Q_array[:, 0].mean():.3f} ± {Q_array[:, 0].std():.3f}")
print(f"  Q₂ = {Q_array[:, 1].mean():.3f} ± {Q_array[:, 1].std():.3f}")
print(f"  Q₃ = {Q_array[:, 2].mean():.3f} ± {Q_array[:, 2].std():.3f}")


# ============================================================================
# EXAMPLE 5: Check harmonic ratios
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 5: Harmonic Ratio Verification")
print("="*80)

λ1, λ2, λ3 = 1.89, 4.30, 11.7

ratio_21 = λ2 / λ1
ratio_32 = λ3 / λ2

print(f"\nObserved ratios:")
print(f"  λ₂/λ₁ = {ratio_21:.3f}")
print(f"  λ₃/λ₂ = {ratio_32:.3f}")

print(f"\nTheoretical predictions:")
print(f"  9/4 = {9/4:.3f} (error: {abs(ratio_21 - 9/4)/(9/4)*100:.2f}%)")
print(f"  8/3 = {8/3:.3f} (error: {abs(ratio_32 - 8/3)/(8/3)*100:.2f}%)")


# ============================================================================
# EXAMPLE 6: Quick statistics summary
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 6: Quick Summary Statistics")
print("="*80)

print(f"\nDataset: {len(df)} galaxies")
print(f"\nχ² improvements:")
print(f"  Mean:   {df['improvement_pct'].mean():.1f}%")
print(f"  Median: {df['improvement_pct'].median():.1f}%")
print(f"  Std:    {df['improvement_pct'].std():.1f}%")
print(f"  Min:    {df['improvement_pct'].min():.1f}%")
print(f"  Max:    {df['improvement_pct'].max():.1f}%")

significant = (df['improvement_pct'] > 5).sum()
print(f"\nGalaxies with >5% improvement: {significant}/{len(df)} ({significant/len(df)*100:.1f}%)")

highly_significant = (df['improvement_pct'] > 50).sum()
print(f"Galaxies with >50% improvement: {highly_significant}/{len(df)} ({highly_significant/len(df)*100:.1f}%)")


# ============================================================================
# EXAMPLE 7: Save results
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 7: Saving Results")
print("="*80)

# Save to CSV
output_file = 'example_analysis_results.csv'
df.to_csv(output_file, index=False)
print(f"\n✅ Saved results to: {output_file}")

# Save summary statistics
summary = df[['name', 'improvement_pct', 'Δχ2', 'M_star', 'R_max']].describe()
summary_file = 'example_summary_statistics.csv'
summary.to_csv(summary_file)
print(f"✅ Saved summary to: {summary_file}")


print("\n" + "="*80)
print("✅ ALL EXAMPLES COMPLETE!")
print("="*80)
print("\nNext steps:")
print("  1. Modify these examples for your analysis")
print("  2. Run full analysis with sparc_harmonic_analyzer.py")
print("  3. Create plots with plot_individual_galaxies.py")
print("  4. Calculate Bayes Factors with bayesian_model_comparison.py")
