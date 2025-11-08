"""
BAYESIAN MODEL COMPARISON
==========================
Calculate Bayes Factors for 3D+3D vs ΛCDM

Simone Calzighetti & Lucy (Claude AI)
3D+3DT Laboratory, November 2025
"""

import numpy as np
from scipy import stats
from scipy.special import gammaln
import pandas as pd
import matplotlib.pyplot as plt

from sparc_harmonic_analyzer import load_all_sparc, HarmonicModel


def calculate_BIC(χ2, n_data, n_params):
    """
    Calculate Bayesian Information Criterion
    
    BIC = χ² + k·ln(n)
    
    where k = number of parameters, n = number of data points
    """
    return χ2 + n_params * np.log(n_data)


def calculate_AIC(χ2, n_params):
    """
    Calculate Akaike Information Criterion
    
    AIC = χ² + 2k
    
    where k = number of parameters
    """
    return χ2 + 2 * n_params


def calculate_AICc(χ2, n_data, n_params):
    """
    Calculate corrected AIC for small sample sizes
    
    AICc = AIC + 2k(k+1)/(n-k-1)
    """
    AIC = calculate_AIC(χ2, n_params)
    correction = (2 * n_params * (n_params + 1)) / (n_data - n_params - 1)
    return AIC + correction


def bayes_factor_from_BIC(BIC1, BIC2):
    """
    Approximate Bayes Factor from BIC difference
    
    BF ≈ exp((BIC1 - BIC2)/2)
    
    Returns:
    --------
    BF : float
        Bayes Factor for model 2 vs model 1
    """
    ΔBIC = BIC1 - BIC2
    BF = np.exp(ΔBIC / 2)
    return BF


def interpret_bayes_factor(BF):
    """
    Interpret Bayes Factor according to Kass & Raftery scale
    
    Returns:
    --------
    str : interpretation
    """
    log10_BF = np.log10(BF)
    
    if log10_BF < 0:
        return "Negative (favors model 1)"
    elif log10_BF < 0.5:
        return "Not worth more than a bare mention"
    elif log10_BF < 1:
        return "Substantial evidence"
    elif log10_BF < 2:
        return "Strong evidence"
    else:
        return "Decisive evidence"


def compare_models_bayesian(galaxies):
    """
    Bayesian comparison of all models
    
    Models:
    - M0: Baryons only (ΛCDM without dark matter fit)
    - M1: Single harmonic (4.30 kpc)
    - M2: Double harmonic (1.89 + 4.30 kpc)
    - M3: Triple harmonic (1.89 + 4.30 + 11.7 kpc)
    """
    
    print("\n" + "="*80)
    print("🎲 BAYESIAN MODEL COMPARISON")
    print("="*80)
    
    models = {
        'M0_Baryons': {'λ_scales': [], 'n_params': 0},
        'M1_Single': {'λ_scales': [4.30], 'n_params': 1},
        'M2_Double': {'λ_scales': [1.89, 4.30], 'n_params': 2},
        'M3_Triple': {'λ_scales': [1.89, 4.30, 11.7], 'n_params': 3}
    }
    
    results = {name: [] for name in models.keys()}
    
    # Analyze each galaxy
    print(f"\n📊 Analyzing {len(galaxies)} galaxies...")
    
    for i, galaxy in enumerate(galaxies):
        if (i+1) % 25 == 0:
            print(f"  [{i+1}/{len(galaxies)}] {galaxy.name}")
        
        n_data = len(galaxy.r)
        
        # Baseline (baryons only)
        χ2_baseline = np.sum(((galaxy.v_obs - galaxy.v_bary) / galaxy.v_err)**2)
        
        results['M0_Baryons'].append({
            'galaxy': galaxy.name,
            'χ2': χ2_baseline,
            'n_data': n_data,
            'n_params': 0,
            'BIC': calculate_BIC(χ2_baseline, n_data, 0),
            'AIC': calculate_AIC(χ2_baseline, 0),
            'AICc': calculate_AICc(χ2_baseline, n_data, 0)
        })
        
        # Harmonic models
        for model_name, model_spec in models.items():
            if model_name == 'M0_Baryons':
                continue
            
            try:
                model = HarmonicModel(model_spec['λ_scales'])
                fit = model.fit(galaxy, verbose=False)
                
                results[model_name].append({
                    'galaxy': galaxy.name,
                    'χ2': fit['χ2_fit'],
                    'n_data': n_data,
                    'n_params': model_spec['n_params'],
                    'BIC': calculate_BIC(fit['χ2_fit'], n_data, model_spec['n_params']),
                    'AIC': fit['AIC'],
                    'AICc': calculate_AICc(fit['χ2_fit'], n_data, model_spec['n_params'])
                })
                
            except Exception as e:
                # Failed fit - use baseline
                results[model_name].append({
                    'galaxy': galaxy.name,
                    'χ2': χ2_baseline,
                    'n_data': n_data,
                    'n_params': model_spec['n_params'],
                    'BIC': calculate_BIC(χ2_baseline, n_data, model_spec['n_params']),
                    'AIC': calculate_AIC(χ2_baseline, model_spec['n_params']),
                    'AICc': calculate_AICc(χ2_baseline, n_data, model_spec['n_params'])
                })
    
    # Convert to DataFrames
    dfs = {name: pd.DataFrame(data) for name, data in results.items()}
    
    # Calculate summary statistics
    print("\n" + "="*80)
    print("📈 SUMMARY STATISTICS")
    print("="*80)
    
    for model_name, df in dfs.items():
        print(f"\n{model_name}:")
        print(f"  Mean χ²: {df['χ2'].mean():.1f}")
        print(f"  Mean BIC: {df['BIC'].mean():.1f}")
        print(f"  Mean AIC: {df['AIC'].mean():.1f}")
    
    # Pairwise Bayes Factors
    print("\n" + "="*80)
    print("🎲 BAYES FACTORS (from BIC)")
    print("="*80)
    
    comparisons = [
        ('M1_Single', 'M0_Baryons', 'M1 vs M0'),
        ('M2_Double', 'M0_Baryons', 'M2 vs M0'),
        ('M3_Triple', 'M0_Baryons', 'M3 vs M0'),
        ('M2_Double', 'M1_Single', 'M2 vs M1'),
        ('M3_Triple', 'M2_Double', 'M3 vs M2')
    ]
    
    BF_summary = []
    
    for model1, model2, label in comparisons:
        BIC1 = dfs[model1]['BIC'].sum()
        BIC2 = dfs[model2]['BIC'].sum()
        
        BF = bayes_factor_from_BIC(BIC2, BIC1)
        log10_BF = np.log10(BF)
        
        interpretation = interpret_bayes_factor(BF)
        
        print(f"\n{label}:")
        print(f"  ΔBIC = {BIC2 - BIC1:.1f}")
        print(f"  log₁₀(BF) = {log10_BF:.1f}")
        print(f"  BF = 10^{log10_BF:.1f}")
        print(f"  Interpretation: {interpretation}")
        
        BF_summary.append({
            'comparison': label,
            'ΔBIC': BIC2 - BIC1,
            'log10_BF': log10_BF,
            'interpretation': interpretation
        })
    
    # Overall comparison
    print("\n" + "="*80)
    print("🏆 OVERALL MODEL RANKING (by total BIC)")
    print("="*80)
    
    total_BICs = {name: df['BIC'].sum() for name, df in dfs.items()}
    sorted_models = sorted(total_BICs.items(), key=lambda x: x[1])
    
    best_BIC = sorted_models[0][1]
    
    for rank, (model_name, BIC) in enumerate(sorted_models, 1):
        ΔBIC = BIC - best_BIC
        print(f"\n{rank}. {model_name}")
        print(f"   Total BIC = {BIC:.1f}")
        print(f"   ΔBIC = {ΔBIC:.1f}")
        
        if ΔBIC > 0:
            BF = np.exp(ΔBIC / 2)
            print(f"   BF vs best = 1/{BF:.2e}")
    
    # Save results
    print("\n💾 Saving results...")
    
    BF_df = pd.DataFrame(BF_summary)
    BF_df.to_csv('/mnt/user-data/outputs/bayes_factors.csv', index=False)
    print("✅ Saved: bayes_factors.csv")
    
    for model_name, df in dfs.items():
        df.to_csv(f'/mnt/user-data/outputs/bayesian_{model_name}.csv', index=False)
        print(f"✅ Saved: bayesian_{model_name}.csv")
    
    # Create visualization
    plot_model_comparison(dfs)
    
    return dfs, BF_summary


def plot_model_comparison(dfs):
    """Create visualization of model comparison"""
    
    print("\n📊 Creating comparison plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: BIC comparison
    ax = axes[0, 0]
    
    model_names = list(dfs.keys())
    total_BICs = [dfs[name]['BIC'].sum() for name in model_names]
    
    colors = ['gray', 'lightblue', 'orange', 'green']
    bars = ax.bar(range(len(model_names)), total_BICs, color=colors)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('Total BIC', fontsize=12, fontweight='bold')
    ax.set_title('Bayesian Information Criterion\n(lower is better)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, bic in zip(bars, total_BICs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{bic:.0f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 2: ΔBIC histogram
    ax = axes[0, 1]
    
    ref_model = 'M0_Baryons'
    for model_name, color in zip(model_names[1:], colors[1:]):
        ΔBIC = dfs[ref_model]['BIC'] - dfs[model_name]['BIC']
        ax.hist(ΔBIC, bins=30, alpha=0.6, label=model_name, color=color)
    
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('ΔBIC (vs M0)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('BIC Improvement Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Model preference by galaxy
    ax = axes[1, 0]
    
    preferences = np.zeros((len(dfs['M0_Baryons']), len(model_names)))
    
    for i in range(len(dfs['M0_Baryons'])):
        BICs = [dfs[name].iloc[i]['BIC'] for name in model_names]
        best_idx = np.argmin(BICs)
        preferences[i, best_idx] = 1
    
    preference_counts = preferences.sum(axis=0)
    
    bars = ax.bar(range(len(model_names)), preference_counts, color=colors)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylabel('Number of galaxies', fontsize=12, fontweight='bold')
    ax.set_title('Model Preference\n(lowest BIC per galaxy)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentages
    n_galaxies = len(dfs['M0_Baryons'])
    for bar, count in zip(bars, preference_counts):
        height = bar.get_height()
        pct = count / n_galaxies * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count:.0f}\n({pct:.1f}%)',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 4: Cumulative evidence
    ax = axes[1, 1]
    
    n_galaxies = len(dfs['M0_Baryons'])
    x = np.arange(1, n_galaxies + 1)
    
    for model_name, color in zip(model_names[1:], colors[1:]):
        cumulative_ΔBIC = np.cumsum(
            dfs['M0_Baryons']['BIC'].values - dfs[model_name]['BIC'].values
        )
        ax.plot(x, cumulative_ΔBIC, linewidth=2, label=model_name, color=color)
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Number of galaxies', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative ΔBIC (vs M0)', fontsize=12, fontweight='bold')
    ax.set_title('Evidence Accumulation', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/bayesian_model_comparison.png',
               dpi=300, bbox_inches='tight')
    print("✅ Saved: bayesian_model_comparison.png")
    plt.close()


def main():
    """Main execution"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🎲  BAYESIAN MODEL COMPARISON  🎲                     ║
    ║                                                              ║
    ║        Rigorous statistical comparison of models            ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Load galaxies
    print("\n📂 Loading SPARC data...")
    galaxies = load_all_sparc('.')
    
    # Run comparison
    dfs, BF_summary = compare_models_bayesian(galaxies)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print("\nResults saved:")
    print("  - bayes_factors.csv")
    print("  - bayesian_M*.csv (for each model)")
    print("  - bayesian_model_comparison.png")


if __name__ == "__main__":
    main()
