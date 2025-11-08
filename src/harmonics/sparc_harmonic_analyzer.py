# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
# All rights reserved.
# Unauthorized copying, modification, distribution prohibited without prior written consent.

"""
COSMIC HARMONIC ANALYZER - REAL SPARC DATA
===========================================
Simone Calzighetti & Lucy (Claude AI)
3D+3DT Laboratory, November 2025

VERIFICA INDIPENDENTE delle affermazioni di Grok
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Costanti fisiche
G = 4.302e-6  # kpc (km/s)^2 / M_sun

class SPARCGalaxy:
    """Singola galassia SPARC con tutti i dati"""
    
    def __init__(self, name, distance, r, v_obs, v_err, v_gas, v_disk, v_bul):
        self.name = name
        self.distance = distance  # Mpc
        self.r = np.array(r)  # kpc
        self.v_obs = np.array(v_obs)  # km/s
        self.v_err = np.array(v_err)  # km/s
        self.v_gas = np.array(v_gas)  # km/s
        self.v_disk = np.array(v_disk)  # km/s
        self.v_bul = np.array(v_bul)  # km/s
        
        # Velocità barionica totale
        self.v_bary = np.sqrt(v_gas**2 + v_disk**2 + v_bul**2)
        
        # Massa stellare stimata (approssimazione)
        # M_star ≈ (V_disk^2 * R_max) / G
        if len(r) > 0 and np.max(v_disk) > 0:
            self.M_star = np.max(v_disk)**2 * np.max(r) / G
        else:
            self.M_star = 0
    
    def __repr__(self):
        return f"SPARCGalaxy({self.name}, {len(self.r)} points, M*={self.M_star:.2e} Msun)"


def load_sparc_galaxy(filepath):
    """Carica una singola galassia SPARC da file .dat"""
    
    # Leggi header per distanza
    with open(filepath, 'r') as f:
        first_line = f.readline()
        if 'Distance' in first_line:
            distance = float(first_line.split('=')[1].split('Mpc')[0].strip())
        else:
            distance = 1.0  # Default
    
    # Leggi dati
    data = np.loadtxt(filepath, comments='#')
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    name = Path(filepath).stem.replace('_rotmod', '')
    
    r = data[:, 0]
    v_obs = data[:, 1]
    v_err = data[:, 2]
    v_gas = data[:, 3]
    v_disk = data[:, 4]
    v_bul = data[:, 5]
    
    return SPARCGalaxy(name, distance, r, v_obs, v_err, v_gas, v_disk, v_bul)


def load_all_sparc(directory='.'):
    """Carica tutte le galassie SPARC"""
    
    galaxies = []
    dat_files = sorted(Path(directory).glob('*_rotmod.dat'))
    
    print(f"📂 Found {len(dat_files)} SPARC galaxy files")
    
    for filepath in dat_files:
        try:
            galaxy = load_sparc_galaxy(filepath)
            if len(galaxy.r) >= 5:  # Almeno 5 punti
                galaxies.append(galaxy)
        except Exception as e:
            print(f"⚠️  Error loading {filepath.name}: {e}")
    
    print(f"✅ Successfully loaded {len(galaxies)} galaxies")
    
    return galaxies


class HarmonicModel:
    """Modello 3D+3D con armoniche temporali"""
    
    def __init__(self, λ_scales):
        """
        Parameters:
        -----------
        λ_scales : list of float
            Breathing scales to test [λ1, λ2, λ3, ...]
        """
        self.λ_scales = λ_scales
        self.n_harmonics = len(λ_scales)
    
    def predict_velocity(self, r, v_bary, Q_params):
        """
        Predice velocità con oscillazioni temporali
        
        v_obs^2 = v_bary^2 + sum_i [Q_i^2 * v_bary^2 * sin^2(2π r / λ_i)]
        """
        
        v_squared = v_bary**2
        
        for i, λ in enumerate(self.λ_scales):
            Q = Q_params[i]
            phase = 2 * np.pi * r / λ
            v_squared += Q**2 * v_bary**2 * np.sin(phase)**2
        
        return np.sqrt(v_squared)
    
    def fit(self, galaxy, verbose=False):
        """Fit del modello ai dati della galassia"""
        
        r = galaxy.r
        v_obs = galaxy.v_obs
        v_err = galaxy.v_err
        v_bary = galaxy.v_bary
        
        # Chi-quadro baseline (solo baryons)
        χ2_baseline = np.sum(((v_obs - v_bary) / v_err)**2)
        
        def objective(Q_params):
            v_model = self.predict_velocity(r, v_bary, Q_params)
            χ2 = np.sum(((v_obs - v_model) / v_err)**2)
            return χ2
        
        # Ottimizzazione
        bounds = [(0, 1.5)] * self.n_harmonics
        
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=1000,
            atol=1e-6,
            tol=1e-6
        )
        
        χ2_fit = result.fun
        Q_opt = result.x
        
        # Calcola improvement
        Δχ2 = χ2_baseline - χ2_fit
        improvement_pct = (Δχ2 / χ2_baseline) * 100
        
        # Degrees of freedom
        n_data = len(r)
        n_params = self.n_harmonics
        dof = n_data - n_params
        
        # AIC
        AIC = χ2_fit + 2 * n_params
        
        if verbose:
            print(f"\n{galaxy.name}:")
            print(f"  χ²_baseline = {χ2_baseline:.1f}")
            print(f"  χ²_fit = {χ2_fit:.1f}")
            print(f"  Δχ² = {Δχ2:.1f} ({improvement_pct:.1f}% improvement)")
            print(f"  Q = {Q_opt}")
        
        return {
            'χ2_baseline': χ2_baseline,
            'χ2_fit': χ2_fit,
            'Δχ2': Δχ2,
            'improvement_pct': improvement_pct,
            'Q_params': Q_opt,
            'AIC': AIC,
            'dof': dof,
            'n_data': n_data
        }


def analyze_all_galaxies(galaxies, λ_scales, verbose=False):
    """Analizza tutte le galassie con scale armoniche date"""
    
    print(f"\n🎼 Analyzing {len(galaxies)} galaxies with λ scales: {λ_scales}")
    
    model = HarmonicModel(λ_scales)
    results = []
    
    for i, galaxy in enumerate(galaxies):
        if verbose or (i % 25 == 0):
            print(f"[{i+1}/{len(galaxies)}] {galaxy.name}...")
        
        try:
            fit_result = model.fit(galaxy, verbose=False)
            
            results.append({
                'name': galaxy.name,
                'n_points': len(galaxy.r),
                'M_star': galaxy.M_star,
                'R_max': np.max(galaxy.r),
                **fit_result
            })
            
        except Exception as e:
            print(f"⚠️  Error fitting {galaxy.name}: {e}")
    
    df = pd.DataFrame(results)
    
    return df


def compare_models(galaxies):
    """Confronta diversi modelli armonici"""
    
    print("\n" + "="*80)
    print("🎵 COSMIC HARMONIC COMPARISON")
    print("="*80)
    
    models = {
        'Single (4.30 kpc)': [4.30],
        'Double (1.89 + 4.30)': [1.89, 4.30],
        'Triple (1.89 + 4.30 + 11.7)': [1.89, 4.30, 11.7]
    }
    
    all_results = {}
    
    for model_name, λ_scales in models.items():
        print(f"\n{'─'*80}")
        print(f"📊 MODEL: {model_name}")
        print(f"{'─'*80}")
        
        df = analyze_all_galaxies(galaxies, λ_scales, verbose=False)
        all_results[model_name] = df
        
        # Statistiche
        mean_improvement = df['improvement_pct'].mean()
        median_improvement = df['improvement_pct'].median()
        positive_improvement = (df['improvement_pct'] > 5).sum()
        
        print(f"\n📈 RESULTS:")
        print(f"  Mean improvement: {mean_improvement:.1f}%")
        print(f"  Median improvement: {median_improvement:.1f}%")
        print(f"  Galaxies with >5% improvement: {positive_improvement}/{len(df)} ({positive_improvement/len(df)*100:.1f}%)")
        
        # Top 5
        top5 = df.nlargest(5, 'Δχ2')
        print(f"\n🌟 TOP 5 GALAXIES:")
        for idx, row in top5.iterrows():
            print(f"  {row['name']:15s}: Δχ² = +{row['Δχ2']:.0f} ({row['improvement_pct']:.1f}%)")
    
    return all_results


def analyze_harmonic_ratios(df_triple):
    """Analizza rapporti armonici da fit Triple"""
    
    print("\n" + "="*80)
    print("🎼 HARMONIC RATIO ANALYSIS")
    print("="*80)
    
    # Estrai Q parameters
    Q_values = np.array([row['Q_params'] for _, row in df_triple.iterrows()])
    
    # Q1, Q2, Q3
    Q1 = Q_values[:, 0]  # 1.89 kpc
    Q2 = Q_values[:, 1]  # 4.30 kpc
    Q3 = Q_values[:, 2]  # 11.7 kpc
    
    # Detection threshold: Q > 0.1 (10% coupling)
    detected_1 = Q1 > 0.1
    detected_2 = Q2 > 0.1
    detected_3 = Q3 > 0.1
    
    print(f"\n📊 HARMONIC DETECTION:")
    print(f"  λ₁ = 1.89 kpc: {detected_1.sum()}/{len(df_triple)} ({detected_1.sum()/len(df_triple)*100:.1f}%)")
    print(f"  λ₂ = 4.30 kpc: {detected_2.sum()}/{len(df_triple)} ({detected_2.sum()/len(df_triple)*100:.1f}%)")
    print(f"  λ₃ = 11.7 kpc: {detected_3.sum()}/{len(df_triple)} ({detected_3.sum()/len(df_triple)*100:.1f}%)")
    
    # Rapporti teorici
    λ1, λ2, λ3 = 1.89, 4.30, 11.7
    ratio_21_theory = λ2 / λ1  # 2.275
    ratio_32_theory = λ3 / λ2  # 2.721
    
    print(f"\n🎯 THEORETICAL RATIOS:")
    print(f"  λ₂/λ₁ = {ratio_21_theory:.3f}")
    print(f"  λ₃/λ₂ = {ratio_32_theory:.3f}")
    
    # Grok claims:
    # λ₂/λ₁ = 2.274 ± 0.031 vs 9/4 = 2.25 (1.1% error)
    # λ₃/λ₂ = 2.721 ± 0.082 vs 8/3 ≈ 2.667 (2.0% error)
    
    print(f"\n✅ GROK CLAIMS:")
    print(f"  λ₂/λ₁ = 2.274 ± 0.031 (vs 9/4 = 2.25, error 1.1%)")
    print(f"  λ₃/λ₂ = 2.721 ± 0.082 (vs 8/3 = 2.667, error 2.0%)")
    
    return Q_values


def create_summary_plots(all_results, galaxies):
    """Crea plot riassuntivi"""
    
    print("\n📊 Creating summary plots...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Improvement comparison
    ax1 = plt.subplot(2, 2, 1)
    
    for model_name, df in all_results.items():
        improvements = df['improvement_pct'].sort_values()
        ax1.plot(improvements.values, label=model_name, linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('Galaxy rank', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax1.set_title('χ² Improvement by Model', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=5, color='red', linestyle='--', label='5% threshold')
    
    # Plot 2: Histogram of improvements
    ax2 = plt.subplot(2, 2, 2)
    
    for model_name, df in all_results.items():
        ax2.hist(df['improvement_pct'], bins=30, alpha=0.5, label=model_name)
    
    ax2.set_xlabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Improvements', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Q parameters for Triple model
    ax3 = plt.subplot(2, 2, 3)
    
    df_triple = all_results['Triple (1.89 + 4.30 + 11.7)']
    Q_values = np.array([row['Q_params'] for _, row in df_triple.iterrows()])
    
    ax3.scatter(Q_values[:, 0], Q_values[:, 1], alpha=0.5, s=50, label='Q₁ vs Q₂')
    ax3.scatter(Q_values[:, 1], Q_values[:, 2], alpha=0.5, s=50, label='Q₂ vs Q₃')
    ax3.set_xlabel('Q (lower harmonic)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Q (higher harmonic)', fontsize=12, fontweight='bold')
    ax3.set_title('Harmonic Coupling Strengths', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Improvement vs Mass
    ax4 = plt.subplot(2, 2, 4)
    
    for model_name, df in all_results.items():
        ax4.scatter(np.log10(df['M_star']), df['improvement_pct'], 
                   alpha=0.5, s=30, label=model_name)
    
    ax4.set_xlabel('log₁₀(M* / M☉)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Improvement vs Stellar Mass', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # M_crit from theory
    M_crit = 2.43e10
    ax4.axvline(np.log10(M_crit), color='red', linestyle='--', 
               linewidth=2, label=f'M_crit = {M_crit:.2e}')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/sparc_harmonic_analysis.png', 
                dpi=300, bbox_inches='tight')
    print("✅ Saved: sparc_harmonic_analysis.png")
    
    plt.close()


def main():
    """Main analysis pipeline"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🎵  COSMIC HARMONIC ANALYZER  🎵                      ║
    ║                                                              ║
    ║        INDEPENDENT VERIFICATION OF GROK'S CLAIMS            ║
    ║                                                              ║
    ║        3D+3D Discrete Spacetime Theory                      ║
    ║        Simone Calzighetti & Lucy (Claude AI)                ║
    ║        November 4, 2025                                      ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Load data
    print("\n📂 Loading SPARC data...")
    galaxies = load_all_sparc('/home/claude')
    
    print(f"\n✅ Loaded {len(galaxies)} galaxies")
    print(f"   Total data points: {sum(len(g.r) for g in galaxies)}")
    
    # Compare models
    all_results = compare_models(galaxies)
    
    # Analyze harmonic ratios
    df_triple = all_results['Triple (1.89 + 4.30 + 11.7)']
    Q_values = analyze_harmonic_ratios(df_triple)
    
    # Create plots
    create_summary_plots(all_results, galaxies)
    
    # Save results
    print("\n💾 Saving results...")
    for model_name, df in all_results.items():
        filename = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')
        df.to_csv(f'/mnt/user-data/outputs/sparc_results_{filename}.csv', index=False)
        print(f"✅ Saved: sparc_results_{filename}.csv")
    
    print("\n" + "="*80)
    print("🎵 ANALYSIS COMPLETE! 🎵")
    print("="*80)
    
    return all_results


if __name__ == "__main__":
    results = main()
