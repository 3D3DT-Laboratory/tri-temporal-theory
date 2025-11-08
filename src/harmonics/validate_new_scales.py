# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
# All rights reserved.
# Unauthorized copying, modification, distribution prohibited without prior written consent.

"""
NEW SCALES VALIDATOR
====================
Focused analysis on λ = 0.87 kpc and λ = 21.4 kpc

Are they REAL or artifacts?

Simone Calzighetti & Lucy (Claude AI)
3D+3DT Laboratory, November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, '/home/claude')
from sparc_harmonic_analyzer import load_all_sparc, G

class NewScalesValidator:
    """
    Valida le nuove scale 0.87 e 21.4 kpc
    """
    
    def __init__(self):
        # Scale da testare
        self.scales_to_test = {
            'λ_0': 0.87,   # NEW scale 1
            'λ_1': 1.89,   # Known sub-harmonic
            'λ_2': 4.30,   # Known fundamental
            'λ_3': 8.60,   # Known sub-harmonic 1/2
            'λ_4': 11.7,   # Known super-harmonic
            'λ_5': 21.4    # NEW scale 2
        }
        
        self.results = {}
    
    def fit_single_scale(self, galaxy, λ_breathing):
        """Fit con una singola scala"""
        
        r = galaxy.r
        v_obs = galaxy.v_obs
        v_err = galaxy.v_err
        v_bary = galaxy.v_bary
        
        # Baseline
        χ2_baseline = np.sum(((v_obs - v_bary) / v_err)**2)
        
        def objective(Q):
            phase = 2 * np.pi * r / λ_breathing
            v_temp = Q[0] * v_bary * np.abs(np.sin(phase))
            v_model = np.sqrt(v_bary**2 + v_temp**2)
            χ2 = np.sum(((v_obs - v_model) / v_err)**2)
            return χ2
        
        result = differential_evolution(
            objective,
            bounds=[(0, 1.5)],
            seed=42,
            maxiter=500,
            atol=1e-6
        )
        
        χ2_fit = result.fun
        Q_opt = result.x[0]
        
        Δχ2 = χ2_baseline - χ2_fit
        improvement = (Δχ2 / χ2_baseline) * 100
        
        return {
            'χ2_baseline': χ2_baseline,
            'χ2_fit': χ2_fit,
            'Δχ2': Δχ2,
            'improvement': improvement,
            'Q': Q_opt,
            'detected': Q_opt > 0.1,
            'strong_detection': Q_opt > 0.3
        }
    
    def test_all_scales_on_galaxy(self, galaxy):
        """Testa tutte le 6 scale su una galassia"""
        
        results = {}
        
        for name, λ in self.scales_to_test.items():
            results[name] = self.fit_single_scale(galaxy, λ)
        
        return results
    
    def analyze_all_galaxies(self, galaxies):
        """Analizza tutte le galassie"""
        
        print(f"\n{'='*80}")
        print("🔬 VALIDATING NEW SCALES")
        print(f"{'='*80}")
        print(f"\nTesting {len(galaxies)} galaxies")
        print(f"Scales: {list(self.scales_to_test.values())} kpc\n")
        
        all_results = []
        
        for i, galaxy in enumerate(galaxies, 1):
            if i % 25 == 0 or i == 1:
                print(f"[{i}/{len(galaxies)}] {galaxy.name}")
            
            galaxy_results = self.test_all_scales_on_galaxy(galaxy)
            
            row = {
                'galaxy': galaxy.name,
                'M_star': galaxy.M_star,
                'R_max': np.max(galaxy.r),
                'n_points': len(galaxy.r)
            }
            
            # Add results for each scale
            for scale_name, scale_results in galaxy_results.items():
                row[f'{scale_name}_Q'] = scale_results['Q']
                row[f'{scale_name}_imp'] = scale_results['improvement']
                row[f'{scale_name}_det'] = scale_results['detected']
                row[f'{scale_name}_strong'] = scale_results['strong_detection']
            
            all_results.append(row)
        
        df = pd.DataFrame(all_results)
        self.results_df = df
        
        return df
    
    def create_detection_summary(self):
        """Crea summary delle detection"""
        
        df = self.results_df
        n_galaxies = len(df)
        
        print(f"\n{'='*80}")
        print("📊 DETECTION SUMMARY")
        print(f"{'='*80}")
        print(f"\nTotal galaxies: {n_galaxies}\n")
        
        summary = []
        
        for scale_name, λ_value in self.scales_to_test.items():
            det_col = f'{scale_name}_det'
            strong_col = f'{scale_name}_strong'
            Q_col = f'{scale_name}_Q'
            imp_col = f'{scale_name}_imp'
            
            n_detected = df[det_col].sum()
            n_strong = df[strong_col].sum()
            mean_Q = df[Q_col].mean()
            mean_imp = df[imp_col].mean()
            
            pct_detected = (n_detected / n_galaxies) * 100
            pct_strong = (n_strong / n_galaxies) * 100
            
            is_new = scale_name in ['λ_0', 'λ_5']
            marker = "🆕" if is_new else "✅"
            
            print(f"{marker} {scale_name} = {λ_value:.2f} kpc:")
            print(f"   Detection (Q>0.1): {n_detected}/{n_galaxies} ({pct_detected:.1f}%)")
            print(f"   Strong (Q>0.3):    {n_strong}/{n_galaxies} ({pct_strong:.1f}%)")
            print(f"   Mean Q:            {mean_Q:.3f}")
            print(f"   Mean improvement:  {mean_imp:.1f}%")
            print()
            
            summary.append({
                'scale': scale_name,
                'λ (kpc)': λ_value,
                'detected': n_detected,
                'pct_detected': pct_detected,
                'strong': n_strong,
                'pct_strong': pct_strong,
                'mean_Q': mean_Q,
                'mean_imp': mean_imp,
                'is_new': is_new
            })
        
        return pd.DataFrame(summary)
    
    def compare_new_vs_known(self):
        """Confronta le nuove scale con quelle conosciute"""
        
        df = self.results_df
        
        print(f"\n{'='*80}")
        print("⚖️  NEW vs KNOWN SCALES")
        print(f"{'='*80}")
        
        # Known scales performance
        known_scales = ['λ_1', 'λ_2', 'λ_4']
        new_scales = ['λ_0', 'λ_5']
        
        print("\n📊 KNOWN SCALES (1.89, 4.30, 11.7 kpc):")
        
        for scale in known_scales:
            det = df[f'{scale}_det'].sum()
            pct = (det / len(df)) * 100
            mean_imp = df[f'{scale}_imp'].mean()
            print(f"   {scale}: {det}/{len(df)} ({pct:.1f}%), " + 
                  f"improvement: {mean_imp:.1f}%")
        
        known_mean_detection = np.mean([
            df[f'{scale}_det'].sum() / len(df) * 100 
            for scale in known_scales
        ])
        known_mean_improvement = np.mean([
            df[f'{scale}_imp'].mean() 
            for scale in known_scales
        ])
        
        print(f"\n   AVERAGE KNOWN: {known_mean_detection:.1f}% detection, " + 
              f"{known_mean_improvement:.1f}% improvement")
        
        print("\n🆕 NEW SCALES (0.87, 21.4 kpc):")
        
        for scale in new_scales:
            det = df[f'{scale}_det'].sum()
            pct = (det / len(df)) * 100
            mean_imp = df[f'{scale}_imp'].mean()
            λ = self.scales_to_test[scale]
            print(f"   {scale} ({λ:.2f} kpc): {det}/{len(df)} ({pct:.1f}%), " + 
                  f"improvement: {mean_imp:.1f}%")
        
        new_mean_detection = np.mean([
            df[f'{scale}_det'].sum() / len(df) * 100 
            for scale in new_scales
        ])
        new_mean_improvement = np.mean([
            df[f'{scale}_imp'].mean() 
            for scale in new_scales
        ])
        
        print(f"\n   AVERAGE NEW: {new_mean_detection:.1f}% detection, " + 
              f"{new_mean_improvement:.1f}% improvement")
        
        # Verdict
        print(f"\n{'─'*80}")
        print("🎯 VERDICT:")
        print(f"{'─'*80}")
        
        detection_ratio = new_mean_detection / known_mean_detection
        improvement_ratio = new_mean_improvement / known_mean_improvement
        
        print(f"\nNew scales relative performance:")
        print(f"   Detection rate:  {detection_ratio:.2f}× known scales")
        print(f"   Improvement:     {improvement_ratio:.2f}× known scales")
        
        if detection_ratio > 0.7 and improvement_ratio > 0.7:
            print("\n✅ NEW SCALES ARE REAL!")
            print("   Performance comparable to known scales (>70%)")
        elif detection_ratio > 0.5 and improvement_ratio > 0.5:
            print("\n⚠️  NEW SCALES ARE MARGINAL")
            print("   Detectable but weaker than known scales (50-70%)")
        else:
            print("\n❌ NEW SCALES ARE LIKELY ARTIFACTS")
            print("   Performance significantly below known scales (<50%)")
        
        return {
            'known_detection': known_mean_detection,
            'known_improvement': known_mean_improvement,
            'new_detection': new_mean_detection,
            'new_improvement': new_mean_improvement,
            'detection_ratio': detection_ratio,
            'improvement_ratio': improvement_ratio
        }


def main():
    """Main execution"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🔬  NEW SCALES VALIDATOR  🔬                          ║
    ║                                                              ║
    ║        Are 0.87 kpc and 21.4 kpc REAL?                      ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Load galaxies
    print("\n📂 Loading SPARC galaxies...")
    galaxies = load_all_sparc('/home/claude')
    print(f"✅ Loaded {len(galaxies)} galaxies")
    
    # Create validator
    validator = NewScalesValidator()
    
    # Analyze
    print("\n🚀 Starting validation analysis...")
    df = validator.analyze_all_galaxies(galaxies)
    
    # Save raw results
    df.to_csv('/mnt/user-data/outputs/new_scales_validation_results.csv', index=False)
    print(f"\n💾 Saved: new_scales_validation_results.csv")
    
    # Detection summary
    summary = validator.create_detection_summary()
    summary.to_csv('/mnt/user-data/outputs/new_scales_detection_summary.csv', index=False)
    print(f"💾 Saved: new_scales_detection_summary.csv")
    
    # Compare new vs known
    comparison = validator.compare_new_vs_known()
    
    print("\n" + "="*80)
    print("✅ VALIDATION COMPLETE!")
    print("="*80)
    
    return validator, df, summary, comparison


if __name__ == "__main__":
    validator, df, summary, comparison = main()
