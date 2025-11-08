# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
# All rights reserved.
# Unauthorized copying, modification, distribution prohibited without prior written consent.



#!/usr/bin/env python3
"""
🌌 HARMONIC POWER ANALYZER - TEORIA 3D+3D 🌌
============================================
Analisi CORRETTA dell'ampiezza armonica σ_n nei RESIDUI DI VELOCITÀ

METRICA CORRETTA:
1. ΔV(r) = V_obs(r) - V_bar(r)  (residui velocità)
2. FFT(ΔV) → power spectrum
3. Band-pass filter intorno a λ_n
4. σ_n = √(∫ |ΔV_fft|² dk) per ogni scala
5. Regressione: log(σ_n) ~ α·log(M_bar)

PREDIZIONE: α ≈ +0.30 (non il SNR break in RAR!)

Autori: Simone Calzighetti & Lucy
Data: Novembre 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from scipy import stats, signal, interpolate
from scipy.fft import fft, fftfreq, fftshift
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from tqdm import tqdm
import warnings
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import argparse

warnings.filterwarnings('ignore')

# Costanti fisiche
G_KPC = 4.30091e-6  # kpc km²/s² / M_sun

# Scale teoriche (kpc)
THEORETICAL_SCALES = {
    'lambda_2/5': 0.87,
    'lambda_2/2': 1.89,  # CORRETTO da 2.15!
    'lambda_2': 4.30,
    '3lambda_2/2': 6.51,  # CORRETTO da 6.45!
    '~2.7lambda_2': 11.7,
    '5lambda_2': 21.4
}

# Alias
LAMBDA_2 = THEORETICAL_SCALES['lambda_2']

print(f"\n{'='*70}")
print(f"🎯 SCALE TEORICHE ARMONICHE")
print(f"{'='*70}")
for name, scale in THEORETICAL_SCALES.items():
    ratio = scale / LAMBDA_2
    print(f"{name:<15} λ = {scale:5.2f} kpc  (λ/λ₂ = {ratio:.3f})")
print(f"{'='*70}\n")

@dataclass
class GalaxyData:
    """Struttura dati per singola galassia"""
    name: str
    r_kpc: np.ndarray
    v_obs: np.ndarray
    v_err: np.ndarray
    v_gas: np.ndarray
    v_disk: np.ndarray
    v_bulge: np.ndarray
    distance_mpc: float
    
    @property
    def mass_baryonic(self) -> float:
        """Massa barionica totale (M_sun)"""
        r_max = self.r_kpc[-1]
        v_bar = self.v_baryonic
        return r_max * v_bar[-1]**2 / G_KPC
    
    @property
    def v_baryonic(self) -> np.ndarray:
        """Velocità barionica (combinazione gas+disk+bulge)"""
        return np.sqrt(self.v_gas**2 + self.v_disk**2 + self.v_bulge**2)
    
    @property
    def v_residuals(self) -> np.ndarray:
        """Residui: ΔV = V_obs - V_bar"""
        return self.v_obs - self.v_baryonic
    
    @property
    def v_flat(self) -> float:
        """Velocità piatta (mediana regione esterna)"""
        mask = self.r_kpc > 2.0
        if mask.sum() > 3:
            return np.median(self.v_obs[mask])
        else:
            return np.max(self.v_obs)

class DataLoader:
    """Caricamento dataset SPARC"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.galaxies: List[GalaxyData] = []
        
    def load_all(self) -> List[GalaxyData]:
        """Carica tutte le galassie"""
        dat_files = sorted(self.data_dir.glob("*_rotmod.dat"))
        
        print(f"\n{'='*70}")
        print(f"📂 CARICAMENTO DATASET SPARC")
        print(f"{'='*70}")
        print(f"Directory: {self.data_dir}")
        print(f"File trovati: {len(dat_files)}")
        
        for fpath in tqdm(dat_files, desc="Caricamento galassie"):
            try:
                gal = self._load_single(fpath)
                self.galaxies.append(gal)
            except Exception as e:
                print(f"⚠️  Errore caricando {fpath.name}: {e}")
                
        print(f"✅ Caricate: {len(self.galaxies)} galassie")
        return self.galaxies
    
    def _load_single(self, fpath: Path) -> GalaxyData:
        """Carica singola galassia"""
        name = fpath.stem.replace('_rotmod', '')
        
        # Leggi header per distanza
        distance = 10.0  # default
        with open(fpath) as f:
            for line in f:
                if 'Distance' in line:
                    distance = float(line.split('=')[1].split()[0])
                    break
        
        # Carica dati
        data = np.loadtxt(fpath, comments='#')
        
        return GalaxyData(
            name=name,
            r_kpc=data[:, 0],
            v_obs=data[:, 1],
            v_err=data[:, 2],
            v_gas=data[:, 3],
            v_disk=data[:, 4],
            v_bulge=data[:, 5],
            distance_mpc=distance
        )

class HarmonicPowerAnalyzer:
    """Analizzatore energia armonica nei residui di velocità"""
    
    @staticmethod
    def compute_power_spectrum(r: np.ndarray, v_residuals: np.ndarray,
                               v_err: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcola power spectrum dei residui di velocità
        
        Returns:
            wavelengths (kpc), power (km²/s²)
        """
        
        if len(r) < 5:
            return np.array([]), np.array([])
        
        # Interpola su griglia uniforme
        r_min, r_max = r.min(), r.max()
        n_points = min(len(r) * 2, 256)  # oversample ma non troppo
        r_uniform = np.linspace(r_min, r_max, n_points)
        
        v_res_interp = np.interp(r_uniform, r, v_residuals)
        
        # Hann window per ridurre edge effects
        window = np.hanning(len(v_res_interp))
        v_windowed = v_res_interp * window
        
        # FFT
        dr = r_uniform[1] - r_uniform[0]
        freqs = fftfreq(len(v_windowed), d=dr)
        v_fft = fft(v_windowed)
        power = np.abs(v_fft)**2
        
        # Solo frequenze positive
        mask = freqs > 0
        wavelengths = 1.0 / freqs[mask]  # kpc
        power = power[mask]  # (km/s)²
        
        # Normalizza per lunghezza segnale
        power = power / len(v_windowed)
        
        return wavelengths, power
    
    @staticmethod
    def extract_harmonic_amplitude(wavelengths: np.ndarray, power: np.ndarray,
                                   lambda_target: float, 
                                   bandwidth: float = 0.3) -> float:
        """
        Estrae ampiezza armonica σ_n per scala λ_n
        
        σ_n = √(∫ Power(λ) dλ) nella banda [λ - Δ, λ + Δ]
        
        Args:
            wavelengths: array lunghezze d'onda (kpc)
            power: array potenza (km²/s²)
            lambda_target: scala target (kpc)
            bandwidth: frazione di λ per banda (default 30%)
        
        Returns:
            σ_n (km/s)
        """
        
        if len(wavelengths) == 0:
            return 0.0
        
        # Banda passante
        lambda_min = lambda_target * (1 - bandwidth)
        lambda_max = lambda_target * (1 + bandwidth)
        
        # Maschera banda
        mask = (wavelengths >= lambda_min) & (wavelengths <= lambda_max)
        
        if not mask.any():
            return 0.0
        
        # Integra potenza nella banda
        power_band = power[mask]
        wavelengths_band = wavelengths[mask]
        
        # Integrazione trapezoidale
        # Sort per sicurezza
        sort_idx = np.argsort(wavelengths_band)
        wl_sorted = wavelengths_band[sort_idx]
        pw_sorted = power_band[sort_idx]
        
        # Energia = integrale potenza
        energy = np.trapz(pw_sorted, wl_sorted)
        
        # Ampiezza = √energia
        sigma = np.sqrt(max(energy, 0))
        
        return sigma
    
    @staticmethod
    def analyze_single_galaxy(gal: GalaxyData, 
                             scales: Dict[str, float]) -> Dict:
        """
        Analisi completa singola galassia
        
        Returns:
            Dict con σ_n per ogni scala + metadati
        """
        
        # Power spectrum dei residui
        wavelengths, power = HarmonicPowerAnalyzer.compute_power_spectrum(
            gal.r_kpc, gal.v_residuals, gal.v_err
        )
        
        if len(wavelengths) == 0:
            return {
                'name': gal.name,
                'mass': gal.mass_baryonic,
                'v_flat': gal.v_flat,
                'n_points': len(gal.r_kpc),
                'valid': False,
                'sigmas': {name: 0.0 for name in scales.keys()}
            }
        
        # Estrai σ per ogni scala
        sigmas = {}
        for name, lambda_n in scales.items():
            sigma_n = HarmonicPowerAnalyzer.extract_harmonic_amplitude(
                wavelengths, power, lambda_n, bandwidth=0.3
            )
            sigmas[name] = sigma_n
        
        # Ampiezza totale (RMS residui)
        rms_total = np.sqrt(np.mean(gal.v_residuals**2))
        
        return {
            'name': gal.name,
            'mass': gal.mass_baryonic,
            'v_flat': gal.v_flat,
            'n_points': len(gal.r_kpc),
            'r_max': gal.r_kpc[-1],
            'valid': True,
            'sigmas': sigmas,
            'rms_total': rms_total,
            'wavelengths': wavelengths,
            'power': power
        }
    
    @staticmethod
    def analyze_ensemble(galaxies: List[GalaxyData],
                        scales: Dict[str, float]) -> Dict:
        """Analisi ensemble"""
        
        print(f"\n{'='*70}")
        print(f"🔬 ANALISI HARMONIC POWER - ENSEMBLE")
        print(f"{'='*70}")
        
        results = []
        
        for gal in tqdm(galaxies, desc="Analisi galassie"):
            res = HarmonicPowerAnalyzer.analyze_single_galaxy(gal, scales)
            results.append(res)
        
        # Statistiche per scala
        summary = {'scales': {}}
        
        for scale_name in scales.keys():
            sigmas = [r['sigmas'][scale_name] for r in results if r['valid']]
            masses = [r['mass'] for r in results if r['valid'] and r['sigmas'][scale_name] > 0]
            
            detection_count = sum(1 for s in sigmas if s > 1.0)  # soglia 1 km/s
            detection_rate = detection_count / len(sigmas) if sigmas else 0
            
            summary['scales'][scale_name] = {
                'lambda': scales[scale_name],
                'mean_sigma': np.mean(sigmas) if sigmas else 0,
                'median_sigma': np.median(sigmas) if sigmas else 0,
                'std_sigma': np.std(sigmas) if sigmas else 0,
                'detection_rate': detection_rate,
                'n_detected': detection_count,
                'n_total': len(sigmas)
            }
        
        summary['individual_results'] = results
        summary['n_galaxies'] = len(galaxies)
        summary['n_valid'] = sum(1 for r in results if r['valid'])
        
        # Print summary
        print(f"\n📊 RISULTATI HARMONIC POWER:")
        print(f"{'='*70}")
        print(f"Galassie analizzate: {summary['n_galaxies']}")
        print(f"Analisi valide: {summary['n_valid']}")
        
        print(f"\n{'Scale':<15} {'λ (kpc)':<10} {'<σ> (km/s)':<12} {'Detection':<12} {'Status'}")
        print(f"{'-'*70}")
        
        for name, stats in summary['scales'].items():
            status = "✅" if stats['detection_rate'] > 0.5 else "⚠️" if stats['detection_rate'] > 0.3 else "❌"
            print(f"{name:<15} {stats['lambda']:<10.2f} {stats['mean_sigma']:<12.2f} "
                  f"{stats['detection_rate']:<12.1%} {status}")
        
        return summary

class MassScalingAnalyzer:
    """Analisi correlazione σ vs M (questa è la metrica CORRETTA!)"""
    
    @staticmethod
    def fit_mass_scaling(results: List[Dict], scale_name: str) -> Dict:
        """
        Fit: log(σ_n) = α·log(M) + const
        
        QUESTA è la regressione corretta per testare σ ∝ M^α
        """
        
        # Estrai dati
        masses = []
        sigmas = []
        
        for res in results:
            if res['valid'] and res['sigmas'][scale_name] > 0:
                masses.append(res['mass'])
                sigmas.append(res['sigmas'][scale_name])
        
        masses = np.array(masses)
        sigmas = np.array(sigmas)
        
        if len(masses) < 10:
            return {
                'scale': scale_name,
                'n_points': len(masses),
                'valid': False
            }
        
        # Log-log fit
        log_m = np.log10(masses)
        log_s = np.log10(sigmas)
        
        # Rimuovi outliers (3σ in log space)
        mean_log_s = np.mean(log_s)
        std_log_s = np.std(log_s)
        mask = np.abs(log_s - mean_log_s) < 3 * std_log_s
        
        log_m_clean = log_m[mask]
        log_s_clean = log_s[mask]
        
        if len(log_m_clean) < 10:
            return {
                'scale': scale_name,
                'n_points': len(masses),
                'valid': False
            }
        
        # Linear fit in log-log
        slope, intercept = np.polyfit(log_m_clean, log_s_clean, 1)
        
        # Statistiche
        r_pearson, p_pearson = stats.pearsonr(log_m_clean, log_s_clean)
        r_spearman, p_spearman = stats.spearmanr(masses[mask], sigmas[mask])
        
        # Residuals
        log_s_pred = slope * log_m_clean + intercept
        residuals = log_s_clean - log_s_pred
        rms_residuals = np.sqrt(np.mean(residuals**2))
        
        return {
            'scale': scale_name,
            'lambda': THEORETICAL_SCALES[scale_name],
            'n_points': len(log_m_clean),
            'alpha': float(slope),
            'alpha_err': float(rms_residuals / np.sqrt(len(log_m_clean))),
            'intercept': float(intercept),
            'r_pearson': float(r_pearson),
            'p_pearson': float(p_pearson),
            'r_spearman': float(r_spearman),
            'p_spearman': float(p_spearman),
            'rms_residuals': float(rms_residuals),
            'valid': True,
            'masses': masses[mask],
            'sigmas': sigmas[mask],
            'log_m': log_m_clean,
            'log_s': log_s_clean
        }
    
    @staticmethod
    def analyze_all_scales(results: List[Dict]) -> Dict:
        """Analizza tutte le scale"""
        
        print(f"\n{'='*70}")
        print(f"💪 CORRELAZIONE MASSA - AMPIEZZA ARMONICA")
        print(f"{'='*70}")
        print(f"Formula: σ_n = σ₀ (M/M₀)^α")
        print(f"Predizione teoria 3D+3D: α ≈ +0.30")
        
        scaling_results = {}
        
        for scale_name in THEORETICAL_SCALES.keys():
            fit_res = MassScalingAnalyzer.fit_mass_scaling(results, scale_name)
            if fit_res['valid']:
                scaling_results[scale_name] = fit_res
        
        # Print results
        print(f"\n📊 RISULTATI PER SCALA:")
        print(f"{'Scale':<15} {'λ (kpc)':<10} {'α':<12} {'r':<10} {'p-value':<12} {'Status'}")
        print(f"{'-'*75}")
        
        for name, fit in scaling_results.items():
            status = "✅" if 0.15 < fit['alpha'] < 0.45 and fit['p_pearson'] < 0.01 else "⚠️"
            print(f"{name:<15} {fit['lambda']:<10.2f} {fit['alpha']:<12.3f} "
                  f"{fit['r_pearson']:<10.3f} {fit['p_pearson']:<12.2e} {status}")
        
        # Media pesata degli α
        if scaling_results:
            alphas = [f['alpha'] for f in scaling_results.values()]
            weights = [1/f['alpha_err']**2 for f in scaling_results.values()]
            
            alpha_mean = np.average(alphas, weights=weights)
            alpha_std = np.sqrt(1 / np.sum(weights))
            
            print(f"\n📈 MEDIA PESATA:")
            print(f"  α = {alpha_mean:.3f} ± {alpha_std:.3f}")
            print(f"  Teorico: α = 0.333")
            
            z_score = abs(alpha_mean - 0.333) / alpha_std
            print(f"  Z-score: {z_score:.2f}σ")
            
            if abs(alpha_mean - 0.333) < 0.10:
                print(f"  ✅ MATCH ECCELLENTE con teoria!")
            elif abs(alpha_mean - 0.333) < 0.15:
                print(f"  ✅ CONSISTENTE con teoria")
            else:
                print(f"  ⚠️  Deviazione dalla teoria")
        
        return scaling_results

class Visualizer:
    """Generazione grafici"""
    
    @staticmethod
    def create_all_plots(summary: Dict, scaling: Dict,
                        galaxies: List[GalaxyData], output_dir: Path):
        """Crea tutti i grafici"""
        
        print(f"\n{'='*70}")
        print(f"📊 GENERAZIONE GRAFICI")
        print(f"{'='*70}")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Detection rates
        Visualizer._plot_detection_rates(summary, output_dir)
        
        # Plot 2: Mass-amplitude per scala
        Visualizer._plot_mass_amplitude_all_scales(scaling, output_dir)
        
        # Plot 3: Combined mass-amplitude
        Visualizer._plot_combined_scaling(scaling, output_dir)
        
        # Plot 4: Power spectra examples
        Visualizer._plot_power_spectra_examples(summary, galaxies, output_dir)
        
        # Plot 5: Sigma distribution per scala
        Visualizer._plot_sigma_distributions(summary, output_dir)
        
        print(f"\n✅ Grafici salvati in: {output_dir}")
    
    @staticmethod
    def _plot_detection_rates(summary: Dict, output_dir: Path):
        """Detection rates per scala"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        scales = list(summary['scales'].keys())
        stats = [summary['scales'][s] for s in scales]
        
        # Detection rates
        rates = [s['detection_rate'] for s in stats]
        colors = ['green' if r > 0.5 else 'orange' if r > 0.3 else 'red' for r in rates]
        
        bars1 = ax1.bar(range(len(scales)), rates, color=colors, alpha=0.7,
                       edgecolor='black', linewidth=1.5)
        ax1.axhline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax1.set_xticks(range(len(scales)))
        ax1.set_xticklabels([s.replace('lambda_2', 'λ₂') for s in scales],
                           rotation=45, ha='right')
        ax1.set_ylabel('Detection Rate', fontsize=13, fontweight='bold')
        ax1.set_title('Detection Rates (σ > 1 km/s)', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar, rate in zip(bars1, rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                    f'{rate:.1%}', ha='center', fontweight='bold', fontsize=10)
        
        # Mean sigma
        sigmas = [s['mean_sigma'] for s in stats]
        bars2 = ax2.bar(range(len(scales)), sigmas, color='skyblue', alpha=0.7,
                       edgecolor='black', linewidth=1.5)
        ax2.set_xticks(range(len(scales)))
        ax2.set_xticklabels([s.replace('lambda_2', 'λ₂') for s in scales],
                           rotation=45, ha='right')
        ax2.set_ylabel('Mean σ [km/s]', fontsize=13, fontweight='bold')
        ax2.set_title('Mean Harmonic Amplitude', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, sig in zip(bars2, sigmas):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + 0.2,
                    f'{sig:.1f}', ha='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'harmonic_detection_rates.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def _plot_mass_amplitude_all_scales(scaling: Dict, output_dir: Path):
        """Mass-amplitude per tutte le scale (pannelli separati)"""
        
        n_scales = len(scaling)
        if n_scales == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (scale_name, fit) in enumerate(scaling.items()):
            if idx >= 6:
                break
            
            ax = axes[idx]
            
            masses = fit['masses']
            sigmas = fit['sigmas']
            
            # Scatter
            ax.scatter(masses, sigmas, alpha=0.6, s=60,
                      edgecolors='black', linewidth=0.5, c='blue')
            
            # Fit line
            m_fit = np.logspace(np.log10(masses.min()), np.log10(masses.max()), 100)
            s_fit = 10**(fit['alpha'] * np.log10(m_fit) + fit['intercept'])
            
            ax.plot(m_fit, s_fit, 'r-', linewidth=2.5,
                   label=f"α = {fit['alpha']:.3f}")
            
            # Theoretical
            s_theory = 10**(0.333 * np.log10(m_fit) + fit['intercept'])
            ax.plot(m_fit, s_theory, 'g--', linewidth=2,
                   label="α = 0.333 (theory)")
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('M_bar [M☉]', fontsize=11, fontweight='bold')
            ax.set_ylabel('σ [km/s]', fontsize=11, fontweight='bold')
            ax.set_title(f"{scale_name} (λ={fit['lambda']:.2f} kpc)\n"
                        f"r={fit['r_pearson']:.3f}, p={fit['p_pearson']:.2e}",
                        fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Mass-Amplitude Scaling - All Scales\nσ ∝ M^α',
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_dir / 'mass_amplitude_all_scales.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def _plot_combined_scaling(scaling: Dict, output_dir: Path):
        """Plot combinato tutte le scale"""
        
        if not scaling:
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        colors_map = {
            'lambda_2/5': 'purple',
            'lambda_2/2': 'blue',
            'lambda_2': 'red',
            '3lambda_2/2': 'green',
            '~2.7lambda_2': 'orange',
            '5lambda_2': 'brown'
        }
        
        for scale_name, fit in scaling.items():
            if not fit['valid']:
                continue
            
            color = colors_map.get(scale_name, 'gray')
            
            ax.scatter(fit['masses'], fit['sigmas'], 
                      alpha=0.5, s=40, c=color,
                      label=f"{scale_name} (α={fit['alpha']:.2f})",
                      edgecolors='black', linewidth=0.3)
        
        # Linea teorica media
        all_masses = np.concatenate([f['masses'] for f in scaling.values() if f['valid']])
        m_range = np.logspace(np.log10(all_masses.min()), np.log10(all_masses.max()), 100)
        
        # α medio
        alphas = [f['alpha'] for f in scaling.values() if f['valid']]
        alpha_mean = np.mean(alphas)
        
        # Reference line con α teorico
        s_ref = 2.0 * (m_range / 1e10)**0.333
        ax.plot(m_range, s_ref, 'k--', linewidth=3, alpha=0.7,
               label=f'Theory: α=0.333')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Baryonic Mass [M☉]', fontsize=14, fontweight='bold')
        ax.set_ylabel('Harmonic Amplitude σ [km/s]', fontsize=14, fontweight='bold')
        ax.set_title('Combined Mass-Amplitude Scaling\nAll Harmonic Scales',
                    fontsize=15, fontweight='bold')
        ax.legend(ncol=2, fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Stats box
        stats_text = f"Mean α = {alpha_mean:.3f}\n"
        stats_text += f"Theory: α = 0.333\n"
        stats_text += f"N scales = {len(scaling)}"
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'combined_mass_amplitude.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def _plot_power_spectra_examples(summary: Dict, galaxies: List[GalaxyData],
                                     output_dir: Path, n_examples: int = 6):
        """Esempi power spectra"""
        
        results = summary['individual_results']
        valid = [r for r in results if r['valid'] and len(r.get('wavelengths', [])) > 0]
        
        if not valid:
            return
        
        examples = np.random.choice(valid, min(n_examples, len(valid)), replace=False)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, res in enumerate(examples):
            if idx >= 6:
                break
            
            ax = axes[idx]
            
            wl = res['wavelengths']
            pw = res['power']
            
            # Plot power spectrum
            ax.plot(wl, pw, 'b-', alpha=0.7, linewidth=1.5)
            
            # Vertical lines at theoretical scales
            for scale_name, lambda_n in THEORETICAL_SCALES.items():
                if wl.min() < lambda_n < wl.max():
                    sigma = res['sigmas'][scale_name]
                    color = 'red' if sigma > 2.0 else 'orange' if sigma > 1.0 else 'gray'
                    ax.axvline(lambda_n, color=color, linestyle='--', alpha=0.6, linewidth=1.5)
            
            ax.set_xlabel('Wavelength λ [kpc]', fontsize=11)
            ax.set_ylabel('Power [(km/s)²]', fontsize=11)
            ax.set_title(f"{res['name']}\nM={res['mass']:.2e} M☉",
                        fontsize=11, fontweight='bold')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.5, 30)
        
        plt.suptitle('Power Spectra Examples - Velocity Residuals\nTheoretical Scales Marked',
                    fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_dir / 'power_spectra_examples.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def _plot_sigma_distributions(summary: Dict, output_dir: Path):
        """Distribuzioni σ per scala"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (scale_name, stats) in enumerate(summary['scales'].items()):
            if idx >= 6:
                break
            
            ax = axes[idx]
            
            # Raccogli σ per questa scala
            results = summary['individual_results']
            sigmas = [r['sigmas'][scale_name] for r in results 
                     if r['valid'] and r['sigmas'][scale_name] > 0]
            
            if not sigmas:
                continue
            
            # Histogram
            ax.hist(sigmas, bins=30, alpha=0.7, color='skyblue',
                   edgecolor='black', linewidth=1)
            
            # Statistics lines
            mean_sig = np.mean(sigmas)
            median_sig = np.median(sigmas)
            
            ax.axvline(mean_sig, color='red', linestyle='-', linewidth=2,
                      label=f'Mean: {mean_sig:.2f} km/s')
            ax.axvline(median_sig, color='blue', linestyle='--', linewidth=2,
                      label=f'Median: {median_sig:.2f} km/s')
            
            ax.set_xlabel('σ [km/s]', fontsize=11)
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title(f"{scale_name} (λ={stats['lambda']:.2f} kpc)\n"
                        f"Detection: {stats['detection_rate']:.1%}",
                        fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Harmonic Amplitude Distributions',
                    fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_dir / 'sigma_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main execution"""
    
    parser = argparse.ArgumentParser(
        description='🌌 Harmonic Power Analyzer - Teoria 3D+3D',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data_dir', type=str, default='/home/claude',
                       help='Directory contenente file SPARC .dat')
    parser.add_argument('--output_dir', type=str,
                       default='/mnt/user-data/outputs/harmonic_power_analysis',
                       help='Directory output risultati')
    parser.add_argument('--bandwidth', type=float, default=0.3,
                       help='Banda passante per estrazione (frazione di λ)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🌌 HARMONIC POWER ANALYZER - TEORIA 3D+3D 🌌")
    print("="*70)
    print(f"Simone Calzighetti & Lucy")
    print(f"Novembre 2025")
    print("="*70)
    print(f"\nMETRICA CORRETTA:")
    print(f"  ΔV(r) = V_obs - V_bar")
    print(f"  σ_n = √(∫ |FFT(ΔV)|² dλ) in banda λ_n")
    print(f"  Predizione: σ ∝ M^(+0.30)")
    print("="*70)
    
    # Load data
    loader = DataLoader(args.data_dir)
    galaxies = loader.load_all()
    
    if len(galaxies) == 0:
        print("\n❌ ERRORE: Nessuna galassia caricata!")
        return
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analisi harmonic power
    summary = HarmonicPowerAnalyzer.analyze_ensemble(galaxies, THEORETICAL_SCALES)
    
    # Mass-scaling analysis
    scaling = MassScalingAnalyzer.analyze_all_scales(summary['individual_results'])
    
    # Salva risultati
    results_file = output_dir / 'harmonic_power_results.json'
    
    # Prepara per JSON
    summary_json = {
        k: v for k, v in summary.items()
        if k not in ['individual_results']
    }
    
    scaling_json = {
        k: {kk: vv for kk, vv in v.items() 
            if not isinstance(vv, np.ndarray)}
        for k, v in scaling.items()
    }
    
    with open(results_file, 'w') as f:
        json.dump({
            'summary': summary_json,
            'scaling': scaling_json
        }, f, indent=2)
    
    print(f"\n💾 Risultati salvati: {results_file}")
    
    # Genera grafici
    Visualizer.create_all_plots(summary, scaling, galaxies, output_dir)
    
    # Final summary
    print("\n" + "="*70)
    print("🎯 SUMMARY FINALE")
    print("="*70)
    
    if scaling:
        alphas = [f['alpha'] for f in scaling.values() if f['valid']]
        if alphas:
            alpha_mean = np.mean(alphas)
            alpha_std = np.std(alphas)
            
            print(f"\n📈 MASS-AMPLITUDE SCALING:")
            print(f"  α medio:   {alpha_mean:.3f} ± {alpha_std:.3f}")
            print(f"  α teorico: 0.333")
            print(f"  Deviazione: {abs(alpha_mean - 0.333):.3f}")
            
            if abs(alpha_mean - 0.333) < 0.10:
                print(f"  ✅ MATCH ECCELLENTE!")
            elif abs(alpha_mean - 0.333) < 0.15:
                print(f"  ✅ CONSISTENTE!")
            else:
                print(f"  ⚠️  Deviazione significativa")
    
    print(f"\n📊 DETECTION RATES:")
    for name, stats in summary['scales'].items():
        status = "✅" if stats['detection_rate'] > 0.5 else "⚠️" if stats['detection_rate'] > 0.3 else "❌"
        print(f"  {name:<15} {stats['detection_rate']:>6.1%} {status}")
    
    print("\n" + "="*70)
    print(f"📁 Output completo in: {output_dir}")
    print("="*70)
    print("\n🌟 Simone & Lucy - Harmonic Power Analysis Complete! 🌟\n")

if __name__ == '__main__':
    main()
