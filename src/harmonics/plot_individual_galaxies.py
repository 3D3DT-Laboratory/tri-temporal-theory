"""
INDIVIDUAL GALAXY PLOTTER
==========================
Plot rotation curves with 3D+3D harmonic fits

Simone Calzighetti & Lucy (Claude AI)
3D+3DT Laboratory, November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Import from main analyzer
from sparc_harmonic_analyzer import (
    load_sparc_galaxy, 
    HarmonicModel, 
    G
)

def plot_galaxy_fit(galaxy, λ_scales=[1.89, 4.30, 11.7], save_path=None):
    """
    Plot single galaxy with harmonic fit
    
    Parameters:
    -----------
    galaxy : SPARCGalaxy
        Galaxy to plot
    λ_scales : list
        Breathing scales to use
    save_path : str
        Where to save figure
    """
    
    # Fit the model
    model = HarmonicModel(λ_scales)
    fit_result = model.fit(galaxy, verbose=True)
    
    # Generate smooth model curve
    r_smooth = np.linspace(galaxy.r.min(), galaxy.r.max(), 200)
    
    # Baryonic velocity on smooth grid
    v_bary_smooth = np.interp(r_smooth, galaxy.r, galaxy.v_bary)
    
    # Model velocity
    v_model_smooth = model.predict_velocity(
        r_smooth, 
        v_bary_smooth, 
        fit_result['Q_params']
    )
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                     gridspec_kw={'height_ratios': [3, 1]})
    
    # Main plot
    ax1.errorbar(galaxy.r, galaxy.v_obs, yerr=galaxy.v_err,
                fmt='o', color='black', markersize=8, capsize=3,
                label='Observed', zorder=3, alpha=0.8)
    
    ax1.plot(galaxy.r, galaxy.v_bary, 'o--', color='gray', 
            markersize=6, linewidth=2, label='Baryons only', alpha=0.6)
    
    ax1.plot(r_smooth, v_model_smooth, '-', color='red', 
            linewidth=3, label='3D+3D model', zorder=2)
    
    # Add component contributions
    for i, λ in enumerate(λ_scales):
        Q = fit_result['Q_params'][i]
        phase = 2 * np.pi * r_smooth / λ
        v_component = np.sqrt(Q**2 * v_bary_smooth**2 * np.sin(phase)**2)
        
        ax1.fill_between(r_smooth, v_bary_smooth, v_bary_smooth + v_component,
                        alpha=0.2, label=f'λ={λ:.2f} kpc (Q={Q:.2f})')
    
    ax1.set_ylabel('Velocity (km/s)', fontsize=14, fontweight='bold')
    ax1.set_title(f'{galaxy.name} - 3D+3D Harmonic Fit\n' + 
                 f'Δχ² = +{fit_result["Δχ2"]:.0f} ({fit_result["improvement_pct"]:.1f}% improvement)',
                 fontsize=16, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Residuals
    v_model_data = model.predict_velocity(galaxy.r, galaxy.v_bary, 
                                          fit_result['Q_params'])
    residuals = galaxy.v_obs - v_model_data
    
    ax2.errorbar(galaxy.r, residuals, yerr=galaxy.v_err,
                fmt='o', color='blue', markersize=6, capsize=3, alpha=0.7)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.fill_between(galaxy.r, -galaxy.v_err, galaxy.v_err, 
                    color='gray', alpha=0.2)
    
    ax2.set_xlabel('Radius (kpc)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Residuals (km/s)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Statistics box
    stats_text = (
        f"χ²_bary = {fit_result['χ2_baseline']:.1f}\n"
        f"χ²_3D+3D = {fit_result['χ2_fit']:.1f}\n"
        f"Improvement = {fit_result['improvement_pct']:.1f}%\n"
        f"N_data = {fit_result['n_data']}"
    )
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    return fig


def plot_top_galaxies(galaxies, n_top=5, output_dir='galaxy_fits'):
    """
    Plot top N galaxies by improvement
    
    Parameters:
    -----------
    galaxies : list of SPARCGalaxy
        All galaxies
    n_top : int
        Number of top galaxies to plot
    output_dir : str
        Directory for output files
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\n🎨 Finding top {n_top} galaxies...")
    
    # Fit all galaxies to find best ones
    model = HarmonicModel([1.89, 4.30, 11.7])
    
    improvements = []
    for galaxy in galaxies:
        try:
            result = model.fit(galaxy, verbose=False)
            improvements.append((galaxy, result['improvement_pct']))
        except:
            pass
    
    # Sort by improvement
    improvements.sort(key=lambda x: x[1], reverse=True)
    
    # Plot top N
    print(f"\n📊 Plotting top {n_top} galaxies...")
    
    for i, (galaxy, improvement) in enumerate(improvements[:n_top], 1):
        print(f"\n[{i}/{n_top}] {galaxy.name} ({improvement:.1f}% improvement)")
        
        save_path = output_path / f"{i:02d}_{galaxy.name}_fit.png"
        plot_galaxy_fit(galaxy, save_path=str(save_path))
        
        plt.close()
    
    print(f"\n✅ All plots saved to {output_dir}/")


def main():
    """Main execution"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🎨  INDIVIDUAL GALAXY PLOTTER  🎨                     ║
    ║                                                              ║
    ║        Creates publication-quality rotation curve fits      ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check arguments
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python plot_individual_galaxies.py <galaxy_name>")
        print("  python plot_individual_galaxies.py top5")
        print("  python plot_individual_galaxies.py all")
        print("\nExamples:")
        print("  python plot_individual_galaxies.py DDO154")
        print("  python plot_individual_galaxies.py IC2574")
        print("  python plot_individual_galaxies.py top5")
        sys.exit(1)
    
    target = sys.argv[1].lower()
    
    if target == 'top5':
        # Plot top 5 galaxies
        from sparc_harmonic_analyzer import load_all_sparc
        galaxies = load_all_sparc('.')
        plot_top_galaxies(galaxies, n_top=5, output_dir='galaxy_fits')
        
    elif target == 'all':
        # Plot all galaxies
        from sparc_harmonic_analyzer import load_all_sparc
        galaxies = load_all_sparc('.')
        plot_top_galaxies(galaxies, n_top=len(galaxies), output_dir='galaxy_fits')
        
    else:
        # Plot specific galaxy
        filepath = f"{target.upper()}_rotmod.dat"
        
        if not Path(filepath).exists():
            # Try with different case
            dat_files = list(Path('.').glob('*_rotmod.dat'))
            matches = [f for f in dat_files if target in f.stem.lower()]
            
            if not matches:
                print(f"❌ Galaxy file not found: {filepath}")
                print(f"\nAvailable galaxies:")
                for f in sorted(dat_files)[:10]:
                    print(f"  - {f.stem.replace('_rotmod', '')}")
                print(f"  ... and {len(dat_files)-10} more")
                sys.exit(1)
            
            filepath = matches[0]
        
        print(f"\n📂 Loading {filepath}...")
        galaxy = load_sparc_galaxy(filepath)
        
        print(f"\n🎨 Plotting {galaxy.name}...")
        fig = plot_galaxy_fit(galaxy, save_path=f"{galaxy.name}_fit.png")
        
        print(f"\n✅ Done! Figure saved as {galaxy.name}_fit.png")
        
        # Show plot
        plt.show()


if __name__ == "__main__":
    main()
