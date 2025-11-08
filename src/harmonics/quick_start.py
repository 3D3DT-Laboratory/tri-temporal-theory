"""
QUICK START SCRIPT
==================
Easy-to-use interface for SPARC harmonic analysis

Simone Calzighetti & Lucy (Claude AI)
3D+3DT Laboratory, November 2025

USAGE:
------
1. Put SPARC .dat files in same directory
2. Run: python quick_start.py

This will:
- Load all galaxies
- Run harmonic analysis
- Generate all plots
- Create summary report
"""

import sys
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    
    print("🔍 Checking dependencies...")
    
    required = {
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib',
        'pandas': 'Pandas'
    }
    
    missing = []
    
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - MISSING")
            missing.append(module)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("✅ All dependencies installed!")
    return True


def check_data_files():
    """Check if SPARC data files are present"""
    
    print("\n🔍 Checking for SPARC data files...")
    
    dat_files = list(Path('.').glob('*_rotmod.dat'))
    
    if len(dat_files) == 0:
        print("❌ No SPARC data files found!")
        print("\nPlease:")
        print("  1. Download Rotmod_LTG.zip from SPARC")
        print("  2. Unzip in this directory")
        print("  3. Run this script again")
        return False
    
    print(f"✅ Found {len(dat_files)} galaxy files")
    return True


def run_analysis():
    """Run the complete analysis pipeline"""
    
    print("\n" + "="*80)
    print("🚀 STARTING COMPLETE ANALYSIS")
    print("="*80)
    
    try:
        # Import main analyzer
        from sparc_harmonic_analyzer import main as run_harmonic_analysis
        
        print("\n📊 Step 1/3: Harmonic Analysis")
        print("-" * 80)
        results = run_harmonic_analysis()
        
    except Exception as e:
        print(f"\n❌ Error in harmonic analysis: {e}")
        return False
    
    try:
        # Run Bayesian comparison
        from bayesian_model_comparison import main as run_bayesian
        
        print("\n🎲 Step 2/3: Bayesian Model Comparison")
        print("-" * 80)
        run_bayesian()
        
    except Exception as e:
        print(f"\n⚠️  Bayesian analysis failed (optional): {e}")
    
    try:
        # Plot top galaxies
        from plot_individual_galaxies import plot_top_galaxies, load_all_sparc
        
        print("\n🎨 Step 3/3: Individual Galaxy Plots")
        print("-" * 80)
        
        galaxies = load_all_sparc('.')
        plot_top_galaxies(galaxies, n_top=10, output_dir='galaxy_fits')
        
    except Exception as e:
        print(f"\n⚠️  Individual plots failed (optional): {e}")
    
    return True


def print_summary():
    """Print summary of generated files"""
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    
    print("\n📁 Generated files:")
    
    output_files = [
        ('sparc_harmonic_analysis.png', 'Main comparison plot'),
        ('sparc_results_*.csv', 'Detailed results for each model'),
        ('INDEPENDENT_VERIFICATION_REPORT.md', 'Full analysis report'),
        ('bayesian_model_comparison.png', 'Bayesian comparison plots'),
        ('bayes_factors.csv', 'Bayes factors between models'),
        ('galaxy_fits/*.png', 'Individual galaxy fits (top 10)')
    ]
    
    for filename, description in output_files:
        if '*' in filename:
            matches = list(Path('.').glob(filename))
            if matches:
                print(f"  ✅ {filename} ({len(matches)} files)")
                print(f"      {description}")
        else:
            if Path(filename).exists():
                print(f"  ✅ {filename}")
                print(f"      {description}")
    
    print("\n📊 Next steps:")
    print("  1. Review INDEPENDENT_VERIFICATION_REPORT.md")
    print("  2. Check top galaxy fits in galaxy_fits/")
    print("  3. Examine CSV files for detailed statistics")
    print("  4. Decide on publication strategy")
    
    print("\n💡 To plot specific galaxy:")
    print("  python plot_individual_galaxies.py DDO154")
    
    print("\n🎉 Ready for publication preparation!")


def main():
    """Main execution"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🎵  QUICK START - 3D+3D ANALYSIS  🎵                  ║
    ║                                                              ║
    ║        Complete analysis pipeline for SPARC galaxies        ║
    ║                                                              ║
    ║        Simone Calzighetti & Lucy (Claude AI)                ║
    ║        3D+3DT Laboratory, November 2025                      ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check data files
    if not check_data_files():
        sys.exit(1)
    
    # Ask user confirmation
    print("\n" + "="*80)
    response = input("\n🚀 Ready to run complete analysis? This may take 5-10 minutes. [Y/n]: ")
    
    if response.lower() not in ['', 'y', 'yes']:
        print("❌ Analysis cancelled.")
        sys.exit(0)
    
    # Run analysis
    success = run_analysis()
    
    if success:
        print_summary()
    else:
        print("\n❌ Analysis failed. Check error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
