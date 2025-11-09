# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
# All rights reserved. Unauthorized copying, modification, or distribution is prohibited.

# Known Issues and Troubleshooting

Common problems, solutions, and workarounds for the Tri-Temporal Theory validation package.

**Last Updated:** November 2025  
**Version:** 2.3

---

## 📋 Table of Contents

1. [Installation Issues](#installation-issues)
2. [Data Loading Problems](#data-loading-problems)
3. [Analysis Errors](#analysis-errors)
4. [Performance Issues](#performance-issues)
5. [Plotting Problems](#plotting-problems)
6. [Platform-Specific Issues](#platform-specific-issues)
7. [Getting Help](#getting-help)

---

## 🔧 Installation Issues

### Issue 1: NumPy Version Conflict

**Problem:**
```
ERROR: Cannot install numpy>=2.3 and scipy>=1.11 together
```

**Cause:** NumPy 2.x broke compatibility with older SciPy

**Solution:**
```bash
# Option 1: Upgrade both
pip install --upgrade numpy scipy

# Option 2: Use NumPy 1.x
pip install "numpy<2.0" scipy

# Option 3: Use conda (handles dependencies)
conda install numpy scipy matplotlib pandas
```

**Prevention:** Use virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### Issue 2: Pandas Dataframe Warning

**Problem:**
```
FutureWarning: DataFrame.applymap has been deprecated. Use DataFrame.map instead.
```

**Cause:** Pandas 2.1+ deprecated `applymap`

**Solution:**
Update code or suppress warning:
```python
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
```

**Fix:** We'll update code in v2.4

---

### Issue 3: Missing C Compiler (Windows)

**Problem:**
```
error: Microsoft Visual C++ 14.0 or greater is required
```

**Cause:** SciPy requires C compiler on Windows

**Solution:**
```powershell
# Option 1: Install pre-compiled wheels
pip install --only-binary :all: scipy

# Option 2: Install Build Tools
# Download from: https://visualstudio.microsoft.com/downloads/
# Install "Desktop development with C++"

# Option 3: Use conda (includes compiled binaries)
conda install scipy
```

---

## 📂 Data Loading Problems

### Issue 4: SPARC Archive Not Found

**Problem:**
```
FileNotFoundError: data/Rotmod_LTG.zip not found
```

**Solution:**
```bash
# Download SPARC data
wget http://astroweb.case.edu/SPARC/RotMod_LTG.zip -O data/Rotmod_LTG.zip

# Or specify custom path
python -m src.analysis.rar_relation --rotmod /path/to/Rotmod_LTG.zip
```

**Alternative:** Use Zenodo mirror
```bash
wget https://zenodo.org/record/16284118/files/Rotmod_LTG.zip
```

---

### Issue 5: RAR Data Parsing Error

**Problem:**
```
ParserError: Error tokenizing data. C error: Expected 5 fields, saw 6
```

**Cause:** Corrupted or incorrect CSV format

**Solution:**
```python
# Check file integrity
import pandas as pd
df = pd.read_csv('data/processed/rar_data.csv', on_bad_lines='warn')

# Re-generate from SPARC
python -m src.analysis.rar_relation --rotmod data/Rotmod_LTG.zip
```

---

### Issue 6: Missing Columns

**Problem:**
```
KeyError: 'gbar' not found in DataFrame
```

**Cause:** Old RAR data format or corrupted file

**Solution:**
```python
# Verify columns
df = pd.read_csv('data/processed/rar_data.csv')
print(df.columns.tolist())
# Expected: ['galaxy', 'r_kpc', 'gbar', 'gobs', 'e_gobs', ...]

# If missing, regenerate:
python -m src.analysis.rar_relation --rotmod data/Rotmod_LTG.zip --force
```

---

## 🔬 Analysis Errors

### Issue 7: Optimization Failure

**Problem:**
```
RuntimeWarning: invalid value encountered in scalar divide
OptimizeWarning: Desired error not necessarily achieved
```

**Cause:** Bad initial parameters or numerical instability

**Solution:**
```python
# In rar_fit_logspace.py, adjust bounds:
bounds = [
    (0.5, 0.8),     # gamma (was 0.6-0.7)
    (1e-11, 1e-9)   # g0 (wider range)
]

# Or use robust optimizer:
from scipy.optimize import differential_evolution
result = differential_evolution(objective, bounds, seed=42)
```

---

### Issue 8: NaN in Residuals

**Problem:**
```
RuntimeWarning: invalid value in log10
ValueError: Input contains NaN
```

**Cause:** Zero or negative values in g_bar/g_obs

**Solution:**
```python
# Filter invalid data
mask = (data['gbar'] > 0) & (data['gobs'] > 0) & np.isfinite(data['e_gobs'])
data_clean = data[mask].copy()

# Or use safe log
def log10_safe(x):
    return np.log10(np.maximum(x, 1e-20))
```

---

### Issue 9: Memory Error on Large Dataset

**Problem:**
```
MemoryError: Unable to allocate array with shape (10000, 10000)
```

**Cause:** Processing too many galaxies at once

**Solution:**
```python
# Process in batches
batch_size = 50
for i in range(0, len(galaxies), batch_size):
    batch = galaxies[i:i+batch_size]
    results = analyze_batch(batch)
    save_results(results, f'batch_{i}.json')

# Or increase memory limit
import resource
resource.setrlimit(resource.RLIMIT_AS, (8e9, 8e9))  # 8GB
```

---

### Issue 10: FFT Fails with Non-Uniform Grid

**Problem:**
```
ValueError: FFT requires evenly spaced samples
```

**Cause:** Rotation curve has irregular radial spacing

**Solution:**
```python
# Interpolate to uniform grid
from scipy.interpolate import interp1d

R_uniform = np.linspace(R.min(), R.max(), 256)
V_interp = interp1d(R, V, kind='cubic')(R_uniform)

# Then apply FFT
fft = np.fft.rfft(V_interp)
```

---

## ⚡ Performance Issues

### Issue 11: Tests Run Slowly

**Problem:** Test suite takes >10 seconds

**Causes:**
- Loading large datasets repeatedly
- Not using fixtures
- Expensive computations in each test

**Solution:**
```python
# Use class-level setup (runs once)
@classmethod
def setUpClass(cls):
    cls.data = load_expensive_data()  # Load once

# Use pytest fixtures
@pytest.fixture(scope="session")
def sparc_data():
    return load_data_once()  # Cached

# Skip slow tests in development
@pytest.mark.slow
def test_full_sample():
    ...

# Run: pytest -m "not slow"
```

---

### Issue 12: Analysis Takes Hours

**Problem:** Harmonic analysis on 175 galaxies takes 2+ hours

**Solution:**
```python
# Parallelize with multiprocessing
from multiprocessing import Pool

def analyze_one(galaxy_data):
    return detect_harmonics(galaxy_data)

with Pool(processes=8) as pool:
    results = pool.map(analyze_one, galaxy_list)

# Or use joblib (better for NumPy)
from joblib import Parallel, delayed
results = Parallel(n_jobs=8)(
    delayed(analyze_one)(g) for g in galaxy_list
)
```

**Expected speedup:** 4-8× on modern CPUs

---

### Issue 13: High Memory Usage

**Problem:** Python process uses >4GB RAM

**Cause:** Storing all intermediate results

**Solution:**
```python
# Use generators instead of lists
def process_galaxies(galaxy_list):
    for galaxy in galaxy_list:
        result = analyze(galaxy)
        yield result  # Don't store all in memory

# Process and save incrementally
for i, result in enumerate(process_galaxies(galaxies)):
    save_result(result, f'galaxy_{i}.json')
    del result  # Free memory

# Force garbage collection
import gc
gc.collect()
```

---

## 📊 Plotting Problems

### Issue 14: Matplotlib Backend Error

**Problem:**
```
UserWarning: Matplotlib is currently using agg, a non-GUI backend
```

**Cause:** No GUI available (SSH session, Docker, etc.)

**Solution:**
```python
# Use non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Or specify in script
plt.switch_backend('Agg')

# Save instead of show
plt.savefig('plot.png')
# Don't call plt.show()
```

---

### Issue 15: Figures Cut Off

**Problem:** Axis labels or titles truncated in saved figures

**Solution:**
```python
# Use tight_layout
plt.tight_layout()
plt.savefig('plot.png')

# Or manually adjust
plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)

# Or save with bbox
plt.savefig('plot.png', bbox_inches='tight')
```

---

### Issue 16: Font Warnings

**Problem:**
```
findfont: Font family 'Arial' not found
```

**Solution:**
```python
# Use default fonts
plt.rcParams['font.family'] = 'DejaVu Sans'

# Or install fonts
# Linux: sudo apt-get install msttcorefonts
# Mac: Already has Arial
# Windows: Already has Arial

# Or suppress warning
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
```

---

## 💻 Platform-Specific Issues

### Windows Issues

**Issue 17: Path Separators**

**Problem:** Paths with `\` don't work

**Solution:**
```python
# Use pathlib (cross-platform)
from pathlib import Path
data_path = Path('data') / 'processed' / 'rar_data.csv'

# Or raw strings
path = r'C:\Users\Name\tri-temporal-theory\data\rar_data.csv'

# Or forward slashes (work on Windows!)
path = 'C:/Users/Name/tri-temporal-theory/data/rar_data.csv'
```

---

**Issue 18: PowerShell Script Execution**

**Problem:**
```
cannot be loaded because running scripts is disabled
```

**Solution:**
```powershell
# Allow scripts in current session
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# Or activate venv differently
.venv\Scripts\Activate.ps1  # Instead of activate
```

---

### macOS Issues

**Issue 19: M1/M2 Compatibility**

**Problem:** NumPy/SciPy installation fails on Apple Silicon

**Solution:**
```bash
# Use conda with osx-arm64 channel
conda create -n ttn python=3.11
conda activate ttn
conda install -c conda-forge numpy scipy matplotlib pandas

# Or use Rosetta 2
arch -x86_64 pip install -r requirements.txt
```

---

**Issue 20: Permission Denied**

**Problem:**
```
PermissionError: [Errno 13] Permission denied: 'outputs/rar'
```

**Solution:**
```bash
# Create directory first
mkdir -p outputs/rar outputs/harmonics

# Fix permissions
chmod -R 755 outputs/

# Or run with sudo (not recommended)
sudo python analysis.py
```

---

### Linux Issues

**Issue 21: libGL.so Missing**

**Problem:**
```
ImportError: libGL.so.1: cannot open shared object file
```

**Cause:** Matplotlib needs OpenGL libraries

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx

# RHEL/CentOS
sudo yum install mesa-libGL

# Or use headless backend
export MPLBACKEND=Agg
```

---

**Issue 22: Segmentation Fault**

**Problem:** Python crashes with "Segmentation fault (core dumped)"

**Cause:** Usually NumPy/BLAS incompatibility

**Solution:**
```bash
# Reinstall NumPy with different BLAS
pip uninstall numpy
pip install numpy --no-binary numpy

# Or use system BLAS
sudo apt-get install libblas-dev liblapack-dev
pip install numpy --no-binary :all:

# Check which BLAS is used
python -c "import numpy; numpy.show_config()"
```

---

## 🔍 Debugging Tips

### General Debugging Strategy

```python
# 1. Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# 2. Add debug prints
print(f"Shape: {data.shape}")
print(f"Range: {data.min():.2e} to {data.max():.2e}")
print(f"NaNs: {np.isnan(data).sum()}")

# 3. Save intermediate results
np.save('debug_data.npy', data)
df.to_csv('debug_df.csv')

# 4. Use pdb debugger
import pdb; pdb.set_trace()  # Breakpoint

# 5. Profile slow code
import cProfile
cProfile.run('analyze_all()', 'profile.stats')
```

### Checking Data Integrity

```python
def validate_rar_data(df):
    """Check RAR data for common issues"""
    issues = []
    
    # Required columns
    required = ['galaxy', 'r_kpc', 'gbar', 'gobs', 'e_gobs']
    missing = set(required) - set(df.columns)
    if missing:
        issues.append(f"Missing columns: {missing}")
    
    # Check for NaNs
    for col in required:
        if df[col].isna().any():
            n_nan = df[col].isna().sum()
            issues.append(f"{col} has {n_nan} NaN values")
    
    # Check for non-positive values
    for col in ['gbar', 'gobs']:
        if (df[col] <= 0).any():
            n_bad = (df[col] <= 0).sum()
            issues.append(f"{col} has {n_bad} non-positive values")
    
    # Check for outliers
    for col in ['gbar', 'gobs']:
        median = df[col].median()
        mad = (df[col] - median).abs().median()
        outliers = (df[col] - median).abs() > 5 * 1.4826 * mad
        if outliers.any():
            n_out = outliers.sum()
            issues.append(f"{col} has {n_out} outliers (>5 MAD)")
    
    return issues

# Use:
issues = validate_rar_data(data)
if issues:
    print("Data issues found:")
    for issue in issues:
        print(f"  - {issue}")
```

---

## ❓ Getting Help

### Before Asking

**Checklist:**
- [ ] Read error message carefully
- [ ] Check this document for solution
- [ ] Search GitHub issues
- [ ] Try minimal example
- [ ] Check Python/package versions

### Where to Ask

**GitHub Issues:**
- Bug reports: Use `[bug]` tag
- Feature requests: Use `[feature]` tag
- Questions: Use `[question]` tag

**Email:**
- Technical issues: condoor76@gmail.com
- Include: Error message, code snippet, system info

**Provide This Information:**

```python
# System info
import sys, platform
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")

# Package versions
import numpy, scipy, pandas, matplotlib
print(f"NumPy: {numpy.__version__}")
print(f"SciPy: {scipy.__version__}")
print(f"Pandas: {pandas.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")

# Minimal reproducible example
import numpy as np
data = np.array([1, 2, 3])
result = buggy_function(data)  # Fails here
```

---

## 🆘 Emergency Fixes

### Nuclear Option: Clean Reinstall

```bash
# 1. Remove everything
rm -rf .venv/
rm -rf outputs/
rm -rf __pycache__/
rm -rf src/__pycache__/

# 2. Fresh virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Reinstall from scratch
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify installation
pytest tests/ -v

# 5. Re-run analysis
python -m src.analysis.rar_relation --rotmod data/Rotmod_LTG.zip
```

### Reset to Known-Good State

```bash
# Git: Reset to last working commit
git stash  # Save local changes
git checkout v2.3  # Known good version
pip install -r requirements.txt
pytest tests/

# If tests pass:
git checkout main
git stash pop  # Restore changes
```

---

## 📚 Additional Resources

**Documentation:**
- README.md - General overview
- CONTRIBUTING.md - Development guide
- tests/README.md - Testing guide

**External:**
- NumPy docs: https://numpy.org/doc/
- SciPy docs: https://docs.scipy.org/
- Pandas docs: https://pandas.pydata.org/docs/
- Matplotlib docs: https://matplotlib.org/

**Stack Overflow Tags:**
- numpy
- scipy
- pandas
- matplotlib
- python-3.x

---

## 🔄 Reporting New Issues

**GitHub Issue Template:**

```markdown
**Issue Type:** Bug / Feature Request / Question

**Description:**
Clear description of the problem or request.

**Steps to Reproduce:**
1. Load data: `python -m src.analysis.rar_relation`
2. Run analysis: `...`
3. Error occurs at line X

**Expected Behavior:**
What should happen.

**Actual Behavior:**
What actually happens (include error message).

**Environment:**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.12.1]
- NumPy: [e.g., 2.3.1]
- Repo version: [e.g., v2.3]

**Additional Context:**
- Does it happen with synthetic data?
- Does it happen with all galaxies?
- Minimal code example?
```

---

**Most issues have solutions — don't give up!** 💪

*Last updated: November 2025*  
*3D+3D Spacetime Laboratory*
