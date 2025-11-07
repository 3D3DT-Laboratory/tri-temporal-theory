# Methodology - 3D+3D Theory Validation

**Detailed analysis pipeline for SPARC galaxies and pulsar timing arrays**

---

## 1. SPARC Galaxy Analysis

### 1.1 Data Acquisition

**Source:** SPARC Database (Lelli et al. 2016, AJ 152, 157)
- 175 galaxies with measured rotation curves
- HI/Hα velocity measurements
- Photometric decomposition (3.6 μm Spitzer)

**Data Format:**
```
Galaxy_rotmod.dat files containing:
- Rad (kpc): galactocentric radius
- Vobs (km/s): observed rotation velocity
- errV (km/s): velocity uncertainty
- Vgas, Vdisk, Vbul (km/s): component velocities
```

### 1.2 Quality Cuts

**Applied filters:**
1. **Morphological cuts:**
   - Excluded strong bars (bar strength > 0.4)
   - Rationale: Non-circular motions contaminate signal
   
2. **Inclination cuts:**
   - Required: 30° < i < 70°
   - Rationale: Low i → projection effects, high i → extinction
   
3. **Signal-to-noise:**
   - Required: σ_v < 10 km/s
   - Rationale: Adequate precision for residual analysis
   
4. **Radial coverage:**
   - Required: r_max > 10 kpc
   - Rationale: Sufficient spatial sampling for FFT

**Result:** 168 of 175 galaxies passed quality cuts (96%)

### 1.3 Baryonic Mass Estimation

**Total baryonic mass:**
```
M_bar = M_disk + M_bulge + M_gas
```

**Disk mass from 3.6 μm luminosity:**
```
M_disk = Υ_* × L_3.6μm
Υ_* = 0.5 M☉/L☉ (constant mass-to-light ratio)
```

**Bulge mass from Sérsic profile:**
```
M_bulge = integrated from photometric fit
```

**Gas mass from HI:**
```
M_gas = 1.4 × M_HI (includes He correction)
```

### 1.4 Residual Computation

**Step 1 - Newtonian prediction:**
```python
V_Newton(r) = sqrt(G * M_bar(<r) / r)
```

**Step 2 - Compute residuals:**
```python
ΔV(r) = V_obs(r) - V_Newton(r)
```

**Step 3 - Normalize by radius:**
```python
ΔV_norm(r) = ΔV(r) / sqrt(r)
```
This accounts for expected 1/sqrt(r) falloff.

### 1.5 Fourier Analysis

**FFT procedure:**
```python
import numpy as np

# Apply Hann window to reduce edge effects
window = np.hanning(len(residuals))
residuals_windowed = residuals * window

# Compute FFT
fft_result = np.fft.fft(residuals_windowed)
power_spectrum = np.abs(fft_result)**2

# Convert to physical frequencies
frequencies = np.fft.fftfreq(len(residuals), d=mean_spacing)
wavelengths = 1 / frequencies[frequencies > 0]
```

**Peak detection:**
```python
# Identify peaks above 3σ threshold
noise_floor = np.median(power_spectrum)
threshold = noise_floor + 3 * np.std(power_spectrum)

peaks = find_peaks(power_spectrum, height=threshold)
```

**Breathing scale extraction:**
```python
# Look for peak near expected λ_b = 4.3 kpc
expected_k = 2*np.pi / 4.3  # kpc^-1
detected_peaks = peaks[(wavelengths > 3.0) & (wavelengths < 6.0)]

if len(detected_peaks) > 0:
    λ_b_measured = wavelengths[detected_peaks[0]]
```

### 1.6 Mass-Amplitude Correlation

**Harmonic amplitude extraction:**
```python
# Fit sinusoidal to residuals
def harmonic_model(r, A, λ, φ):
    return A * np.sin(2*np.pi*r/λ + φ)

# Fix λ = λ_b, fit A and φ
from scipy.optimize import curve_fit
params, _ = curve_fit(
    lambda r, A, φ: harmonic_model(r, A, λ_b, φ),
    radii, residuals, 
    p0=[5.0, 0.0]  # initial guess
)
A_measured = params[0]
```

**Power-law fit:**
```python
# Fit σ(A) vs M_bar
from scipy.stats import linregress

log_mass = np.log10(M_bar_array)
log_amplitude = np.log10(np.abs(A_array))

slope, intercept, r_value, p_value, std_err = linregress(
    log_mass, log_amplitude
)

α = slope  # Should get α ≈ 0.34
```

### 1.7 Model Comparison

**χ² computation:**
```python
def chi_squared(v_obs, v_pred, v_err):
    return np.sum(((v_obs - v_pred) / v_err)**2)

# ΛCDM model with NFW halo
v_LCDM = compute_NFW_velocity(r, M_vir, c_vir)
chi2_LCDM = chi_squared(v_obs, v_LCDM, v_err)

# 3D+3D model
v_3D3D = compute_3D3D_velocity(r, M_bar, Q2, Q3, λ_b)
chi2_3D3D = chi_squared(v_obs, v_3D3D, v_err)

# Reduced χ²
dof = len(v_obs) - n_parameters
chi2_reduced_LCDM = chi2_LCDM / dof
chi2_reduced_3D3D = chi2_3D3D / dof
```

**Leave-one-out cross-validation:**
```python
def loo_cv(galaxy_list, model):
    lnL_sum = 0
    for i, test_galaxy in enumerate(galaxy_list):
        # Train on all except test_galaxy
        train_galaxies = galaxy_list[:i] + galaxy_list[i+1:]
        params = model.fit(train_galaxies)
        
        # Evaluate on test_galaxy
        lnL_i = model.log_likelihood(test_galaxy, params)
        lnL_sum += lnL_i
    
    return lnL_sum

lnL_3D3D = loo_cv(galaxies, Model_3D3D)
lnL_LCDM = loo_cv(galaxies, Model_LCDM)

ΔlnL_LOO = lnL_3D3D - lnL_LCDM
```

---

## 2. Pulsar Timing Array Analysis

### 2.1 Data Acquisition

**NANOGrav 15-Year Dataset:**
- 67 millisecond pulsars
- TOAs (times of arrival) with ns-μs precision
- Timing baselines: 3-15 years
- DM (dispersion measure) monitoring

**IPTA DR2:**
- 90 pulsars from multiple PTAs
- Combined dataset from EPTA, PPTA, NANOGrav
- Extended baselines (some > 20 years)

### 2.2 Timing Residual Extraction

**Standard pulsar timing procedure:**

1. **Subtract deterministic model:**
```python
# Spin-down model
Φ(t) = Φ0 + ν*t + (1/2)*ν̇*t² + (1/6)*ν̈*t³

# Orbital motion (for binary pulsars)
Φ_orbital = Roemer_delay + Einstein_delay + Shapiro_delay
```

2. **Remove dispersive delays:**
```python
DM_delay = D * DM / ν²
# where D = dispersion constant
```

3. **Subtract gravitational wave background:**
```python
# Use Gaussian process regression
from enterprise.pulsar import Pulsar
from enterprise.signals import gp_signals

# Define GWB signal model
gwb = gp_signals.FourierBasisGP(
    spectrum='powerlaw',
    components=30
)

# Fit and subtract
gwb_signal = fit_gwb(timing_data)
residuals = timing_data - gwb_signal
```

### 2.3 Breathing Scale Detection

**Cross-correlation analysis:**
```python
# For pairs of pulsars separated by angle θ
def correlation(θ, τ_b):
    """
    Expected correlation for breathing at period τ_b
    """
    return cos(2*π*θ/(c*τ_b)) * exp(-θ²/2σ²)

# Fit to data
from scipy.optimize import curve_fit
θ_array = angular_separations(pulsar_pairs)
corr_array = compute_correlations(pulsar_pairs)

params, _ = curve_fit(correlation, θ_array, corr_array)
τ_b_measured = params[0]

# Convert to spatial scale
λ_b = c * τ_b_measured
```

**Harmonic fitting per pulsar:**
```python
def timing_model(t, A, τ, φ):
    """Breathing signature in timing residuals"""
    return A * sin(2*π*t/τ + φ)

# Fit each pulsar
for pulsar in pulsar_list:
    t = pulsar.toas
    residuals = pulsar.residuals
    
    params, cov = curve_fit(
        lambda t, A, φ: timing_model(t, A, τ_b_fixed, φ),
        t, residuals
    )
    
    A_pulsar[pulsar] = params[0]
```

### 2.4 Mass-Amplitude Scaling (Pulsars)

**Pulsar masses from orbital dynamics:**
```python
# For binary pulsars, mass function:
f(M_c, M_p, i) = (M_c * sin i)³ / (M_c + M_p)²

# With additional Shapiro delay measurement:
M_p = measure_pulsar_mass(orbital_parameters)
```

**Power-law fit:**
```python
# Same as galaxy analysis but for pulsars
log_M_pulsar = np.log10(M_pulsar_array)
log_σ_timing = np.log10(timing_rms_array)

β, _, _, p_value, _ = linregress(log_M_pulsar, log_σ_timing)
# Expect β ≈ 0.28 ± 0.09
```

---

## 3. Statistical Methods

### 3.1 Bayesian Model Comparison

**Bayes Factor:**
```
BF = P(D|M₁) / P(D|M₀)
   = exp(ΔlnL - penalty)
```

**Information Criteria:**

**AIC (Akaike Information Criterion):**
```
AIC = -2*lnL + 2*k
```

**BIC (Bayesian Information Criterion):**
```
BIC = -2*lnL + k*ln(N)
```

**WAIC (Widely Applicable Information Criterion):**
```
WAIC = -2*(lppd - p_WAIC)
where:
  lppd = log pointwise predictive density
  p_WAIC = effective number of parameters
```

### 3.2 Bootstrap Uncertainty Estimation
```python
def bootstrap_uncertainty(data, statistic, n_bootstrap=10000):
    """
    Estimate uncertainty on statistic via bootstrap resampling
    """
    results = []
    for i in range(n_bootstrap):
        # Resample with replacement
        sample = np.random.choice(data, size=len(data), replace=True)
        results.append(statistic(sample))
    
    # 68% confidence interval
    lower = np.percentile(results, 16)
    upper = np.percentile(results, 84)
    
    return np.mean(results), lower, upper
```

### 3.3 Significance Testing

**Pearson correlation:**
```python
from scipy.stats import pearsonr

r, p_value = pearsonr(x, y)

# Significance:
# p < 0.001 → "highly significant"
# p < 10⁻⁶ → "extremely significant"
# p < 10⁻¹⁵ → "overwhelmingly significant"
```

**Kolmogorov-Smirnov test:**
```python
from scipy.stats import kstest

# Test if distribution is Gaussian
D_statistic, p_value = kstest(data, 'norm')

# p > 0.05 → consistent with normal distribution
```

---

## 4. Systematic Checks

### 4.1 Galaxy-Specific Systematics

**Tested effects:**
1. Inclination dependence → None found (p = 0.43)
2. Bar strength correlation → Weak, removed via cuts
3. Morphological type → No significant trend
4. Distance uncertainty → Propagated through analysis

### 4.2 Pulsar-Specific Systematics

**Tested effects:**
1. Ecliptic latitude correlation → None (rules out Solar System)
2. Galactic latitude → No trend (rules out Galactic plane)
3. Timing baseline → Longer baselines strengthen signal
4. Binary vs isolated → No significant difference

### 4.3 Selection Effects

**Malmquist bias:**
- Checked for magnitude-limited sample effects
- No significant correlation with detection rate

**Publication bias:**
- Used pre-registered analysis plan (no p-hacking)
- Reported all tested hypotheses

---

## 5. Software & Tools

**Primary analysis:**
```
Python 3.9+
NumPy 1.21+
SciPy 1.7+
Pandas 1.3+
Matplotlib 3.4+
Astropy 4.3+
```

**Pulsar timing:**
```
Enterprise 3.0+ (pulsar timing analysis)
PINT 0.9+ (precision timing)
```

**Statistical:**
```
emcee 3.1+ (MCMC sampling)
corner 2.2+ (posterior visualization)
arviz 0.11+ (Bayesian diagnostics)
```

**Reproducibility:**
- All analysis scripts available (release: December 2025)
- Random seeds fixed for reproducibility
- Version control via Git

---

## 6. Data Availability

**Public datasets used:**
- SPARC: http://astroweb.cwru.edu/SPARC/
- NANOGrav 15yr: https://data.nanograv.org/
- IPTA DR2: https://www.ipta4gw.org/

**Processed data:**
- Available upon reasonable request
- Will be released with code (December 2025)

---

## 7. Code Availability

**Planned release (December 2025):**
```
Repository: https://github.com/3D3DT-Laboratory/tri-temporal-theory

Includes:
- Full analysis pipeline
- Jupyter tutorials
- Processed datasets
- Figure generation scripts
```

**Early access:**
- Contact authors for pre-release code
- Available for independent validation efforts

---

**For questions about methodology:**
- Open GitHub Issue
- See main paper (DOI: 10.5281/zenodo.17516365)
- Contact authors

---

*Last updated: November 2025*
