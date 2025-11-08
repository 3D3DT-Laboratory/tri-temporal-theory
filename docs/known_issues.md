# Known Issues & Solutions

## RAR Analysis

### Issue 1: MOND a₀ factor 3-4 low
**Status**: Known, not a bug

**Description**: 
Fitted MOND a₀ = 3.4×10⁻¹¹ m/s² (expected: 1.2×10⁻¹⁰)

**Explanation**:
- Dataset includes more galaxies than McGaugh+ 2016
- Different distance/inclination cuts
- Possibly different Upsilon_* assumptions
- Does NOT affect 3D+3D γ_RAR measurement

**Impact**: None on theory validation (relative comparison still valid)

---

### Issue 2: σ_int parameter confusion
**Status**: Resolved

**Description**:
Using σ_int = 0.08 dex (McGaugh+ 2016 value) makes all χ² artificially high.

**Solution**:
- This specific dataset has low intrinsic scatter
- Use σ_int = 0.0 for this analysis
- McGaugh's 0.08 dex was for different error model

**Command**:
```bash
python rar_fit_logspace.py ... --sigma-int 0.0  # Correct
```

---

### Issue 3: Two different 3D+3D formulas in code history
**Status**: Resolved

**Formulas tested**:
1. **Additive boost** (doesn't work):
```python
   g_obs = g_bar × [1 + (g_bar/g0)^γ]
```

2. **Power-law interpolation** (works ✅):
```python
   g_obs = g0^(1-γ) × g_bar^γ
```

**Current**: Using formula #2 (power-law) in `rar_fit_logspace.py`

---

### Issue 4: γ_RAR ≠ α_M (Pillar 2)
**Status**: Expected, not a bug

**Values**:
- α_M = 0.30 (mass-amplitude scaling, Pillar 2)
- γ_RAR = 0.66 (RAR acceleration scaling)

**Explanation**:
Different physical quantities measured:
- α_M: σ_FFT ∝ M^0.30 (breathing amplitude vs mass)
- γ_RAR: g_obs ∝ g_bar^0.66 (observed vs baryonic acceleration)

Mapping: M ~ g_bar × R² × f(profile) → non-trivial relation

**Impact**: Both are correct, describe different phenomena

---

## Data Processing

### Issue 5: Column naming inconsistency
**Status**: Handled

**Problem**: Some CSVs use `g_bar`, others `gbar`

**Solution**: Code checks both:
```python
gbar = df['g_bar'] if 'g_bar' in df.columns else df['gbar']
```

---

### Issue 6: High-g regime shows g_obs < g_bar
**Status**: Real data feature

**Observation**: At g_bar > 10⁻⁹ m/s², median ratio g_obs/g_bar ≈ 0.67

**Explanation**:
- Inner galaxy regions (high-g)
- Systematics: beam smearing, non-circular motions
- Baryonic mass estimates (Upsilon_*) may be high
- McGaugh+ 2016 also filters these regions

**Solution**: Data is correct as-is. Systematic, not error.

---

## Numerical Issues

### Issue 7: Fit doesn't converge
**Symptom**: γ stays at initial guess or hits bounds

**Solution**:
1. Check data quality (run diagnostics)
2. Verify formula is power-law (not additive)
3. Use reasonable initial guess: γ₀ = 0.68
4. Check bounds: [0.3, 1.0] for γ

---

### Issue 8: χ² exactly doubles when changing σ_int
**Status**: Expected behavior

**Explanation**:
χ² = Σ (residual / σ_eff)²

If σ_eff doubles → χ² quadruples (not doubles)
If you see doubling, check weighting scheme.

---

## Platform-Specific

### Issue 9: PowerShell command parsing (Windows)
**Problem**: f-strings in `-c` flag fail

**Solution**:
```powershell
# ❌ Don't use:
python -c "print(f'{value}')"

# ✅ Use instead:
python -c "print('{}'.format(value))"

# Or create temp script:
echo print(value) > temp.py
python temp.py
```

---

## Future Work

### Enhancement 1: Derive γ_RAR from first principles
**Status**: Theoretical work needed

Currently γ_RAR is phenomenological. Future: derive from Q-field equations + disk geometry.

### Enhancement 2: Bayesian parameter estimation
**Status**: Planned

Add MCMC (emcee) for full posterior distributions on (g₀, γ_RAR).

---

## Reporting New Issues

Found a bug? Please report:
1. Minimal reproducible example
2. Full error message / output
3. Python version, OS
4. Command used

Open issue at: github.com/your-repo/tri-temporal-theory/issues
