# RAR Analysis Guide

## Quick Start

### Prerequisites
```bash
pip install numpy scipy pandas matplotlib
```

### Basic Usage
```bash
# Fit all models (ΛCDM, MOND, 3D+3D)
python src/models/analysis/rar_fit_logspace.py \
    --rar-csv data/processed/rar_data.csv \
    --outdir outputs/rar \
    --sigma-int 0.0
```

### Expected Output
```
Model         χ²_red    R²_w     RMS
---------------------------------------------------------------
ΛCDM          2.27   0.871  0.210
MOND          2.65   0.849  0.222
3D+3D         2.44   0.861  0.217
---------------------------------------------------------------
```

## 3D+3D Model

### Formula
The 3D+3D model uses power-law interpolation in log-space:
```
g_obs = g₀^(1-γ) × g_bar^γ
```

Where:
- `g₀ = 1.2×10⁻¹⁰ m/s²` (characteristic scale, fixed)
- `γ_RAR` (phenomenological exponent, fitted)

### Physical Interpretation

**Low acceleration regime** (g_bar << g₀):
```
g_obs ≈ g₀^0.34 × g_bar^0.66
→ Sublinear boost (dark matter-like)
```

**High acceleration regime** (g_bar >> g₀):
```
g_obs ≈ g_bar
→ Newtonian (no boost)
```

## Advanced Options

### Free g₀ fit
```bash
python src/models/analysis/rar_fit_logspace.py \
    --rar-csv data/processed/rar_data.csv \
    --outdir outputs/rar_free_g0 \
    --sigma-int 0.0 \
    --fit-g0
```

### Custom intrinsic scatter
```bash
# Only use if you know your dataset needs it
python src/models/analysis/rar_fit_logspace.py \
    --rar-csv data/processed/rar_data.csv \
    --outdir outputs/rar_custom \
    --sigma-int 0.06  # dex
```

## Diagnostics

### Data Quality Check
```bash
python src/models/analysis/rar_diagnostics.py \
    --rar-csv data/processed/rar_data.csv \
    --outdir outputs/diagnostics
```

This will check:
- Column integrity (g_bar, g_obs present)
- Physical consistency (g = V²/R)
- RAR shape (DM boost at low g, Newtonian at high g)
- MOND fit sanity check

### Detailed Analysis
```bash
python src/utils/diagnose_rar_detailed.py \
    data/processed/rar_data.csv
```

## Interpreting Results

### Good Fit Indicators
- χ²_red ≈ 2.0-2.5 (acceptable given systematics)
- R²_w > 0.85 (high correlation)
- γ_RAR ∈ [0.5, 0.8] (physical range)
- Residuals scatter randomly around zero

### Red Flags
- χ²_red > 5 → Check data quality
- γ_RAR < 0 → Wrong formula or swapped columns
- MOND a₀ < 10⁻¹¹ → Dataset issues

## Comparison with Literature

### McGaugh+ 2016 (SPARC RAR)
- Dataset: 153 galaxies, 2693 points
- MOND: χ²_red ~ 1.0-1.5, a₀ = 1.2×10⁻¹⁰ m/s²
- Scatter: 0.11 ± 0.02 dex (observed)

### Our Analysis
- Dataset: 175 galaxies, 3391 points (extended SPARC)
- 3D+3D: χ²_red = 2.44, g₀ = 1.2×10⁻¹⁰ m/s²
- γ_RAR = 0.66 ± 0.04

**Note**: Higher χ² expected due to:
1. Larger dataset (more galaxies)
2. No distance/inclination cuts
3. Conservative error estimates

## Troubleshooting

### Problem: χ² too high (>10)
**Solution**: Check σ_int parameter. For this dataset, use 0.0.

### Problem: γ negative
**Solution**: Verify you're using `rar_fit_logspace.py` (power-law formula), not older additive boost version.

### Problem: MOND a₀ very low (<10⁻¹²)
**Solution**: Check that g_bar and g_obs aren't swapped in CSV.

## Citation

If you use this code, please cite:
```bibtex
@article{Calzighetti2025,
  title={3D+3D Spacetime: Empirical Validation via SPARC Galaxies},
  author={Calzighetti, Simone},
  journal={arXiv preprint},
  year={2025}
}
```
