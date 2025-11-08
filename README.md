# Tri‑Temporal Theory — v2.1 (Research Package)

This package provides a clean, modular structure to:
- load **SPARC** rotation‑curve data (from `Rotmod_LTG.zip`),
- compute the **Radial Acceleration Relation (RAR)** with a MOND‑like fit,
- compare with simple baselines (ΛCDM / MOND stubs),
- export **JSON / CSV** results and **PNG** plots for GitHub review.

> **Copyright**
> TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab. All rights reserved.

## Quickstart

```bash
# 1) Place your SPARC archive:
#    ./data/Rotmod_LTG.zip   (recommended path)
#
# 2) Create a virtualenv and install
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e .

# 3) Run RAR on real SPARC data
python -m src.analysis.rar_relation --rotmod data/Rotmod_LTG.zip

# 4) Outputs
#    outputs/rar/rar_data.csv
#    outputs/rar/rar_summary.json
#    outputs/rar/rar_plot.png
```

If your archive lives elsewhere, edit `configs/sparc_local.yaml` or pass
`--rotmod /path/to/Rotmod_LTG.zip`.

## Project layout

```
tri-temporal-theory_v2_1/
├─ src/
│  ├─ data_io.py
│  ├─ models/
│  │   ├─ ttn_core.py
│  │   └─ baselines.py
│  ├─ analysis/
│  │   ├─ rar_relation.py
│  │   └─ fft_rar.py
│  ├─ train_eval.py
│  ├─ utils.py
│  └─ cli.py
├─ configs/
│  ├─ default.yaml
│  └─ sparc_local.yaml
├─ outputs/
│  └─ figures/
├─ pyproject.toml
└─ README.md
```
### 🔍 RAR Fit and Interpretation

The `rar_fit_CORRECTED.py` module performs a full weighted analysis of the  
Radial Acceleration Relation (RAR) using the physical 3D+3D formulation  
(g = g_bar × [1 + (g_bar/g0)^α]).

📘 See full theoretical discussion in [`docs/RAR_EXPLANATION.md`](docs/RAR_EXPLANATION.md)

## Notes

- `src/data_io.py` provides **robust loaders** for `Rotmod_LTG.zip`.
  It looks for CSV/TSV inside the zip with columns like `R_kpc` and `V_kms`.
  If the archive is the canonical SPARC set with per‑galaxy files, the loader will
  try common patterns and raise a **clear error** if it can’t auto‑detect them.
- `src/analysis/rar_relation.py` saves CSV/JSON/PNG so that results can be pushed to GitHub.
- The **TTN model** is kept as a stub (`ttn_core.py`) with clear extension points.
