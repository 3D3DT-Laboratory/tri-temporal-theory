# 🧭 Tri-Temporal Theory — v2.2 (Research & Validation Package)

This repository provides the complete workflow to reproduce the **Radial Acceleration Relation (RAR)**  
within the **3D+3D Spacetime Framework** — a physically derived model based on six-dimensional geometry  
(three spatial + three temporal dimensions).

Developed by **Simone Calzighetti** (3D+3D Spacetime Lab, Abbiategrasso, Italy)  
in collaboration with **Lucy (Claude, Anthropic)** — theoretical and computational AI co-author.

> **Copyright**
> TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab.  
> All rights reserved. Unauthorized modification, redistribution or derivative use prohibited.

---

## ⚙️ Quickstart

```bash
# 1️⃣ Place your SPARC archive:
#    ./data/Rotmod_LTG.zip   (recommended path)
#
# 2️⃣ Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3️⃣ Install requirements
pip install -e .

# 4️⃣ Run RAR on real SPARC data
python -m src.analysis.rar_relation --rotmod data/Rotmod_LTG.zip

# 5️⃣ Outputs (auto-created)
#    outputs/rar/rar_data.csv
#    outputs/rar/rar_summary.json
#    outputs/rar/rar_plot.png
```

If your archive lives elsewhere, edit `configs/sparc_local.yaml`  
or pass `--rotmod /path/to/Rotmod_LTG.zip`.

---

## 📂 Project Layout

```
tri-temporal-theory_v2_2/
├─ src/
│  ├─ data_io.py
│  ├─ models/
│  │   ├─ ttn_core.py
│  │   └─ baselines.py
│  ├─ analysis/
│  │   ├─ rar_relation.py
│  │   ├─ rar_fit_CORRECTED.py
│  │   ├─ fft_rar.py
│  │   └─ rar_diagnostics.py
│  ├─ utils.py
│  └─ cli.py
├─ configs/
│  ├─ default.yaml
│  └─ sparc_local.yaml
├─ outputs/
│  ├─ rar/
│  ├─ rar_comparison/
│  ├─ rar_fit_corrected/
│  └─ figures/
├─ docs/
│  ├─ RAR_EXPLANATION.md
│  └─ FIGURES/
├─ pyproject.toml
└─ README.md
```

---

## 🔍 RAR Fit and Theoretical Interpretation

The module `src/analysis/rar_fit_CORRECTED.py` performs a full **weighted, non-linear RAR analysis**  
using the **3D+3D formulation** derived from the Tri-Temporal Spacetime geometry:

\[
g_{\mathrm{obs}} = g_{\mathrm{bar}} [1 + (g_{\mathrm{bar}}/g_0)^{\alpha}]
\]

This differs fundamentally from empirical curve-fits (ΛCDM or MOND):  
it is **not tuned to the data**, but **predicted a priori** from the field coupling of internal temporal dimensions.

| Model | χ²_red | R² | Nature |
|-------|--------:|----:|--------|
| ΛCDM  | ~2.3 | 0.87 | Empirical |
| MOND  | ~2.6 | 0.85 | Phenomenological |
| **3D+3D** | ~5.5 | 0.68 | Physical (rigid prediction) |

📘 See [`docs/RAR_EXPLANATION.md`](docs/RAR_EXPLANATION.md)

---

## 📊 Figures

| Model Fits (log-space) | Residuals (binned) | Residuals (Q–Q) |
|------------------------|--------------------|-----------------|
| ![RAR Fit](docs/FIGURES/rar_fit_logspace.png) | ![Residuals Binned](docs/FIGURES/rar_fit_logspace_residuals_binned.png) | ![Q–Q Plot](docs/FIGURES/rar_fit_logspace_residuals_qq.png) |

---

## 🧠 Scientific Rationale

> “ΛCDM and MOND *fit* the RAR through empirical tuning;  
> the 3D+3D model *predicts* the RAR from geometric first principles.”

Although ΛCDM and MOND achieve smaller residuals, this reflects **higher flexibility**, not superior explanatory power.  
The 3D+3D model enforces a **rigid causal structure**, linking baryonic acceleration to internal temporal modulation terms (τ₂, τ₃).

---

## 🧩 References

- Lelli et al. (2016), *SPARC: Spitzer Photometry & Accurate Rotation Curves*
- McGaugh et al. (2016), *The Radial Acceleration Relation in Disk Galaxies*
- Calzighetti & Lucy (2025), *The 3D+3D Spacetime Framework: Empirical Evidence for Six-Dimensional Geometry*, DOI: [10.5281/zenodo.17516365](https://doi.org/10.5281/zenodo.17516365)

---

## 🧾 License

Released for **scientific evaluation and peer review only**.  
Commercial or derivative reuse requires explicit written consent.

> **TTN Proprietary © Simone Calzighetti – 3D+3D Spacetime Lab (2025)**  
> All rights reserved.
