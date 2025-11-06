# Repository Structure

This document explains the organization of the tri-temporal-theory repository.

## Folder Structure

```
tri-temporal-theory/
│
├── README.md                 # Main project description
├── LICENSE                   # MIT License
├── CITATION.cff              # Citation information for academic use
├── requirements.txt          # Python dependencies
├── .gitignore               # Files to ignore in git
│
├── docs/                    # Documentation
│   ├── theory_summary.md   # Conceptual overview of 3D+3D theory
│   ├── derivation.md       # Mathematical derivation details
│   └── analysis_guide.md   # Guide to running analyses
│
├── data/                    # Data files
│   ├── README.md           # Description of datasets
│   ├── sparc/              # SPARC galaxy data
│   ├── nanoGrav/           # NANOGrav pulsar data
│   └── processed/          # Processed/derived data
│
├── src/                     # Source code
│   ├── __init__.py
│   ├── sparc_analysis.py   # SPARC rotation curve analysis
│   ├── pulsar_timing.py    # Pulsar timing analysis
│   ├── mass_correlation.py # Mass-amplitude correlation
│   ├── loo_cv.py          # Leave-one-out cross-validation
│   └── utils.py           # Utility functions
│
├── notebooks/               # Jupyter notebooks
│   ├── 01_sparc_overview.ipynb
│   ├── 02_mass_correlation.ipynb
│   ├── 03_pulsar_analysis.ipynb
│   └── 04_model_comparison.ipynb
│
├── figures/                 # Generated figures
│   └── README.md
│
└── tests/                   # Unit tests
    ├── test_sparc.py
    ├── test_pulsar.py
    └── test_correlation.py
```

## Current Status

**✅ Available:**
- Repository structure
- Documentation framework
- Requirements specification

**🚧 In Development:**
- Analysis scripts
- Data processing pipelines
- Jupyter notebooks
- Unit tests

**📧 Request Access:**
For immediate access to pre-release code or data, please contact the authors.

## Future Additions

1. **Data Files** (after publication approval)
   - Processed SPARC rotation curves
   - Pulsar timing residuals
   - Model fit results

2. **Analysis Scripts** (in development)
   - Complete SPARC analysis pipeline
   - FFT and harmonic detection
   - Statistical validation tools

3. **Notebooks** (planned)
   - Step-by-step tutorials
   - Reproducible analysis workflows
   - Interactive visualizations

## Contributing

See main README.md for contribution guidelines.
