"""
3D+3D Theory Analysis Package
==============================

Empirical validation of six-dimensional discrete spacetime.

Modules:
    sparc_analysis: Galaxy rotation curve analysis
    pulsar_timing: Pulsar timing array analysis
    model_comparison: Statistical model comparison
    statistical_tests: Bayesian inference and hypothesis testing
    plotting: Publication-quality figure generation
    utils: Utility functions and constants

Authors: Simone Calzighetti & Lucy (AI Collaborator)
License: MIT
DOI: 10.5281/zenodo.17516365
"""

__version__ = "1.0.0"
__author__ = "Simone Calzighetti & Lucy"
__email__ = "contact@3d3dt-lab.org"

from . import sparc_analysis
from . import pulsar_timing
from . import model_comparison
from . import statistical_tests
from . import plotting
from . import utils

# Key constants
LAMBDA_B = 4.30  # kpc - breathing scale
M_CRIT = 2.43e10  # M_sun - critical mass
ALPHA = 0.34  # mass-amplitude exponent
TAU_B = 28.4  # years - pulsar timing period

__all__ = [
    'sparc_analysis',
    'pulsar_timing',
    'model_comparison',
    'statistical_tests',
    'plotting',
    'utils',
    'LAMBDA_B',
    'M_CRIT',
    'ALPHA',
    'TAU_B',
]
