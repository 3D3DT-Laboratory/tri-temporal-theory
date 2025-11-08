# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
# All rights reserved.
# Unauthorized copying, modification, distribution prohibited without prior written consent.

"""
Wrapper semplice sopra fft_rar.py
Uso:
  python -m src.analysis.harmonic_analysis --rar-csv outputs/rar/rar_data.csv
"""

import argparse
from .fft_rar import run as fft_run

def main(rar_csv: str, outdir: str = "outputs/harmonics"):
    return fft_run(rar_csv, outdir, lambda_max_kpc=30.0, min_peak_snr=3.0, per_galaxy=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rar-csv", required=True)
    ap.add_argument("--outdir", default="outputs/harmonics")
    args = ap.parse_args()
    main(args.rar_csv, outdir=args.outdir)
