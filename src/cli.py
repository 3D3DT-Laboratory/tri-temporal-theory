# ===========================================================
#  TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
#  Unauthorized copying of this file, via any medium, is strictly prohibited.
#  Proprietary and confidential. All rights reserved.
# ===========================================================

from __future__ import annotations
import argparse
from src.analysis.rar_relation import compute_rar

def main():
    ap = argparse.ArgumentParser(description="Tri‑Temporal Theory CLI")
    ap.add_argument("--rotmod", default="data/Rotmod_LTG.zip")
    ap.add_argument("--outdir", default="outputs/rar")
    args = ap.parse_args()
    compute_rar(args.rotmod, args.outdir)

if __name__ == "__main__":
    main()
