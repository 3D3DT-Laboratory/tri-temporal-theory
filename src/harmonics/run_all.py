# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
# All rights reserved.
# Unauthorized copying, modification, distribution prohibited without prior written consent.
# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
import json
from pathlib import Path
import numpy as np
from .io_sparc import load_sparc_zip
from .model_harmonics import HarmonicModel

DEFAULT_LAMBDAS = [0.87, 1.89, 4.30, 8.60, 11.7, 21.4]  # kpc

def main(rotmod_zip: str, outdir: str = "outputs/harmonics",
         lambdas_kpc=None):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    if lambdas_kpc is None:
        lambdas_kpc = DEFAULT_LAMBDAS

    galaxies = load_sparc_zip(rotmod_zip)
    model = HarmonicModel(lambdas_kpc)
    rows = []

    for i, g in enumerate(galaxies, 1):
        res = model.fit_single_galaxy(g)
        rows.append({
            "galaxy": g.name,
            "n_points": int(g.r.size),
            "Q": res["Q"],
            "chi2_base": res["chi2_base"],
            "chi2_fit": res["chi2_fit"],
            "improvement_pct": res["improvement_pct"]
        })
        if i % 25 == 0 or i == len(galaxies):
            print(f"[{i}/{len(galaxies)}] {g.name}  Δχ²%={res['improvement_pct']:.1f}")

    # summary
    imp = [r["improvement_pct"] for r in rows]
    summary = {
        "n_galaxies": len(rows),
        "lambdas_kpc": lambdas_kpc,
        "mean_improvement_pct": float(np.mean(imp)),
        "median_improvement_pct": float(np.median(imp)),
        "detected_Q>0.1_pct": float(100.0*np.mean([
            any(qi>0.1 for qi in r["Q"]) for r in rows])),
        "strong_Q>0.3_pct": float(100.0*np.mean([
            any(qi>0.3 for qi in r["Q"]) for r in rows])),
    }

    (out/"harmonics_results.json").write_text(json.dumps(
        {"summary": summary, "per_galaxy": rows}, indent=2))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rotmod", required=True, help="path to Rotmod_LTG.zip")
    ap.add_argument("--outdir", default="outputs/harmonics")
    ap.add_argument("--lambdas", nargs="*", type=float,
                    help="override λ list in kpc (e.g. --lambdas 0.87 1.89 4.30)")
    args = ap.parse_args()
    main(args.rotmod, args.outdir, args.lambdas)
