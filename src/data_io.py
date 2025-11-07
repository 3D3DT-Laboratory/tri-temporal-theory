# ===========================================================
#  TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
#  Unauthorized copying of this file, via any medium, is strictly prohibited.
#  Proprietary and confidential. All rights reserved.
# ===========================================================

from __future__ import annotations
import io, zipfile, re, pathlib
import numpy as np
import pandas as pd

RE_GUESS = re.compile(r"(R[_\s-]*kpc|R\s*\(kpc\))", re.I)
RE_V     = re.compile(r"(V[_\s-]*obs|V\s*\(km/s\)|V_kms|Vrot)", re.I)

def _find_tables_in_zip(zf: zipfile.ZipFile):
    \"\"\"Yield candidate (name, DataFrame) tables within the zip that look like rotation curves.\"\"\"
    for name in zf.namelist():
        lname = name.lower()
        if lname.endswith((".csv", ".tsv", ".txt")) and not lname.endswith(("readme.txt",)):
            try:
                with zf.open(name) as fh:
                    raw = fh.read()
                # try csv, then tsv
                for sep in [",", "\t", r"\s+"]:
                    try:
                        df = pd.read_csv(io.BytesIO(raw), sep=sep if sep != r"\s+" else None, engine="python")
                        if df.shape[1] >= 2:
                            yield name, df
                            break
                    except Exception:
                        continue
            except Exception:
                continue

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame | None:
    cols = {c: str(c) for c in df.columns}
    # guess R_kpc
    R_col = None
    V_col = None
    for c in cols:
        if RE_GUESS.search(cols[c]):
            R_col = c; break
    for c in cols:
        if RE_V.search(cols[c]):
            V_col = c; break
    if R_col is None or V_col is None:
        return None
    out = pd.DataFrame({
        "R_kpc": pd.to_numeric(df[R_col], errors="coerce"),
        "V_kms": pd.to_numeric(df[V_col], errors="coerce"),
    }).dropna()
    # filter positive radii
    out = out[out["R_kpc"] > 0]
    return out if len(out) >= 3 else None

def load_rotmod_zip(path: str | pathlib.Path):
    \"\"\"
    Load SPARC Rotmod archive. Returns dict: galaxy_name -> DataFrame(R_kpc, V_kms)
    Heuristics: scans for CSV/TSV/TXT in the zip and standardizes columns.
    \"\"\"
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"SPARC archive not found: {p}")
    out = {}
    with zipfile.ZipFile(p, "r") as zf:
        for name, df in _find_tables_in_zip(zf):
            std = _standardize_columns(df)
            if std is None:
                continue
            # guess galaxy name from path
            gname = pathlib.Path(name).stem
            out[gname] = std
    if not out:
        raise RuntimeError(
            "No usable rotation-curve tables found.\\n"
            "Expecting files with columns like 'R_kpc' and 'V_kms' (or V_obs, V(km/s)).\\n"
            "Please inspect your Rotmod_LTG.zip content."
        )
    return out

def rotcurves_to_dataframe(curves: dict) -> pd.DataFrame:
    rows = []
    for g, df in curves.items():
        tmp = df.copy()
        tmp["galaxy_id"] = g
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)
