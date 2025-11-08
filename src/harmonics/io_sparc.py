# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
# All rights reserved.
# Unauthorized copying, modification, distribution prohibited without prior written consent.
# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
from pathlib import Path
import zipfile
import numpy as np

class Galaxy:
    def __init__(self, name, r_kpc, v_obs, v_err, v_gas, v_disk, v_bul):
        self.name = name
        self.r = np.asarray(r_kpc, dtype=float)
        self.v_obs = np.asarray(v_obs, dtype=float)
        self.v_err = np.asarray(v_err, dtype=float)
        self.v_gas = np.asarray(v_gas, dtype=float)
        self.v_disk = np.asarray(v_disk, dtype=float)
        self.v_bul = np.asarray(v_bul, dtype=float)
        self.v_bary = np.sqrt(self.v_gas**2 + self.v_disk**2 + self.v_bul**2)

def _parse_rotmod_dat(text: str):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"): 
            continue
        parts = line.split()
        if len(parts) < 6:    # R, Vobs, eV, Vgas, Vdisk, Vbul
            continue
        rows.append([float(x) for x in parts[:6]])
    if not rows:
        return None
    arr = np.array(rows)
    return dict(
        r_kpc=arr[:,0], v_obs=arr[:,1], v_err=arr[:,2],
        v_gas=arr[:,3], v_disk=arr[:,4], v_bul=arr[:,5]
    )

def load_sparc_zip(zip_path: str):
    zpath = Path(zip_path)
    if not zpath.exists():
        raise FileNotFoundError(f"SPARC archive not found: {zip_path}")
    galaxies = []
    with zipfile.ZipFile(zpath, "r") as zf:
        dat_names = [n for n in zf.namelist() if n.lower().endswith("_rotmod.dat")]
        if not dat_names:
            raise RuntimeError("Nessun *_rotmod.dat nello ZIP SPARC.")
        for name in sorted(dat_names):
            with zf.open(name) as f:
                txt = f.read().decode("utf-8", errors="ignore")
            parsed = _parse_rotmod_dat(txt)
            if parsed is None: 
                continue
            gal_name = Path(name).stem.replace("_rotmod","")
            g = Galaxy(gal_name, **parsed)
            if g.r.size >= 5:
                galaxies.append(g)
    return galaxies
