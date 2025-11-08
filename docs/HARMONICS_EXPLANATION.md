# Harmonic Analysis of RAR / Rotation Curves

Questo modulo estrae **scale armoniche** nei residui delle curve (RAR o V(r)).
Nel framework 3D+3D ci si attende una scala dominante **λ_b ≈ 4.3 kpc** con
sotto/sovra-armoniche a rapporti semplici.

## Pipeline

1. Input: `outputs/rar/rar_data.csv` (almeno `gbar`, `gobs`; idealmente `galaxy`, `r_kpc`).
2. Residuo: `r = log10(g_obs) - log10(g_bar)` (se non già fornito).
3. FFT per galassia (se `galaxy` presente) e FFT globale di controllo.
4. Peak picking con SNR (MAD-based).
5. Output:
   - `outputs/harmonics/harmonic_summary.csv`
   - `outputs/harmonics/fft_<GALAXY>.png`
   - `outputs/harmonics/fft_global.png`
   - `outputs/harmonics/diagnostic_log.txt`

> Se `r_kpc` è quasi uniforme per galassia, stimiamo una λ (kpc) dal passo medio.

## Esecuzione

```bash
# Dopo aver generato outputs/rar/rar_data.csv
python -m src.analysis.fft_rar --input outputs/rar/rar_data.csv --per-galaxy
# Opzioni:
python -m src.analysis.fft_rar \
  --input outputs/rar/rar_data.csv \
  --outdir outputs/harmonics \
  --lambda-max 30.0 \
  --min-peak-snr 3.0 \
  --per-galaxy
