"""
SPARC galaxy rotation curve analysis for 3D+3D theory validation.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy import signal
import pandas as pd
from pathlib import Path

from .utils import (
    newtonian_velocity, 
    velocity_3D3D,
    chi_squared,
    reduced_chi_squared,
    read_sparc_galaxy,
    baryonic_mass_enclosed,
    LAMBDA_B,
    G,
    setup_logger
)

logger = setup_logger(__name__)


class SPARCGalaxy:
    """
    Container for SPARC galaxy data and analysis.
    """
    
    def __init__(self, name, data_file=None):
        """
        Initialize galaxy object.
        
        Parameters:
        -----------
        name : str
            Galaxy name (e.g., 'NGC3198')
        data_file : str, optional
            Path to data file
        """
        self.name = name
        self.data_file = data_file
        
        # Data arrays
        self.radius = None
        self.vobs = None
        self.verr = None
        self.vgas = None
        self.vdisk = None
        self.vbul = None
        
        # Derived quantities
        self.mass_profile = None
        self.mass_total = None
        self.residuals = None
        
        # Analysis results
        self.lambda_detected = None
        self.amplitude = None
        self.phase = None
        self.detection_significance = None
        
        # Load data if file provided
        if data_file:
            self.load_data(data_file)
    
    def load_data(self, filename):
        """Load galaxy data from file."""
        logger.info(f"Loading {self.name} from {filename}")
        
        data = read_sparc_galaxy(filename)
        
        self.radius = data['radius']
        self.vobs = data['vobs']
        self.verr = data['verr']
        self.vgas = data['vgas']
        self.vdisk = data['vdisk']
        self.vbul = data['vbul']
        
        # Compute mass profile
        self.compute_mass_profile()
        
        logger.info(f"Loaded {len(self.radius)} data points")
    
    def compute_mass_profile(self):
        """Compute enclosed baryonic mass profile."""
        self.mass_profile = baryonic_mass_enclosed(
            self.radius, self.vgas, self.vdisk, self.vbul
        )
        self.mass_total = self.mass_profile[-1]
        
        logger.info(f"Total baryonic mass: {self.mass_total:.2e} M_sun")
    
    def compute_residuals(self):
        """Compute velocity residuals (observed - Newtonian)."""
        v_newton = newtonian_velocity(self.radius, self.mass_profile)
        self.residuals = self.vobs - v_newton
        
        return self.residuals
    
    def fft_analysis(self, window='hann'):
        """
        Perform FFT analysis on residuals.
        
        Parameters:
        -----------
        window : str
            Window function ('hann', 'hamming', 'blackman', None)
        
        Returns:
        --------
        wavelengths : ndarray
            Wavelengths [kpc]
        power : ndarray
            Power spectrum
        """
        if self.residuals is None:
            self.compute_residuals()
        
        # Apply window
        if window:
            if window == 'hann':
                w = np.hanning(len(self.residuals))
            elif window == 'hamming':
                w = np.hamming(len(self.residuals))
            elif window == 'blackman':
                w = np.blackman(len(self.residuals))
            else:
                w = np.ones(len(self.residuals))
            
            residuals_windowed = self.residuals * w
        else:
            residuals_windowed = self.residuals
        
        # FFT
        fft_result = np.fft.fft(residuals_windowed)
        power = np.abs(fft_result)**2
        
        # Frequencies and wavelengths
        mean_spacing = np.mean(np.diff(self.radius))
        frequencies = np.fft.fftfreq(len(self.residuals), d=mean_spacing)
        
        # Positive frequencies only
        positive = frequencies > 0
        wavelengths = 1.0 / frequencies[positive]
        power = power[positive]
        
        return wavelengths, power
    
    def detect_breathing_scale(self, lambda_range=(3.0, 6.0), threshold_sigma=3.0):
        """
        Detect breathing scale λ_b in FFT power spectrum.
        
        Parameters:
        -----------
        lambda_range : tuple
            Wavelength search range [kpc]
        threshold_sigma : float
            Detection threshold in units of noise σ
        
        Returns:
        --------
        detected : bool
            Whether λ_b was detected
        lambda_b : float or None
            Detected wavelength [kpc]
        significance : float
            Detection significance [σ]
        """
        wavelengths, power = self.fft_analysis()
        
        # Restrict to search range
        mask = (wavelengths >= lambda_range[0]) & (wavelengths <= lambda_range[1])
        wavelengths_search = wavelengths[mask]
        power_search = power[mask]
        
        if len(power_search) == 0:
            logger.warning(f"{self.name}: No data in search range")
            return False, None, 0.0
        
        # Noise floor (from outside search range)
        noise_mask = (wavelengths < lambda_range[0]) | (wavelengths > lambda_range[1])
        if np.sum(noise_mask) > 10:
            noise_floor = np.median(power[noise_mask])
            noise_std = np.std(power[noise_mask])
        else:
            noise_floor = np.median(power)
            noise_std = np.std(power)
        
        # Find peak
        peak_idx = np.argmax(power_search)
        peak_power = power_search[peak_idx]
        peak_wavelength = wavelengths_search[peak_idx]
        
        # Significance
        significance = (peak_power - noise_floor) / noise_std
        
        # Detection criterion
        detected = significance > threshold_sigma
        
        if detected:
            self.lambda_detected = peak_wavelength
            self.detection_significance = significance
            logger.info(f"{self.name}: λ_b = {peak_wavelength:.2f} kpc ({significance:.1f}σ)")
        else:
            self.lambda_detected = None
            self.detection_significance = significance
            logger.info(f"{self.name}: No detection ({significance:.1f}σ)")
        
        return detected, peak_wavelength if detected else None, significance
    
    def fit_harmonic(self, lambda_b=LAMBDA_B, fit_lambda=False):
        """
        Fit harmonic oscillation to residuals.
        
        Parameters:
        -----------
        lambda_b : float
            Fixed wavelength [kpc] (if fit_lambda=False)
        fit_lambda : bool
            Whether to fit λ_b as free parameter
        
        Returns:
        --------
        params : dict
            Fitted parameters (amplitude, phase, lambda)
        """
        if self.residuals is None:
            self.compute_residuals()
        
        def harmonic_model(r, A, phi, lam=lambda_b):
            return A * np.sin(2*np.pi*r/lam + phi)
        
        # Initial guess
        p0 = [5.0, 0.0]  # amplitude, phase
        
        if fit_lambda:
            p0.append(lambda_b)
            popt, pcov = curve_fit(
                lambda r, A, phi, lam: harmonic_model(r, A, phi, lam),
                self.radius, self.residuals, p0=p0,
                sigma=self.verr, absolute_sigma=True,
                bounds=([0, -np.pi, 3.0], [20, np.pi, 6.0])
            )
            self.amplitude, self.phase, self.lambda_detected = popt
        else:
            popt, pcov = curve_fit(
                lambda r, A, phi: harmonic_model(r, A, phi, lambda_b),
                self.radius, self.residuals, p0=p0,
                sigma=self.verr, absolute_sigma=True
            )
            self.amplitude, self.phase = popt
            self.lambda_detected = lambda_b
        
        # Uncertainties
        perr = np.sqrt(np.diag(pcov))
        
        params = {
            'amplitude': self.amplitude,
            'amplitude_err': perr[0],
            'phase': self.phase,
            'phase_err': perr[1],
            'lambda': self.lambda_detected,
        }
        
        if fit_lambda:
            params['lambda_err'] = perr[2]
        
        logger.info(f"{self.name}: A = {self.amplitude:.2f} ± {perr[0]:.2f} km/s")
        
        return params
    
    def chi2_comparison(self, model='3D3D'):
        """
        Compute chi-squared for model.
        
        Parameters:
        -----------
        model : str
            'Newtonian', 'LCDM', or '3D3D'
        
        Returns:
        --------
        chi2 : float
        chi2_reduced : float
        """
        if model == 'Newtonian':
            v_pred = newtonian_velocity(self.radius, self.mass_profile)
            n_params = 1  # just mass normalization
            
        elif model == '3D3D':
            v_pred = velocity_3D3D(
                self.radius, self.mass_profile, self.mass_total
            )
            n_params = 3  # lambda_b, Q2, amplitude
            
        else:
            raise ValueError(f"Unknown model: {model}")
        
        chi2 = chi_squared(self.vobs, v_pred, self.verr)
        chi2_red = reduced_chi_squared(self.vobs, v_pred, self.verr, n_params)
        
        return chi2, chi2_red


def analyze_sparc_sample(data_dir, output_file='sparc_results.csv',
                         lambda_range=(3.0, 6.0), threshold=3.0):
    """
    Analyze full SPARC sample.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing SPARC galaxy files
    output_file : str
        Output CSV file for results
    lambda_range : tuple
        Wavelength search range
    threshold : float
        Detection threshold [σ]
    
    Returns:
    --------
    results : DataFrame
        Analysis results for all galaxies
    """
    data_dir = Path(data_dir)
    galaxy_files = list(data_dir.glob("*_rotmod.dat"))
    
    logger.info(f"Found {len(galaxy_files)} galaxy files")
    
    results_list = []
    
    for i, gal_file in enumerate(galaxy_files):
        galaxy_name = gal_file.stem.replace('_rotmod', '')
        
        try:
            galaxy = SPARCGalaxy(galaxy_name, gal_file)
            
            # FFT detection
            detected, lambda_b, significance = galaxy.detect_breathing_scale(
                lambda_range, threshold
            )
            
            # Harmonic fit
            if detected:
                params = galaxy.fit_harmonic(lambda_b, fit_lambda=False)
            else:
                params = {'amplitude': np.nan, 'phase': np.nan}
            
            # Chi-squared
            chi2_newton, chi2red_newton = galaxy.chi2_comparison('Newtonian')
            chi2_3d3d, chi2red_3d3d = galaxy.chi2_comparison('3D3D')
            
            results_list.append({
                'name': galaxy_name,
                'mass_total': galaxy.mass_total,
                'n_points': len(galaxy.radius),
                'detected': detected,
                'lambda_b': lambda_b,
                'significance': significance,
                'amplitude': params['amplitude'],
                'phase': params['phase'],
                'chi2_newton': chi2_newton,
                'chi2_3d3d': chi2_3d3d,
                'chi2red_newton': chi2red_newton,
                'chi2red_3d3d': chi2red_3d3d,
            })
            
            logger.info(f"[{i+1}/{len(galaxy_files)}] {galaxy_name}: Done")
            
        except Exception as e:
            logger.error(f"Error processing {galaxy_name}: {e}")
            continue
    
    # Convert to DataFrame
    results = pd.DataFrame(results_list)
    
    # Save to CSV
    results.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")
    
    # Summary statistics
    detection_rate = results['detected'].sum() / len(results) * 100
    mean_lambda = results[results['detected']]['lambda_b'].mean()
    std_lambda = results[results['detected']]['lambda_b'].std()
    
    logger.info(f"\nSummary:")
    logger.info(f"Detection rate: {detection_rate:.1f}%")
    logger.info(f"Mean λ_b: {mean_lambda:.2f} ± {std_lambda:.2f} kpc")
    
    return results
