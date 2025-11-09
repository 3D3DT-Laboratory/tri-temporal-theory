# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
# All rights reserved. Unauthorized copying, modification, or distribution is prohibited.

"""
pytest configuration and shared fixtures for tri-temporal-theory tests.

This file provides reusable test data and configuration for the entire test suite.
Fixtures defined here are automatically available to all test files.

Author: Simone Calzighetti & Lucy (Claude, Anthropic)
Date: November 2025
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


# ============================================================================
# FILE PATHS
# ============================================================================

@pytest.fixture(scope="session")
def project_root():
    """Return project root directory"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def data_dir(project_root):
    """Return data directory path"""
    return project_root / "data"


@pytest.fixture(scope="session")
def output_dir(project_root):
    """Return outputs directory path"""
    return project_root / "outputs"


# ============================================================================
# SYNTHETIC TEST DATA
# ============================================================================

@pytest.fixture
def sample_rar_data():
    """
    Generate synthetic RAR data for testing.
    
    Returns:
        pd.DataFrame with columns: galaxy, r_kpc, gbar, gobs, e_gobs
        
    Characteristics:
        - 100 data points
        - Covers 4 orders of magnitude in gbar
        - Realistic uncertainties (~10%)
        - Perfect 3D+3D relation (for validation)
    """
    np.random.seed(42)  # Reproducible
    
    # Generate gbar spanning typical range
    log_gbar = np.linspace(-12, -9, 100)  # log10(m/s²)
    gbar = 10**log_gbar
    
    # Perfect 3D+3D relation
    gamma = 0.66
    g0 = 1.2e-10
    gobs_true = gbar * (1 + gamma * np.exp(-gbar / g0))
    
    # Add realistic noise
    relative_error = 0.1  # 10%
    gobs_noise = gobs_true * (1 + np.random.normal(0, relative_error, 100))
    e_gobs = gobs_true * relative_error
    
    # Create DataFrame
    data = pd.DataFrame({
        'galaxy': 'TEST_GALAXY',
        'r_kpc': np.linspace(1, 30, 100),
        'gbar': gbar,
        'gobs': gobs_noise,
        'e_gobs': e_gobs
    })
    
    return data


@pytest.fixture
def synthetic_rotation_curve():
    """
    Generate synthetic galaxy rotation curve.
    
    Returns:
        dict with keys: r (kpc), v_obs (km/s), v_err (km/s)
        
    Characteristics:
        - Realistic flat rotation curve
        - 100 radial points from 0.5 to 30 kpc
        - 5% velocity uncertainties
    """
    np.random.seed(42)
    
    r = np.linspace(0.5, 30, 100)  # kpc
    
    # Flat rotation curve with smooth rise
    v_flat = 200  # km/s
    r_scale = 5   # kpc
    v_true = v_flat * np.sqrt(1 - np.exp(-r/r_scale))
    
    # Add noise
    v_err = 0.05 * v_flat  # 5% uncertainty
    v_obs = v_true + np.random.normal(0, v_err, 100)
    
    return {
        'r': r,
        'v_obs': v_obs,
        'v_err': np.full(100, v_err)
    }


@pytest.fixture
def synthetic_galaxy_sample():
    """
    Generate sample of multiple synthetic galaxies.
    
    Returns:
        pd.DataFrame with multiple galaxies, varying properties
        
    Characteristics:
        - 5 galaxies (dwarf to massive)
        - Different masses, sizes, velocities
        - Useful for batch processing tests
    """
    np.random.seed(42)
    
    galaxies = []
    
    galaxy_params = [
        {'name': 'DWARF1', 'v_flat': 50, 'r_max': 10, 'mass': 1e9},
        {'name': 'SPIRAL1', 'v_flat': 150, 'r_max': 25, 'mass': 5e10},
        {'name': 'SPIRAL2', 'v_flat': 200, 'r_max': 30, 'mass': 1e11},
        {'name': 'MASSIVE1', 'v_flat': 300, 'r_max': 40, 'mass': 5e11},
        {'name': 'MASSIVE2', 'v_flat': 350, 'r_max': 50, 'mass': 1e12},
    ]
    
    for params in galaxy_params:
        n_points = 50
        r = np.linspace(0.5, params['r_max'], n_points)
        v_true = params['v_flat'] * np.sqrt(1 - np.exp(-r/5))
        v_obs = v_true + np.random.normal(0, 0.05*params['v_flat'], n_points)
        
        for i in range(n_points):
            galaxies.append({
                'galaxy': params['name'],
                'r_kpc': r[i],
                'v_obs': v_obs[i],
                'v_err': 0.05 * params['v_flat'],
                'mass': params['mass']
            })
    
    return pd.DataFrame(galaxies)


# ============================================================================
# KNOWN PARAMETERS
# ============================================================================

@pytest.fixture(scope="session")
def known_3d3d_parameters():
    """
    Known parameter values for 3D+3D model validation.
    
    Returns:
        dict with validated parameter values
        
    Source:
        Calzighetti & Lucy (2025), Zenodo 10.5281/zenodo.17516365
    """
    return {
        'gamma_rar': 0.66,
        'gamma_rar_err': 0.04,
        'g0': 1.2e-10,  # m/s²
        'g0_err': 0.1e-10,
        'chi2_red_3d3d': 2.44,
        'r2_weighted_3d3d': 0.861,
        'lambda_b': 4.30,  # kpc (fundamental harmonic)
        'lambda_b_err': 0.15,
        'alpha_m': 0.30,  # Mass-amplitude scaling
        'alpha_m_err': 0.06,
        'M_crit': 2.43e10,  # Solar masses
    }


@pytest.fixture(scope="session")
def known_mond_parameters():
    """Known MOND parameters for comparison"""
    return {
        'a0': 3.42e-11,  # m/s²
        'chi2_red_mond': 2.65,
        'r2_weighted_mond': 0.849,
    }


@pytest.fixture(scope="session")
def known_lcdm_parameters():
    """Known ΛCDM parameters for comparison"""
    return {
        'B': 0.68,  # Boost factor
        'chi2_red_lcdm': 2.27,
        'r2_weighted_lcdm': 0.871,
    }


@pytest.fixture(scope="session")
def six_harmonic_scales():
    """
    The six characteristic wavelengths predicted by 3D+3D theory.
    
    Returns:
        dict with scale names and wavelengths (kpc)
    """
    return {
        'lambda_0': 0.87,
        'lambda_1': 1.89,
        'lambda_2': 4.30,  # Fundamental
        'lambda_3': 6.51,
        'lambda_4': 11.7,
        'lambda_5': 21.4,
    }


# ============================================================================
# REAL DATA (if available)
# ============================================================================

@pytest.fixture(scope="session")
def sparc_rar_data(data_dir):
    """
    Load real SPARC RAR data if available.
    
    Returns:
        pd.DataFrame or None if file not found
        
    Note:
        Tests should handle None gracefully (skip if data unavailable)
    """
    rar_file = data_dir / "processed" / "rar_data.csv"
    
    if rar_file.exists():
        try:
            df = pd.read_csv(rar_file)
            # Validate columns
            required = ['galaxy', 'r_kpc', 'gbar', 'gobs', 'e_gobs']
            if all(col in df.columns for col in required):
                return df
        except Exception as e:
            pytest.skip(f"Could not load SPARC data: {e}")
    
    return None


# ============================================================================
# UTILITY FIXTURES
# ============================================================================

@pytest.fixture
def mock_optimization_result():
    """
    Mock optimization result for testing fit functions.
    
    Returns:
        Mock object mimicking scipy.optimize.OptimizeResult
    """
    class MockResult:
        def __init__(self):
            self.success = True
            self.x = [0.66, 1.2e-10]  # gamma, g0
            self.fun = 8261.5  # chi2
            self.message = "Optimization terminated successfully"
            self.nfev = 42  # Number of function evaluations
    
    return MockResult()


@pytest.fixture
def tolerance():
    """
    Standard numerical tolerance for floating-point comparisons.
    
    Returns:
        dict with absolute and relative tolerances
    """
    return {
        'atol': 1e-10,  # Absolute tolerance
        'rtol': 1e-6,   # Relative tolerance (0.0001%)
    }


# ============================================================================
# TEST ENVIRONMENT SETUP
# ============================================================================

@pytest.fixture(autouse=True)
def reset_random_seed():
    """
    Reset random seed before each test for reproducibility.
    
    This fixture runs automatically for all tests.
    """
    np.random.seed(42)
    import random
    random.seed(42)


@pytest.fixture(scope="session")
def suppress_warnings():
    """Suppress common warnings during tests"""
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', module='matplotlib')


# ============================================================================
# PARAMETRIZED TEST DATA
# ============================================================================

@pytest.fixture(params=[
    {'gamma': 0.66, 'g0': 1.2e-10},  # 3D+3D best-fit
    {'gamma': 0.62, 'g0': 1.0e-10},  # Lower bound
    {'gamma': 0.70, 'g0': 1.4e-10},  # Upper bound
])
def parameter_variations(request):
    """
    Parametrized fixture for testing with different parameter values.
    
    Tests using this fixture will run 3 times with different parameters.
    """
    return request.param


@pytest.fixture(params=[10, 50, 100, 500])
def sample_sizes(request):
    """
    Parametrized fixture for testing with different sample sizes.
    
    Useful for testing scaling behavior.
    """
    return request.param


# ============================================================================
# MARKERS FOR SLOW TESTS
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """
    Automatically mark slow tests based on name patterns.
    
    Tests with 'full_sample' or 'integration' in name are marked as slow.
    """
    for item in items:
        if "full_sample" in item.nodeid or "integration" in item.nodeid:
            item.add_marker(pytest.mark.slow)


# ============================================================================
# CUSTOM ASSERTIONS
# ============================================================================

@pytest.fixture
def assert_parameters_close(tolerance):
    """
    Custom assertion for comparing parameter values.
    
    Usage:
        assert_parameters_close(result, expected, param_names=['gamma', 'g0'])
    """
    def _assert(result, expected, param_names):
        for i, name in enumerate(param_names):
            assert np.abs(result[i] - expected[name]) < tolerance['rtol'] * expected[name], \
                f"{name}: {result[i]:.4e} != {expected[name]:.4e} (tolerance exceeded)"
    
    return _assert


# ============================================================================
# CLEANUP
# ============================================================================

@pytest.fixture(scope="function")
def temp_output_dir(tmp_path):
    """
    Temporary output directory for tests.
    
    Automatically cleaned up after each test.
    
    Returns:
        Path to temporary directory
    """
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# ============================================================================
# DOCUMENTATION
# ============================================================================

"""
USAGE EXAMPLES:

1. Using fixtures in tests:

    def test_rar_fitting(sample_rar_data, known_3d3d_parameters):
        result = fit_3d3d_model(sample_rar_data)
        assert result['gamma'] == pytest.approx(
            known_3d3d_parameters['gamma_rar'], 
            rel=0.1
        )

2. Using parametrized fixtures:

    def test_parameter_sensitivity(parameter_variations):
        gamma = parameter_variations['gamma']
        g0 = parameter_variations['g0']
        # Test runs 3 times with different parameters

3. Using real data (skips if unavailable):

    def test_on_real_data(sparc_rar_data):
        if sparc_rar_data is None:
            pytest.skip("SPARC data not available")
        # Test with real data

4. Using temporary directory:

    def test_file_creation(temp_output_dir):
        output_file = temp_output_dir / "result.json"
        save_results(output_file)
        assert output_file.exists()
        # Automatically cleaned up after test

5. Marking slow tests:

    @pytest.mark.slow
    def test_full_sample_analysis():
        # This test will be skipped with: pytest -m "not slow"
        pass
"""
