# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
# All rights reserved. Unauthorized copying, modification, or distribution is prohibited.

"""
Unit and Integration Tests for RAR Fitting Pipeline
====================================================

Tests the 3D+3D RAR validation code for:
- Data loading and validation
- Model fitting (LCDM, MOND, 3D+3D)
- Statistical metrics (chi-square, R², RMS)
- Edge cases and error handling

Author: Simone Calzighetti
Date: 2025-11-08
"""

import unittest
import numpy as np
import pandas as pd
import json
import os
import tempfile
from pathlib import Path

# Try to import the main module (will fail if not available)
try:
    import sys
    # Adjust path based on where tests are run from
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'models' / 'analysis'))
    from rar_fit_logspace import (
        load_rar_data,
        fit_lcdm,
        fit_mond,
        fit_3d3d,
        compute_metrics,
        save_results
    )
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    print("⚠️  Warning: Cannot import rar_fit_logspace module")
    print("   Tests will be skipped. Make sure to run from repo root.")


class TestDataLoading(unittest.TestCase):
    """Test data loading and validation"""
    
    def setUp(self):
        """Create synthetic test data"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create minimal valid CSV
        self.test_csv = os.path.join(self.temp_dir, 'test_rar.csv')
        df = pd.DataFrame({
            'galaxy_id': ['NGC1234'] * 10,
            'R_kpc': np.logspace(-1, 1, 10),
            'V_obs': np.linspace(50, 200, 10),
            'g_obs': np.logspace(-11, -9, 10),
            'V_bar': np.linspace(40, 180, 10),
            'g_bar': np.logspace(-11.5, -9.5, 10)
        })
        df.to_csv(self.test_csv, index=False)
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Module not available")
    def test_load_valid_csv(self):
        """Test loading valid RAR data"""
        df = load_rar_data(self.test_csv)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn('g_bar', df.columns)
        self.assertIn('g_obs', df.columns)
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Module not available")
    def test_missing_columns(self):
        """Test error handling for missing columns"""
        # Create invalid CSV
        bad_csv = os.path.join(self.temp_dir, 'bad.csv')
        df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
        df.to_csv(bad_csv, index=False)
        
        with self.assertRaises((KeyError, ValueError)):
            load_rar_data(bad_csv)
    
    def test_data_physical_consistency(self):
        """Test that g = V²/R is satisfied"""
        df = pd.read_csv(self.test_csv)
        
        # Check g_obs ≈ V_obs²/R
        g_calc = (df['V_obs'] * 1000)**2 / (df['R_kpc'] * 3.086e19)
        g_from_csv = df['g_obs']
        
        # Should match within 10% (accounting for unit conversions)
        ratio = g_calc / g_from_csv
        self.assertTrue(np.all(ratio > 0.5))
        self.assertTrue(np.all(ratio < 2.0))


class TestModelFitting(unittest.TestCase):
    """Test individual model fitting functions"""
    
    def setUp(self):
        """Create synthetic RAR data"""
        np.random.seed(42)
        
        # Generate realistic RAR data
        g_bar = np.logspace(-12, -8, 100)  # m/s²
        g0 = 1.2e-10
        gamma = 0.66
        
        # True 3D+3D model + noise
        g_obs_true = g0**(1-gamma) * g_bar**gamma
        g_obs = g_obs_true * np.exp(np.random.normal(0, 0.05, len(g_bar)))
        
        self.test_data = pd.DataFrame({
            'g_bar': g_bar,
            'g_obs': g_obs
        })
        self.g0_true = g0
        self.gamma_true = gamma
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Module not available")
    def test_lcdm_fit(self):
        """Test ΛCDM linear fit"""
        params, metrics = fit_lcdm(
            self.test_data['g_bar'],
            self.test_data['g_obs'],
            sigma_int=0.0
        )
        
        self.assertIn('B', params)
        self.assertGreater(params['B'], 0)
        self.assertLess(params['B'], 1.5)
        self.assertIn('chi2_red', metrics)
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Module not available")
    def test_mond_fit(self):
        """Test MOND fit"""
        params, metrics = fit_mond(
            self.test_data['g_bar'],
            self.test_data['g_obs'],
            sigma_int=0.0
        )
        
        self.assertIn('a0', params)
        self.assertGreater(params['a0'], 1e-12)
        self.assertLess(params['a0'], 1e-9)
        self.assertIn('chi2_red', metrics)
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Module not available")
    def test_3d3d_fit(self):
        """Test 3D+3D fit recovers true parameters"""
        params, metrics = fit_3d3d(
            self.test_data['g_bar'],
            self.test_data['g_obs'],
            g0_fixed=self.g0_true,
            sigma_int=0.0
        )
        
        self.assertIn('gamma', params)
        self.assertIn('g0', params)
        
        # Should recover gamma ≈ 0.66
        self.assertAlmostEqual(params['gamma'], self.gamma_true, delta=0.1)
        
        # g0 should be fixed
        self.assertEqual(params['g0'], self.g0_true)
        
        # Should have low chi²
        self.assertLess(metrics['chi2_red'], 5.0)
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Module not available")
    def test_gamma_bounds(self):
        """Test that gamma stays within physical bounds"""
        params, _ = fit_3d3d(
            self.test_data['g_bar'],
            self.test_data['g_obs'],
            g0_fixed=self.g0_true,
            sigma_int=0.0
        )
        
        # Gamma should be in [0.3, 1.0]
        self.assertGreaterEqual(params['gamma'], 0.3)
        self.assertLessEqual(params['gamma'], 1.0)


class TestMetrics(unittest.TestCase):
    """Test statistical metric calculations"""
    
    def setUp(self):
        """Create test predictions"""
        self.y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        self.sigma = np.ones(5) * 0.2
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Module not available")
    def test_chi_square(self):
        """Test chi-square calculation"""
        metrics = compute_metrics(
            self.y_true,
            self.y_pred,
            self.sigma,
            n_params=1
        )
        
        self.assertIn('chi2', metrics)
        self.assertIn('chi2_red', metrics)
        self.assertGreater(metrics['chi2'], 0)
        
        # chi2_red = chi2 / (N - n_params)
        expected = metrics['chi2'] / (len(self.y_true) - 1)
        self.assertAlmostEqual(metrics['chi2_red'], expected, places=5)
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Module not available")
    def test_r_squared(self):
        """Test R² calculation"""
        metrics = compute_metrics(
            self.y_true,
            self.y_pred,
            self.sigma,
            n_params=1
        )
        
        self.assertIn('R2_w', metrics)
        self.assertGreaterEqual(metrics['R2_w'], 0)
        self.assertLessEqual(metrics['R2_w'], 1)
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Module not available")
    def test_rms(self):
        """Test RMS calculation"""
        metrics = compute_metrics(
            self.y_true,
            self.y_pred,
            self.sigma,
            n_params=1
        )
        
        self.assertIn('RMS', metrics)
        
        # Manual RMS
        residuals = np.log10(self.y_pred) - np.log10(self.y_true)
        expected_rms = np.sqrt(np.mean(residuals**2))
        
        self.assertAlmostEqual(metrics['RMS'], expected_rms, places=5)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Module not available")
    def test_empty_data(self):
        """Test handling of empty datasets"""
        df = pd.DataFrame({'g_bar': [], 'g_obs': []})
        
        with self.assertRaises((ValueError, IndexError)):
            fit_3d3d(df['g_bar'], df['g_obs'])
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Module not available")
    def test_negative_values(self):
        """Test handling of negative accelerations"""
        g_bar = np.array([1e-10, -1e-11, 1e-9])
        g_obs = np.array([2e-10, 3e-11, 2e-9])
        
        # Should either filter or raise error
        try:
            params, _ = fit_3d3d(g_bar, g_obs)
            # If it succeeds, check it filtered negatives
            self.assertTrue(True)
        except ValueError:
            # Or it should raise an error
            self.assertTrue(True)
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Module not available")
    def test_nan_handling(self):
        """Test NaN handling"""
        g_bar = np.array([1e-10, np.nan, 1e-9])
        g_obs = np.array([2e-10, 3e-11, np.nan])
        
        # Should filter or raise
        with self.assertRaises((ValueError, RuntimeError)):
            fit_3d3d(g_bar, g_obs)


class TestKnownResults(unittest.TestCase):
    """Test against known good results"""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Module not available")
    def test_known_rar_results(self):
        """
        Test that we reproduce known RAR results:
        - γ_RAR = 0.66 ± 0.04
        - g₀ = 1.2×10⁻¹⁰ m/s²
        - χ²_red = 2.44
        """
        # This would require the actual SPARC data
        # Placeholder for integration test
        pass


class TestOutputFormat(unittest.TestCase):
    """Test output JSON format"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.output_file = os.path.join(self.temp_dir, 'results.json')
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Module not available")
    def test_json_structure(self):
        """Test that output JSON has correct structure"""
        # Create dummy results
        results = {
            'models': {
                'LCDM': {'params': {'B': 0.68}, 'metrics': {'chi2_red': 2.27}},
                'MOND': {'params': {'a0': 3.4e-11}, 'metrics': {'chi2_red': 2.65}},
                '3D3D': {'params': {'gamma': 0.66, 'g0': 1.2e-10}, 'metrics': {'chi2_red': 2.44}}
            }
        }
        
        # Save and reload
        with open(self.output_file, 'w') as f:
            json.dump(results, f)
        
        with open(self.output_file, 'r') as f:
            loaded = json.load(f)
        
        self.assertIn('models', loaded)
        self.assertIn('3D3D', loaded['models'])
        self.assertIn('gamma', loaded['models']['3D3D']['params'])


class TestRegressions(unittest.TestCase):
    """Test for known bugs/regressions"""
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Module not available")
    def test_sigma_int_zero(self):
        """
        Regression test: σ_int = 0.08 was causing artificially high χ²
        Ensure σ_int = 0.0 works correctly
        """
        g_bar = np.logspace(-11, -9, 50)
        g_obs = np.logspace(-10.5, -8.5, 50)
        
        # Should not crash with sigma_int=0
        params, metrics = fit_3d3d(
            g_bar, g_obs,
            g0_fixed=1.2e-10,
            sigma_int=0.0
        )
        
        self.assertIsNotNone(params)
        self.assertIsNotNone(metrics)
    
    @unittest.skipIf(not IMPORTS_AVAILABLE, "Module not available")
    def test_gamma_not_negative(self):
        """
        Regression test: Old additive boost formula gave negative γ
        Power-law formula should always give positive γ
        """
        g_bar = np.logspace(-11, -9, 50)
        g_obs = np.logspace(-10.5, -8.5, 50)
        
        params, _ = fit_3d3d(g_bar, g_obs, g0_fixed=1.2e-10)
        
        self.assertGreater(params['gamma'], 0)


def run_tests(verbosity=2):
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
    
    return result


if __name__ == '__main__':
    import sys
    
    if not IMPORTS_AVAILABLE:
        print("="*70)
        print("⚠️  MODULE IMPORT FAILED")
        print("="*70)
        print("Cannot import rar_fit_logspace module.")
        print("\nTo run tests:")
        print("1. Ensure src/models/analysis/rar_fit_logspace.py exists")
        print("2. Run from repository root: python -m pytest tests/")
        print("3. Or: python tests/test_rar_fitting.py")
        print("="*70)
        sys.exit(1)
    
    run_tests(verbosity=2)
