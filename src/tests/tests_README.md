# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
# All rights reserved. Unauthorized copying, modification, or distribution is prohibited.

# Test Suite Documentation

Comprehensive test suite for the Tri-Temporal Theory validation package.

**Current Status:** ✅ **20/20 tests passing** | ⏱️ Runtime: ~0.03s | 📊 Coverage: 95%

---

## 🎯 Test Philosophy

### Goals

1. **Reproducibility**: Validate all published results
2. **Regression prevention**: Catch breaking changes early
3. **Scientific rigor**: Test against known theoretical predictions
4. **Continuous validation**: Automated testing on every commit

### Principles

- ✅ **Fast**: Full suite runs in <1 second
- ✅ **Deterministic**: Same input → same output (seeded RNG)
- ✅ **Isolated**: Tests don't depend on each other
- ✅ **Comprehensive**: Cover edge cases and normal operation
- ✅ **Documented**: Clear purpose and expected behavior

---

## 📂 Test Structure

```
tests/
├── __init__.py              # Test configuration
├── conftest.py              # Pytest fixtures (shared test data)
├── test_rar_fitting.py      # RAR analysis tests (20 tests)
└── test_six_harmonics.py    # Harmonic analysis tests (TODO)
```

---

## 🧪 Test Categories

### 1. Utility Functions (2 tests)

**Purpose:** Validate helper functions

```python
def test_log10_safe():
    """Ensure log10 handles zeros/negatives"""
    
def test_inverse_variance_weights():
    """Validate weight calculation"""
```

**Coverage:**
- Edge cases (zeros, negatives, infinities)
- Numerical stability
- Output shapes and types

---

### 2. Model Formulas (3 tests)

**Purpose:** Verify theoretical predictions

```python
def test_3d3d_formula():
    """Test 3D+3D acceleration formula"""
    # g_pred = g_bar + g_bar * γ * exp(-g_bar/g0)
    
def test_mond_formula():
    """Test MOND acceleration formula"""
    # g_pred = g_bar * sqrt(1 + (a0/g_bar)^2)
    
def test_lcdm_formula():
    """Test ΛCDM acceleration formula"""
    # g_pred = B * g_bar
```

**Validation:**
- Asymptotic behavior (g_bar → 0, g_bar → ∞)
- Continuity and smoothness
- Parameter sensitivity
- Dimensional consistency

---

### 3. Data Loading (2 tests)

**Purpose:** Ensure data integrity

```python
def test_load_rar_data():
    """Validate RAR data loader"""
    # Load data/processed/rar_data.csv
    # Check: columns, data types, ranges
    
def test_data_quality():
    """Check data quality"""
    # No NaNs in critical columns
    # Physical ranges (g > 0)
    # Reasonable uncertainties
```

**Checks:**
- File existence and format
- Column names and types
- Data ranges (physical constraints)
- Missing values
- Outliers

---

### 4. Model Fitting (4 tests)

**Purpose:** Validate fitting algorithms

```python
def test_fit_3d3d():
    """Test 3D+3D model fitting"""
    
def test_fit_mond():
    """Test MOND model fitting"""
    
def test_fit_lcdm():
    """Test ΛCDM model fitting"""
    
def test_fitting_convergence():
    """Ensure optimizer converges"""
```

**Validation:**
- Convergence to known parameters (synthetic data)
- Fit quality metrics (χ², R²)
- Optimization stability
- Parameter bounds respected

---

### 5. Weighted Metrics (2 tests)

**Purpose:** Verify statistical calculations

```python
def test_weighted_chi_square():
    """Test χ² calculation with weights"""
    
def test_weighted_r_squared():
    """Test R² calculation with weights"""
```

**Checks:**
- Perfect fit → χ² = 0, R² = 1
- Random data → χ² ≫ 1, R² ≈ 0
- Weight normalization
- Numerical accuracy

---

### 6. Known Results (4 tests)

**Purpose:** Reproduce published findings

```python
def test_gamma_rar():
    """Validate γ_RAR = 0.66 ± 0.04"""
    
def test_g0_convergence():
    """Validate g₀ = 1.2×10⁻¹⁰ m/s²"""
    
def test_chi_square_comparison():
    """Validate χ²_3D3D = 2.44 (< MOND)"""
    
def test_mass_amplitude_scaling():
    """Validate α_M = 0.30 ± 0.06"""
```

**Critical Tests:**
These validate the **core empirical results** of the theory!

---

### 7. Regression Tests (2 tests)

**Purpose:** Prevent breaking changes

```python
def test_output_format_unchanged():
    """Ensure output JSON structure stable"""
    
def test_backward_compatibility():
    """Check old analysis scripts still work"""
```

**Why Important:**
- Users depend on consistent output format
- Scripts written for v2.2 should work in v2.3
- API stability for reproducibility

---

### 8. Integration Tests (1 test)

**Purpose:** Test full pipeline

```python
def test_full_rar_pipeline():
    """End-to-end: Load data → Fit → Compare → Output"""
```

**Validates:**
- All components work together
- No unexpected interactions
- Output files created correctly
- Results match expected values

---

## 🚀 Running Tests

### Quick Test (Development)

```bash
# Run single test file
python tests/test_rar_fitting.py

# Output:
# Ran 20 tests in 0.031s
# OK
```

### Full Test Suite (CI/CD)

```bash
# With pytest
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html

# Specific test
pytest tests/test_rar_fitting.py::TestModelFitting::test_fit_3d3d -v
```

### Test Options

```bash
# Verbose output
pytest tests/ -v

# Stop on first failure
pytest tests/ -x

# Run only failed tests
pytest tests/ --lf

# Run tests matching pattern
pytest tests/ -k "gamma"

# Show print statements
pytest tests/ -s

# Parallel execution (if pytest-xdist installed)
pytest tests/ -n auto
```

---

## 📊 Test Coverage

### Current Coverage: 95%

```
src/models/analysis/rar_fit_logspace.py    98%
src/models/ttn_core.py                     95%
src/models/baselines.py                    92%
src/data_io.py                             96%
src/utils.py                               94%
```

**Uncovered Lines:**
- Error handling for rare edge cases
- Plotting code (visual inspection)
- Interactive CLI features

### Coverage Goals

- **Critical code**: 100% (model formulas, fitting)
- **Analysis code**: >95% (main pipeline)
- **Utilities**: >90% (helpers)
- **Plotting**: >70% (visual verification)

---

## 🔧 Writing New Tests

### Test Template

```python
# tests/test_your_feature.py
# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab

"""
Test suite for your_feature module.

Tests:
- Test 1: Purpose
- Test 2: Purpose
...
"""

import unittest
import numpy as np
from src.your_module import your_function


class TestYourFeature(unittest.TestCase):
    """Test suite for YourFeature"""
    
    @classmethod
    def setUpClass(cls):
        """Run once before all tests (expensive setup)"""
        cls.large_dataset = load_expensive_data()
    
    def setUp(self):
        """Run before each test (test-specific setup)"""
        self.test_data = np.array([1.0, 2.0, 3.0])
        self.expected_output = np.array([2.0, 4.0, 6.0])
    
    def tearDown(self):
        """Run after each test (cleanup)"""
        pass
    
    def test_basic_functionality(self):
        """Test that function works in normal case"""
        result = your_function(self.test_data)
        np.testing.assert_array_almost_equal(
            result, 
            self.expected_output,
            decimal=10
        )
    
    def test_edge_case_empty_input(self):
        """Test behavior with empty input"""
        result = your_function(np.array([]))
        self.assertEqual(len(result), 0)
    
    def test_error_handling(self):
        """Test that invalid input raises appropriate error"""
        with self.assertRaises(ValueError):
            your_function(None)
    
    def test_numerical_stability(self):
        """Test with extreme values"""
        large_values = np.array([1e10, 1e15, 1e20])
        result = your_function(large_values)
        self.assertTrue(np.all(np.isfinite(result)))


if __name__ == '__main__':
    unittest.main()
```

### Best Practices

**DO:**
- ✅ Test one thing per test function
- ✅ Use descriptive test names (`test_gamma_within_bounds`)
- ✅ Use `np.testing` for array comparisons
- ✅ Set random seeds for reproducibility
- ✅ Test edge cases (empty, None, inf, NaN)
- ✅ Document expected behavior in docstring
- ✅ Keep tests fast (<0.1s each)

**DON'T:**
- ❌ Test implementation details (test behavior!)
- ❌ Make tests depend on each other
- ❌ Use sleep() or time-dependent tests
- ❌ Compare floats with `==` (use `np.allclose`)
- ❌ Forget to test error cases
- ❌ Write tests that require internet/external files

---

## 🎯 Test Fixtures (conftest.py)

### Shared Test Data

```python
# tests/conftest.py
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def sample_rar_data():
    """Provide sample RAR data for testing"""
    return pd.DataFrame({
        'gbar': np.logspace(-12, -9, 100),
        'gobs': np.logspace(-11.5, -9.5, 100),
        'e_gobs': np.full(100, 1e-11)
    })


@pytest.fixture
def synthetic_galaxy():
    """Generate synthetic galaxy rotation curve"""
    r = np.linspace(0.1, 30, 100)  # kpc
    v_true = 200 * np.sqrt(1 - np.exp(-r/5))  # km/s
    v_obs = v_true + np.random.normal(0, 5, 100)
    return {'r': r, 'v_obs': v_obs, 'v_err': np.full(100, 5)}


@pytest.fixture
def known_parameters():
    """Known parameter values for validation"""
    return {
        'gamma_rar': 0.66,
        'g0': 1.2e-10,  # m/s²
        'chi2_3d3d': 2.44,
        'alpha_m': 0.30
    }
```

### Using Fixtures

```python
def test_with_fixture(sample_rar_data, known_parameters):
    """Test using shared fixtures"""
    result = fit_3d3d_model(sample_rar_data)
    assert np.abs(result['gamma'] - known_parameters['gamma_rar']) < 0.1
```

---

## 🐛 Debugging Failed Tests

### Reading Test Output

```bash
$ pytest tests/test_rar_fitting.py::test_gamma_rar -v

FAILED tests/test_rar_fitting.py::test_gamma_rar
________________________ test_gamma_rar ________________________

    def test_gamma_rar():
        """Validate γ_RAR = 0.66 ± 0.04"""
        result = fit_rar_3d3d()
>       assert 0.62 <= result['gamma'] <= 0.70
E       AssertionError: assert 0.62 <= 0.75 <= 0.70

tests/test_rar_fitting.py:89: AssertionError
```

**Diagnosis:**
- Expected: γ ∈ [0.62, 0.70]
- Got: γ = 0.75
- Issue: Fitting algorithm found different optimum

### Common Failure Modes

**1. Numerical Precision**
```python
# Bad
assert result == 0.66

# Good
assert np.abs(result - 0.66) < 1e-6
```

**2. Platform Differences**
```python
# Bad (may fail on different OS/NumPy versions)
assert result == 2.440000000001

# Good
np.testing.assert_allclose(result, 2.44, rtol=1e-5)
```

**3. Random Seed Issues**
```python
# Bad (non-reproducible)
data = np.random.randn(100)

# Good (reproducible)
np.random.seed(42)
data = np.random.randn(100)
```

---

## 📈 Continuous Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.10', '3.11', '3.12', '3.13']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: pytest tests/ -v --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

**Benefits:**
- ✅ Automatic testing on every push
- ✅ Test on multiple OS/Python versions
- ✅ Coverage reporting
- ✅ Block merges if tests fail

---

## 📊 Test Metrics

### Current Performance

```
Test Suite Statistics (v2.3):
├─ Total Tests: 20
├─ Passing: 20 (100%)
├─ Failing: 0
├─ Skipped: 0
├─ Runtime: 0.031s
├─ Coverage: 95%
└─ Lines Covered: 1,247 / 1,312
```

### Historical Trends

```
v2.1: 8 tests, 87% coverage
v2.2: 15 tests, 91% coverage
v2.3: 20 tests, 95% coverage ← Current
```

**Goal:** 25+ tests, 98% coverage by v2.4

---

## 🎓 Test-Driven Development (TDD)

### TDD Workflow

```
1. Write test (it fails) 🔴
2. Write minimal code to pass test 🟢
3. Refactor code 🔄
4. Repeat!
```

### Example: Adding New Feature

**Goal:** Add mass-dependent harmonic detection

**Step 1: Write test first**
```python
def test_mass_dependent_harmonics():
    """Test that λ₀ detection increases with mass"""
    low_mass = generate_dwarf_galaxy()
    high_mass = generate_spiral_galaxy()
    
    result_low = detect_harmonics(low_mass)
    result_high = detect_harmonics(high_mass)
    
    # Expect λ₀ stronger in high-mass galaxies
    assert result_high['lambda_0_snr'] > result_low['lambda_0_snr']
```

**Step 2: Run test (fails)**
```bash
$ pytest tests/test_harmonics.py::test_mass_dependent_harmonics
FAILED - Function detect_harmonics() not found
```

**Step 3: Implement feature**
```python
def detect_harmonics(galaxy_data):
    # Implementation...
    return {'lambda_0_snr': snr}
```

**Step 4: Run test (passes)**
```bash
$ pytest tests/test_harmonics.py::test_mass_dependent_harmonics
PASSED
```

**Step 5: Refactor and optimize**

---

## 🏆 Test Quality Checklist

Before submitting PR, ensure:

- [ ] All tests pass locally
- [ ] New features have tests (>90% coverage)
- [ ] Tests are fast (<0.1s each)
- [ ] Tests are deterministic (no random failures)
- [ ] Edge cases tested (empty, None, inf, NaN)
- [ ] Error handling tested
- [ ] Test names are descriptive
- [ ] Docstrings explain test purpose
- [ ] No dependencies between tests
- [ ] Tests use fixtures for shared data
- [ ] Random seeds set for reproducibility

---

## 📚 Additional Resources

### Testing Documentation

- **pytest**: https://docs.pytest.org/
- **unittest**: https://docs.python.org/3/library/unittest.html
- **NumPy testing**: https://numpy.org/doc/stable/reference/routines.testing.html

### Best Practices

- Martin Fowler: *Refactoring* (Chapter on Testing)
- Kent Beck: *Test-Driven Development by Example*
- Python Testing with pytest (Brian Okken)

### Scientific Testing

- Wilson et al. (2014): *Best Practices for Scientific Computing*
- Taschuk & Wilson (2017): *Ten Simple Rules for Making Research Software More Robust*

---

## 📧 Questions?

**Test issues?** Open GitHub issue with `[tests]` tag

**Need help writing tests?** See [CONTRIBUTING.md](../CONTRIBUTING.md)

**Found a bug?** Write a failing test first, then fix it!

---

**"Tests are the specification of correct behavior"**

*Last updated: November 2025*  
*3D+3D Spacetime Laboratory*
