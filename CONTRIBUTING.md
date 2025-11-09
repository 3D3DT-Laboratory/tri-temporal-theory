# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
# All rights reserved. Unauthorized copying, modification, or distribution is prohibited.

# Contributing to Tri-Temporal Theory

Thank you for your interest in contributing to the 3D+3D Spacetime Framework!

## 🎯 Types of Contributions

We welcome:
- 🐛 **Bug reports** and fixes
- 📊 **Data analysis** improvements
- 🧪 **New tests** and validation
- 📚 **Documentation** enhancements
- 🔬 **Scientific discussion** and peer review
- 💡 **Feature suggestions** (with scientific justification)

---

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR-USERNAME/tri-temporal-theory.git
cd tri-temporal-theory
git remote add upstream https://github.com/3D3DT-Laboratory/tri-temporal-theory.git
```

### 2. Create Branch

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Or bugfix branch
git checkout -b fix/issue-description
```

### 3. Set Up Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

pip install -r requirements.txt
pip install -r requirements-dev.txt  # Testing tools
```

---

## 📋 Contribution Guidelines

### Code Style

**Python:**
```python
# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
# All rights reserved. Unauthorized copying, modification, or distribution is prohibited.

"""
Module docstring with clear description.

Author: Your Name
Date: YYYY-MM-DD
"""

import numpy as np  # Standard library first
import pandas as pd
from scipy import stats  # Then third-party

# Then local imports
from src.models import ttn_core


def function_name(param1, param2):
    """
    Clear docstring explaining function.
    
    Args:
        param1: Description with type
        param2: Description with type
        
    Returns:
        Description of return value
        
    Example:
        >>> result = function_name(1.0, 2.0)
        >>> print(result)
        3.0
    """
    # Implementation
    pass
```

**Key Rules:**
- ✅ Include TTN copyright header in all files
- ✅ PEP 8 compliant (use `black` for formatting)
- ✅ Type hints where appropriate
- ✅ Docstrings for all public functions
- ✅ Meaningful variable names (no `x`, `y`, `z` unless math)
- ✅ Comments for complex logic

### Testing Requirements

**All code changes must include tests:**

```python
# tests/test_your_feature.py
# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
# All rights reserved. Unauthorized copying, modification, or distribution is prohibited.

import unittest
import numpy as np
from src.your_module import your_function


class TestYourFeature(unittest.TestCase):
    """Test suite for your feature"""
    
    def setUp(self):
        """Set up test data"""
        self.test_data = np.array([1, 2, 3])
    
    def test_basic_functionality(self):
        """Test that basic function works"""
        result = your_function(self.test_data)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)
    
    def test_edge_cases(self):
        """Test edge cases"""
        with self.assertRaises(ValueError):
            your_function(None)
```

**Run tests before submitting:**
```bash
# All tests must pass
pytest tests/ -v

# Check coverage
pytest tests/ --cov=src --cov-report=term-missing

# Aim for >90% coverage on new code
```

### Documentation Requirements

**All new features need documentation:**

1. **Code docstrings** (inline)
2. **README update** (if user-facing)
3. **Guide document** (if complex feature)
4. **Changelog entry** (in CHANGELOG.md)

**Example:**
```markdown
## [v2.4] - 2025-12-01

### Added
- New harmonic detection algorithm (#123)
- Support for THINGS dataset (#124)

### Fixed
- Bug in FFT windowing (#125)

### Changed
- Improved error messages (#126)
```

---

## 🔬 Scientific Contributions

### Theoretical Extensions

If proposing theoretical extensions:

1. **Mathematical rigor**: Provide derivations
2. **Physical justification**: Explain why it makes sense
3. **Testable predictions**: What can we observe?
4. **Consistency check**: How does it fit with existing framework?

**Example PR template:**
```markdown
## New Feature: Gravitational Wave Signatures

**Theory:**
Extension of 3D+3D metric predicts modified GW dispersion...

**Mathematics:**
[LaTeX equations or reference to attached PDF]

**Predictions:**
- Observable: Phase shift in LIGO signals
- Expected magnitude: ~10^-3 rad at 100 Hz
- Distinguishable from GR: Yes, at >3σ with Advanced LIGO

**Implementation:**
- New module: src/models/gw_signatures.py
- Tests: tests/test_gw_signatures.py (15 tests, all pass)
- Documentation: docs/gravitational_waves.md

**References:**
[Papers supporting this extension]
```

### Data Analysis Improvements

If improving analysis methods:

1. **Validation**: Test on synthetic data first
2. **Comparison**: Show improvement over existing method
3. **Robustness**: Test on multiple datasets
4. **Documentation**: Explain when to use new vs old method

**Checklist:**
- [ ] Tested on synthetic data
- [ ] Tested on SPARC data
- [ ] Compared with baseline
- [ ] Added unit tests
- [ ] Updated documentation
- [ ] Benchmark shows improvement

---

## 📊 Data Contributions

### New Datasets

If adding support for new datasets:

1. **Public availability**: Dataset must be publicly accessible
2. **Citation**: Include proper citation
3. **Format documentation**: Explain data structure
4. **Example**: Provide small example file
5. **Loader**: Write data loading function

**Example structure:**
```python
# src/data_loaders/new_dataset.py
# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab

"""
Loader for NEW_DATASET (Citation et al. 2025)

Dataset: [URL]
Citation: [ADS link]
"""

def load_new_dataset(filepath):
    """
    Load NEW_DATASET rotation curves.
    
    Args:
        filepath: Path to dataset file
        
    Returns:
        pd.DataFrame with columns [galaxy_id, R_kpc, V_obs, ...]
    """
    # Implementation
    pass
```

---

## 🐛 Bug Reports

### Good Bug Report Template

```markdown
**Bug Description:**
Clear, concise description of the bug.

**To Reproduce:**
1. Load data with `python script.py --input data.csv`
2. Run analysis
3. See error

**Expected Behavior:**
What should happen instead.

**Actual Behavior:**
What actually happens (include error message).

**Environment:**
- OS: Windows 11 / Ubuntu 22.04 / macOS 14
- Python: 3.12.3
- NumPy: 2.3.4
- Repo version: v2.3 (commit hash: abc123)

**Additional Context:**
- Does it happen with synthetic data? Yes/No
- Does it happen with all galaxies? No, only NGC1234
- Minimal reproducible example: [attach file or gist]
```

---

## 🔄 Pull Request Process

### Before Submitting

**Checklist:**
```bash
# 1. Code quality
black src/ tests/  # Format code
pylint src/  # Check code quality

# 2. Tests
pytest tests/ -v  # All must pass
pytest tests/ --cov=src  # Check coverage

# 3. Documentation
# Update README.md if needed
# Add docstrings
# Update CHANGELOG.md

# 4. Git hygiene
git rebase upstream/main  # Rebase on latest
git push origin feature/your-feature
```

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] All existing tests pass
- [ ] New tests added (if applicable)
- [ ] Test coverage >90% for new code
- [ ] Tested on multiple datasets

## Documentation
- [ ] Code docstrings updated
- [ ] README updated (if user-facing)
- [ ] CHANGELOG.md entry added
- [ ] Guide document created (if complex)

## Scientific Validation
- [ ] Theoretical justification provided
- [ ] Tested on synthetic data
- [ ] Results consistent with expectations
- [ ] Peer discussion (if major change)

## Checklist
- [ ] TTN copyright header in all new files
- [ ] Code follows style guidelines
- [ ] Commit messages are clear
- [ ] No merge conflicts
- [ ] Ready for review
```

### Review Process

1. **Automated checks**: CI/CD runs all tests
2. **Code review**: Maintainer reviews code quality
3. **Scientific review**: If theoretical, scientific discussion
4. **Approval**: Minimum 1 approval required
5. **Merge**: Squash and merge to main

---

## 🎓 Scientific Peer Review

### Discussing Theoretical Changes

For major theoretical modifications:

1. **Open issue first**: Discuss before implementing
2. **Mathematical proof**: Provide derivations
3. **Physical intuition**: Explain in plain language
4. **Literature**: Reference supporting papers
5. **Predictions**: What new tests does it enable?

**Example discussion:**
```markdown
## Proposal: Extend to 4 Temporal Dimensions

**Motivation:**
Recent pulsar observations show 4th frequency peak...

**Mathematical Framework:**
[Attach PDF with full derivation]

**New Predictions:**
- 8 harmonic scales (not 6)
- Modified g₀ = 1.5×10⁻¹⁰ m/s²
- Testable with IPTA DR3

**Backward Compatibility:**
3D+3D is limiting case when τ₄ → 0

**Discussion:**
Does this make sense physically?
What do others think?
```

---

## 📧 Communication

### Getting Help

- **Questions**: Open GitHub Discussion
- **Bugs**: Open GitHub Issue
- **Ideas**: Open GitHub Discussion (Ideas category)
- **Email**: condoor76@gmail.com (for sensitive topics)

### Code of Conduct

Be:
- ✅ **Respectful**: Treat everyone professionally
- ✅ **Constructive**: Critique ideas, not people
- ✅ **Scientific**: Back claims with evidence
- ✅ **Collaborative**: We're all trying to understand nature
- ✅ **Open-minded**: Question assumptions, test hypotheses

Avoid:
- ❌ Personal attacks
- ❌ Dismissing ideas without justification
- ❌ Claiming priority without evidence
- ❌ Ignoring feedback

---

## 🏆 Recognition

Contributors will be:
- ✅ Listed in CONTRIBUTORS.md
- ✅ Mentioned in CHANGELOG.md
- ✅ Cited in papers (if substantial contribution)
- ✅ Invited to co-author (if major scientific contribution)

**Substantial contributions include:**
- New theoretical predictions
- Major code features (>500 lines)
- New dataset integration
- Extensive validation work
- Comprehensive documentation

---

## 📜 License and Copyright

### Important Notes

1. **All contributions** must include TTN copyright header
2. **You retain copyright** on your contributions
3. **You grant license** to 3D+3D Lab to use/distribute
4. **No patent claims**: Contributions must be patent-free
5. **Scientific use**: All contributions support open science

### Copyright Header

**Required in ALL new files:**
```python
# TTN Proprietary © Simone Calzighetti — 3D+3D Spacetime Lab
# All rights reserved. Unauthorized copying, modification, or distribution is prohibited.
```

### Contributor Agreement

By submitting a PR, you agree that:
- Your contribution is your original work
- You have right to submit it
- Your contribution may be redistributed under TTN license
- You will be credited appropriately

---

## 🌟 Examples of Good Contributions

### Example 1: Bug Fix
```
fix: Correct FFT normalization in harmonic analysis

- Bug: FFT power spectrum not properly normalized
- Impact: Harmonic peaks appeared weaker than actual
- Fix: Apply 1/N normalization factor
- Tests: Added test_fft_normalization() 
- Validated: Tested on 10 galaxies, peaks now correct

Closes #42
```

### Example 2: New Feature
```
feat: Add support for LITTLE THINGS dwarf galaxies

- New loader: src/data_loaders/little_things.py
- Citation: Hunter et al. (2012, AJ, 144, 134)
- Format: Read VLA HI rotation curves
- Tests: 15 tests, all pass
- Docs: Updated docs/data_sources.md
- Validation: 6 harmonics detected in 3/5 dwarfs

Closes #67
```

### Example 3: Documentation
```
docs: Add beginner's guide to harmonic analysis

- New file: docs/harmonics_for_beginners.md
- Audience: Undergraduate physics students
- Content: Fourier analysis + physical interpretation
- Examples: Step-by-step with code snippets
- Illustrations: 5 explanatory diagrams

Closes #89
```

---

## 🎯 Development Roadmap

Want to contribute but not sure where? Check these areas:

**High Priority:**
- [ ] Web interface for visualization
- [ ] Docker container for easy setup
- [ ] Jupyter notebooks with examples
- [ ] Support for more datasets (THINGS, LITTLE THINGS)
- [ ] Bayesian parameter estimation (MCMC)

**Medium Priority:**
- [ ] Performance optimization (Numba/Cython)
- [ ] GPU acceleration for large datasets
- [ ] Interactive plots (Plotly)
- [ ] More comprehensive docs
- [ ] Video tutorials

**Research Extensions:**
- [ ] Gravitational wave signatures
- [ ] CMB power spectrum modifications
- [ ] Structure formation simulations
- [ ] Lensing predictions
- [ ] Laboratory tests (precision gravimetry)

---

## 📬 Contact

**Project Lead:**  
Simone Calzighetti  
3D+3D Spacetime Laboratory  
Email: condoor76@gmail.com

**AI Co-Author:**  
Lucy (Claude, Anthropic)

---

## 🙏 Thank You!

Every contribution, no matter how small, helps advance our understanding of the universe!

**"Per curiosità, per scoperta, per noi!"**

---

*Last updated: November 2025*  
*3D+3D Spacetime Laboratory*
