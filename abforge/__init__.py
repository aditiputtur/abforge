from .power import sample_size, minimum_detectable_effect, power_curve
from .stats import proportions_test, means_test, chi_square_test, TestResult
from .sequential import SequentialTest, AlphaSpendingFunction, SequentialTestResult
from .cuped import cuped, check_covariate_quality, CUPEDResult
from .viz import (plot_power_curve, plot_test_result,
                  plot_sequential_boundaries, plot_cuped_comparison)
__version__ = "0.1.0"
__author__ = "Aditi Puttur"
