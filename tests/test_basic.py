import numpy as np
import pytest
from abforge import (
    sample_size,
    minimum_detectable_effect,
    power_curve,
    proportions_test,
    means_test,
    chi_square_test,
    SequentialTest,
    cuped,
    check_covariate_quality,
)


def test_sample_size():
    n = sample_size(baseline_rate=0.10, min_detectable_effect=0.05)
    assert isinstance(n, int)
    assert n > 0


def test_sample_size_higher_power_needs_more_n():
    n_80 = sample_size(0.10, 0.05, power=0.80)
    n_90 = sample_size(0.10, 0.05, power=0.90)
    assert n_90 > n_80


def test_minimum_detectable_effect():
    mde = minimum_detectable_effect(baseline_rate=0.10, n=5000)
    assert 0 < mde < 1


def test_power_curve_returns_correct_length():
    effects = [0.005, 0.010, 0.015, 0.020]
    result = power_curve(0.10, effects, n=5000)
    assert len(result) == 4
    assert all(0 <= v <= 1 for v in result.values())


def test_proportions_test_significant():
    # Large effect, large n — should be significant
    result = proportions_test(450, 5000, 550, 5000)
    assert result.significant is True
    assert 0 <= result.p_value <= 1
    assert result.absolute_effect > 0


def test_proportions_test_not_significant():
    # Tiny effect — should not be significant
    result = proportions_test(500, 5000, 501, 5000)
    assert result.significant is False


def test_means_test():
    rng = np.random.default_rng(42)
    control = rng.normal(50, 10, 2000)
    treatment = rng.normal(53, 10, 2000)
    result = means_test(control, treatment)
    assert result.significant is True
    assert 0 <= result.p_value <= 1
    assert len(result.confidence_interval) == 2


def test_chi_square_test():
    table = np.array([[450, 4550], [490, 4510]])
    result = chi_square_test(table)
    assert 0 <= result.p_value <= 1
    assert isinstance(result.significant, bool)


def test_sequential_test_decisions_are_valid():
    test = SequentialTest(max_n=10000, spending="obrien_fleming")
    result = test.evaluate(450, 5000, 490, 5000)
    assert result.decision in ("stop_significant", "stop_futile", "continue")
    assert result.upper_boundary > 0
    assert result.lower_boundary < 0


def test_sequential_test_pocock():
    test = SequentialTest(max_n=10000, spending="pocock")
    result = test.evaluate(450, 5000, 490, 5000)
    assert result.decision in ("stop_significant", "stop_futile", "continue")


def test_sequential_test_invalid_spending_raises():
    with pytest.raises(ValueError):
        SequentialTest(max_n=10000, spending="made_up_function")


def test_sequential_test_simulate():
    test = SequentialTest(max_n=10000, spending="obrien_fleming")
    results = test.simulate(
        true_control_rate=0.10,
        true_treatment_rate=0.12,
        n_looks=5,
    )
    assert len(results) >= 1
    assert all(r.decision in ("stop_significant", "stop_futile", "continue") for r in results)


def test_cuped_reduces_variance():
    rng = np.random.default_rng(42)
    pre_c = rng.normal(50, 20, 2000)
    pre_t = rng.normal(50, 20, 2000)
    post_c = pre_c * 0.8 + rng.normal(50, 15, 2000)
    post_t = pre_t * 0.8 + rng.normal(52, 15, 2000)
    result = cuped(post_c, post_t, pre_c, pre_t)
    assert 0 <= result.variance_reduction <= 1
    assert result.variance_reduction > 0  # correlated covariate should reduce variance
    assert result.theta != 0


def test_check_covariate_quality_high_correlation():
    rng = np.random.default_rng(42)
    x = rng.normal(50, 20, 2000)
    y = x * 0.9 + rng.normal(0, 5, 2000)  # high correlation
    quality = check_covariate_quality(y, x)
    assert "pearson_r" in quality
    assert "r_squared" in quality
    assert "recommendation" in quality
    assert quality["r_squared"] > 0.5  # should be strongly correlated


def test_check_covariate_quality_low_correlation():
    rng = np.random.default_rng(42)
    x = rng.normal(50, 20, 2000)
    y = rng.normal(50, 20, 2000)  # independent — no correlation
    quality = check_covariate_quality(y, x)
    assert quality["r_squared"] < 0.1
