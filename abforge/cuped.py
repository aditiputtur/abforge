import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional
from .stats import TestResult


@dataclass
class CUPEDResult:
    """Container for CUPED analysis results."""
    theta: float                    # Covariate coefficient
    variance_reduction: float       # % variance reduced
    original_result: TestResult     # Result without CUPED
    adjusted_result: TestResult     # Result with CUPED

    def __repr__(self):
        return (
            f"\nCUPED Variance Reduction Analysis\n"
            f"{'─' * 50}\n"
            f"  Covariate coefficient (θ):  {self.theta:.4f}\n"
            f"  Variance reduction:          {self.variance_reduction:.1%}\n"
            f"\n  WITHOUT CUPED:\n"
            f"  p-value: {self.original_result.p_value:.4f}  |  "
            f"Significant: {self.original_result.significant}\n"
            f"\n  WITH CUPED:\n"
            f"  p-value: {self.adjusted_result.p_value:.4f}  |  "
            f"Significant: {self.adjusted_result.significant}\n"
            f"\n  Sensitivity improvement: "
            f"{self.original_result.p_value / max(self.adjusted_result.p_value, 1e-10):.2f}x\n"
        )


def cuped(
    control_metric: np.ndarray,
    treatment_metric: np.ndarray,
    control_covariate: np.ndarray,
    treatment_covariate: np.ndarray,
    alpha: float = 0.05,
) -> CUPEDResult:
    """
    CUPED — Controlled-experiment Using Pre-Experiment Data.

    Reduces variance in the outcome metric using a pre-experiment 
    covariate (e.g. pre-experiment revenue, prior session count),
    making the test more sensitive without increasing sample size.

    The key insight: if we can explain some of the variance in the 
    outcome metric using pre-experiment data, we can subtract that 
    explained variance and get a cleaner signal.

    Adjusted metric: Y_adj = Y - θ * (X - E[X])
    where θ = Cov(Y, X) / Var(X)

    Parameters
    ----------
    control_metric : np.ndarray
        Post-experiment outcome values for control group
        (e.g. revenue per user, sessions per user)
    treatment_metric : np.ndarray
        Post-experiment outcome values for treatment group
    control_covariate : np.ndarray
        Pre-experiment covariate for control group
        (e.g. revenue in prior 30 days — same users, before experiment)
    treatment_covariate : np.ndarray
        Pre-experiment covariate for treatment group
    alpha : float
        Significance level (default 0.05)

    Returns
    -------
    CUPEDResult containing original and adjusted TestResults

    Example
    -------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> # Simulate pre-experiment revenue (covariate)
    >>> control_pre = rng.normal(50, 20, 5000)
    >>> treatment_pre = rng.normal(50, 20, 5000)
    >>> # Post-experiment revenue — treatment has +5% lift
    >>> control_post = control_pre * 0.8 + rng.normal(50, 15, 5000)
    >>> treatment_post = treatment_pre * 0.8 + rng.normal(52.5, 15, 5000)
    >>> result = cuped(control_post, treatment_post, 
    ...                control_pre, treatment_pre)
    >>> print(result)

    Notes
    -----
    The covariate should be:
    - Measured BEFORE the experiment starts (no treatment contamination)
    - Correlated with the outcome metric (higher correlation = more reduction)
    - The same metric or a related one (e.g. prior revenue for revenue test)
    
    References
    ----------
    Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). 
    Improving the sensitivity of online controlled experiments by 
    utilizing pre-experiment data. WSDM 2013.
    https://exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf
    """
    from .stats import means_test

    # Pool all data to estimate theta
    all_metric = np.concatenate([control_metric, treatment_metric])
    all_covariate = np.concatenate([control_covariate, treatment_covariate])

    # θ = Cov(Y, X) / Var(X)
    theta = np.cov(all_metric, all_covariate)[0, 1] / np.var(all_covariate, ddof=1)

    # Global covariate mean (E[X])
    covariate_mean = np.mean(all_covariate)

    # Adjusted metrics: Y_adj = Y - θ * (X - E[X])
    control_adjusted = control_metric - theta * (control_covariate - covariate_mean)
    treatment_adjusted = treatment_metric - theta * (treatment_covariate - covariate_mean)

    # Variance reduction achieved
    original_var = np.var(all_metric, ddof=1)
    adjusted_var = np.var(
        np.concatenate([control_adjusted, treatment_adjusted]), ddof=1
    )
    variance_reduction = (original_var - adjusted_var) / original_var

    # Run both tests for comparison
    original_result = means_test(control_metric, treatment_metric, alpha=alpha)
    adjusted_result = means_test(control_adjusted, treatment_adjusted, alpha=alpha)

    # Relabel test names for clarity
    original_result.test_name = "T-Test (without CUPED)"
    adjusted_result.test_name = "T-Test (with CUPED)"

    return CUPEDResult(
        theta=round(theta, 6),
        variance_reduction=round(variance_reduction, 6),
        original_result=original_result,
        adjusted_result=adjusted_result,
    )


def check_covariate_quality(
    metric: np.ndarray,
    covariate: np.ndarray,
) -> dict:
    """
    Assess how useful a covariate will be for CUPED.
    Higher correlation = more variance reduction.

    Parameters
    ----------
    metric : np.ndarray
        Post-experiment outcome metric (pooled across variants)
    covariate : np.ndarray
        Pre-experiment covariate (pooled across variants)

    Returns
    -------
    dict with correlation, expected variance reduction, and recommendation

    Example
    -------
    >>> quality = check_covariate_quality(revenue_post, revenue_pre)
    >>> print(quality)
    """
    r, p_value = stats.pearsonr(metric, covariate)
    expected_reduction = r ** 2  # R² = proportion of variance explained

    if expected_reduction >= 0.30:
        recommendation = "Excellent — strong variance reduction expected"
    elif expected_reduction >= 0.10:
        recommendation = "Good — moderate variance reduction expected"
    elif expected_reduction >= 0.05:
        recommendation = "Weak — modest improvement, still worth using"
    else:
        recommendation = "Poor — covariate explains little variance, consider alternatives"

    return {
        "pearson_r": round(r, 4),
        "r_squared": round(expected_reduction, 4),
        "expected_variance_reduction": f"{expected_reduction:.1%}",
        "p_value": round(p_value, 6),
        "recommendation": recommendation,
    }
