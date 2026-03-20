import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional


@dataclass
class TestResult:
    """Container for hypothesis test results."""
    test_name: str
    control_mean: float
    treatment_mean: float
    absolute_effect: float
    relative_effect: float
    p_value: float
    statistic: float
    confidence_interval: tuple
    significant: bool
    alpha: float

    def __repr__(self):
        sig = "SIGNIFICANT" if self.significant else "NOT SIGNIFICANT"
        return (
            f"\n{self.test_name} Results — {sig}\n"
            f"{'─' * 45}\n"
            f"  Control mean:     {self.control_mean:.4f}\n"
            f"  Treatment mean:   {self.treatment_mean:.4f}\n"
            f"  Absolute effect:  {self.absolute_effect:+.4f}\n"
            f"  Relative effect:  {self.relative_effect:+.2%}\n"
            f"  p-value:          {self.p_value:.4f}\n"
            f"  95% CI:           ({self.confidence_interval[0]:.4f}, "
            f"{self.confidence_interval[1]:.4f})\n"
            f"  Significant:      {self.significant} (alpha={self.alpha})\n"
        )


def proportions_test(
    control_conversions: int,
    control_n: int,
    treatment_conversions: int,
    treatment_n: int,
    alpha: float = 0.05,
    two_tailed: bool = True,
) -> TestResult:
    """
    Two-proportion z-test for conversion rate experiments.

    Parameters
    ----------
    control_conversions : int
        Number of conversions in control group
    control_n : int
        Total users in control group
    treatment_conversions : int
        Number of conversions in treatment group
    treatment_n : int
        Total users in treatment group
    alpha : float
        Significance level (default 0.05)
    two_tailed : bool
        Whether to use two-tailed test (default True)

    Returns
    -------
    TestResult

    Example
    -------
    >>> result = proportions_test(450, 5000, 490, 5000)
    >>> print(result)
    """
    p_control = control_conversions / control_n
    p_treatment = treatment_conversions / treatment_n
    
    p_pooled = (control_conversions + treatment_conversions) / (control_n + treatment_n)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / control_n + 1 / treatment_n))
    
    z = (p_treatment - p_control) / se
    
    if two_tailed:
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        p_value = 1 - stats.norm.cdf(z)
    
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    se_diff = np.sqrt(
        p_control * (1 - p_control) / control_n +
        p_treatment * (1 - p_treatment) / treatment_n
    )
    diff = p_treatment - p_control
    ci = (diff - z_alpha * se_diff, diff + z_alpha * se_diff)
    
    return TestResult(
        test_name="Two-Proportion Z-Test",
        control_mean=round(p_control, 6),
        treatment_mean=round(p_treatment, 6),
        absolute_effect=round(diff, 6),
        relative_effect=round(diff / p_control, 6),
        p_value=round(p_value, 6),
        statistic=round(z, 6),
        confidence_interval=(round(ci[0], 6), round(ci[1], 6)),
        significant=p_value < alpha,
        alpha=alpha,
    )


def means_test(
    control_values: np.ndarray,
    treatment_values: np.ndarray,
    alpha: float = 0.05,
    equal_var: bool = False,
) -> TestResult:
    """
    Welch's t-test for continuous metrics (revenue, time on page, etc.)

    Parameters
    ----------
    control_values : np.ndarray
        Array of metric values for control group
    treatment_values : np.ndarray
        Array of metric values for treatment group
    alpha : float
        Significance level (default 0.05)
    equal_var : bool
        If True, use Student's t-test. Default False (Welch's).

    Returns
    -------
    TestResult
    """
    t_stat, p_value = stats.ttest_ind(
        control_values, treatment_values, equal_var=equal_var
    )
    
    mean_control = np.mean(control_values)
    mean_treatment = np.mean(treatment_values)
    diff = mean_treatment - mean_control
    
    # Confidence interval
    n_c, n_t = len(control_values), len(treatment_values)
    se = np.sqrt(
        np.var(control_values, ddof=1) / n_c +
        np.var(treatment_values, ddof=1) / n_t
    )
    df = n_c + n_t - 2
    t_alpha = stats.t.ppf(1 - alpha / 2, df)
    ci = (diff - t_alpha * se, diff + t_alpha * se)
    
    return TestResult(
        test_name="Welch's T-Test",
        control_mean=round(mean_control, 6),
        treatment_mean=round(mean_treatment, 6),
        absolute_effect=round(diff, 6),
        relative_effect=round(diff / mean_control, 6),
        p_value=round(p_value, 6),
        statistic=round(t_stat, 6),
        confidence_interval=(round(ci[0], 6), round(ci[1], 6)),
        significant=p_value < alpha,
        alpha=alpha,
    )


def chi_square_test(
    contingency_table: np.ndarray,
    alpha: float = 0.05,
) -> TestResult:
    """
    Chi-square test for categorical outcomes with more than 2 variants.

    Parameters
    ----------
    contingency_table : np.ndarray
        2D array of observed frequencies, shape (n_variants, n_outcomes)
    alpha : float
        Significance level

    Returns
    -------
    TestResult
    """
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    row_sums = contingency_table.sum(axis=1)
    rates = contingency_table[:, 0] / row_sums
    
    return TestResult(
        test_name="Chi-Square Test",
        control_mean=round(rates[0], 6),
        treatment_mean=round(rates[1], 6),
        absolute_effect=round(rates[1] - rates[0], 6),
        relative_effect=round((rates[1] - rates[0]) / rates[0], 6),
        p_value=round(p_value, 6),
        statistic=round(chi2, 6),
        confidence_interval=(float("nan"), float("nan")),
        significant=p_value < alpha,
        alpha=alpha,
    )
