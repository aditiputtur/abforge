import numpy as np
from scipy import stats


def sample_size(
    baseline_rate: float,
    min_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_tailed: bool = True,
) -> int:
    """
    Calculate required sample size per variant for a proportions test.

    Parameters
    ----------
    baseline_rate : float
        Expected conversion rate in control (e.g. 0.10 for 10%)
    min_detectable_effect : float
        Minimum relative lift you want to detect (e.g. 0.05 for 5% lift)
    alpha : float
        Significance level (default 0.05)
    power : float
        Statistical power (default 0.80)
    two_tailed : bool
        Whether to use a two-tailed test (default True)

    Returns
    -------
    int
        Required sample size per variant
    
    Example
    -------
    >>> sample_size(baseline_rate=0.10, min_detectable_effect=0.05)
    31234
    """
    treatment_rate = baseline_rate * (1 + min_detectable_effect)
    
    if two_tailed:
        z_alpha = stats.norm.ppf(1 - alpha / 2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    
    z_beta = stats.norm.ppf(power)
    
    pooled_rate = (baseline_rate + treatment_rate) / 2
    
    n = (
        (z_alpha * np.sqrt(2 * pooled_rate * (1 - pooled_rate)) +
         z_beta * np.sqrt(baseline_rate * (1 - baseline_rate) +
                          treatment_rate * (1 - treatment_rate))) ** 2
    ) / (treatment_rate - baseline_rate) ** 2
    
    return int(np.ceil(n))


def minimum_detectable_effect(
    baseline_rate: float,
    n: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """
    Calculate the minimum detectable effect given a fixed sample size.

    Parameters
    ----------
    baseline_rate : float
        Expected conversion rate in control
    n : int
        Available sample size per variant
    alpha : float
        Significance level (default 0.05)
    power : float
        Statistical power (default 0.80)

    Returns
    -------
    float
        Minimum detectable absolute effect size
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    
    mde = (z_alpha + z_beta) * np.sqrt(
        2 * baseline_rate * (1 - baseline_rate) / n
    )
    
    return round(mde, 6)


def power_curve(
    baseline_rate: float,
    effects: list,
    n: int,
    alpha: float = 0.05,
) -> dict:
    """
    Calculate power at different effect sizes for a fixed sample size.
    Useful for plotting power curves.

    Parameters
    ----------
    baseline_rate : float
        Expected conversion rate in control
    effects : list
        List of absolute effect sizes to evaluate
    n : int
        Sample size per variant
    alpha : float
        Significance level

    Returns
    -------
    dict
        Mapping of effect size -> power
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    results = {}
    
    for effect in effects:
        treatment_rate = baseline_rate + effect
        se = np.sqrt(
            baseline_rate * (1 - baseline_rate) / n +
            treatment_rate * (1 - treatment_rate) / n
        )
        z = effect / se
        pwr = 1 - stats.norm.cdf(z_alpha - z) + stats.norm.cdf(-z_alpha - z)
        results[round(effect, 6)] = round(pwr, 4)
    
    return results
