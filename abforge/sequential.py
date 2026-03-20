import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional


@dataclass
class SequentialTestResult:
    """Container for sequential test results."""
    test_name: str
    sample_size: int
    control_mean: float
    treatment_mean: float
    absolute_effect: float
    relative_effect: float
    z_score: float
    upper_boundary: float
    lower_boundary: float
    decision: str  # "stop_significant", "stop_futile", "continue"
    alpha_spent: float

    def __repr__(self):
        decision_map = {
            "stop_significant": "🛑 STOP — Significant result detected",
            "stop_futile":      "🛑 STOP — Futile to continue",
            "continue":         "▶️  CONTINUE — No conclusion yet",
        }
        return (
            f"\nSequential Test — {decision_map[self.decision]}\n"
            f"{'─' * 50}\n"
            f"  Samples seen:      {self.sample_size:,}\n"
            f"  Control mean:      {self.control_mean:.4f}\n"
            f"  Treatment mean:    {self.treatment_mean:.4f}\n"
            f"  Absolute effect:   {self.absolute_effect:+.4f}\n"
            f"  Relative effect:   {self.relative_effect:+.2%}\n"
            f"  Z-score:           {self.z_score:.4f}\n"
            f"  Upper boundary:    {self.upper_boundary:.4f}\n"
            f"  Lower boundary:    {self.lower_boundary:.4f}\n"
            f"  Alpha spent:       {self.alpha_spent:.4f}\n"
        )


class AlphaSpendingFunction:
    """
    Alpha spending functions for sequential testing.
    Controls how the Type I error budget is spent over time.

    Available functions:
    - 'obrien_fleming' : Conservative early, liberal late (recommended)
    - 'pocock'         : Constant boundary (equal spending)
    - 'linear'         : Linearly increasing spending
    """

    @staticmethod
    def obrien_fleming(t: float, alpha: float) -> float:
        """
        O'Brien-Fleming spending function.
        Very conservative early in the experiment, 
        preserves power for later looks.
        
        Parameters
        ----------
        t : float
            Information fraction (0 to 1) — proportion of max sample seen
        alpha : float
            Total alpha budget
        """
        if t <= 0:
            return 0.0
        z = stats.norm.ppf(1 - alpha / 2)
        return 2 * (1 - stats.norm.cdf(z / np.sqrt(t)))

    @staticmethod
    def pocock(t: float, alpha: float) -> float:
        """
        Pocock spending function.
        Spends alpha at a constant rate — equal boundaries at each look.
        
        Parameters
        ----------
        t : float
            Information fraction (0 to 1)
        alpha : float
            Total alpha budget
        """
        if t <= 0:
            return 0.0
        return alpha * np.log(1 + (np.e - 1) * t)

    @staticmethod
    def linear(t: float, alpha: float) -> float:
        """
        Linear spending function.
        Spends alpha proportionally to information fraction.
        
        Parameters
        ----------
        t : float
            Information fraction (0 to 1)
        alpha : float
            Total alpha budget
        """
        return alpha * t


class SequentialTest:
    """
    Sequential hypothesis test with alpha spending.
    
    Allows peeking at results during an experiment without inflating 
    the false positive rate — a common mistake in naive A/B testing.
    
    Parameters
    ----------
    max_n : int
        Maximum planned sample size per variant
    alpha : float
        Total Type I error budget (default 0.05)
    beta : float
        Type II error budget / 1 - power (default 0.20)
    spending : str
        Alpha spending function: 'obrien_fleming', 'pocock', or 'linear'
    two_tailed : bool
        Whether to use two-tailed test (default True)
    
    Example
    -------
    >>> test = SequentialTest(max_n=10000, alpha=0.05, spending='obrien_fleming')
    >>> result = test.evaluate(
    ...     control_conversions=450,
    ...     control_n=5000,
    ...     treatment_conversions=490,
    ...     treatment_n=5000,
    ... )
    >>> print(result)
    """

    SPENDING_FUNCTIONS = {
        "obrien_fleming": AlphaSpendingFunction.obrien_fleming,
        "pocock": AlphaSpendingFunction.pocock,
        "linear": AlphaSpendingFunction.linear,
    }

    def __init__(
        self,
        max_n: int,
        alpha: float = 0.05,
        beta: float = 0.20,
        spending: str = "obrien_fleming",
        two_tailed: bool = True,
    ):
        if spending not in self.SPENDING_FUNCTIONS:
            raise ValueError(
                f"spending must be one of {list(self.SPENDING_FUNCTIONS.keys())}"
            )
        self.max_n = max_n
        self.alpha = alpha
        self.beta = beta
        self.spending_fn = self.SPENDING_FUNCTIONS[spending]
        self.two_tailed = two_tailed
        self.spending_name = spending

    def evaluate(
        self,
        control_conversions: int,
        control_n: int,
        treatment_conversions: int,
        treatment_n: int,
        futility_threshold: Optional[float] = 0.90,
    ) -> SequentialTestResult:
        """
        Evaluate whether to stop or continue the experiment.

        Parameters
        ----------
        control_conversions : int
            Conversions observed in control so far
        control_n : int
            Users seen in control so far
        treatment_conversions : int
            Conversions observed in treatment so far
        treatment_n : int
            Users seen in treatment so far
        futility_threshold : float, optional
            Conditional power below which to stop for futility (default 0.90)

        Returns
        -------
        SequentialTestResult
        """
        p_control = control_conversions / control_n
        p_treatment = treatment_conversions / treatment_n

        # Information fraction — how far through the experiment are we
        t = (control_n + treatment_n) / (2 * self.max_n)
        t = min(t, 1.0)

        # Alpha spent so far
        alpha_spent = self.spending_fn(t, self.alpha)

        # Compute boundaries
        if self.two_tailed:
            z_boundary = stats.norm.ppf(1 - alpha_spent / 2)
        else:
            z_boundary = stats.norm.ppf(1 - alpha_spent)

        # Z-score for current data
        p_pooled = (
            (control_conversions + treatment_conversions) /
            (control_n + treatment_n)
        )
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / control_n + 1 / treatment_n))

        if se == 0:
            z_score = 0.0
        else:
            z_score = (p_treatment - p_control) / se

        diff = p_treatment - p_control

        # Decision logic
        if abs(z_score) >= z_boundary:
            decision = "stop_significant"
        elif futility_threshold is not None:
            # Conditional power — probability of significance given current trend
            remaining_n = self.max_n - max(control_n, treatment_n)
            if remaining_n > 0:
                conditional_power = self._conditional_power(
                    z_score, t, self.alpha, self.beta
                )
                if conditional_power < (1 - futility_threshold):
                    decision = "stop_futile"
                else:
                    decision = "continue"
            else:
                decision = "continue"
        else:
            decision = "continue"

        return SequentialTestResult(
            test_name=f"Sequential Test ({self.spending_name})",
            sample_size=control_n + treatment_n,
            control_mean=round(p_control, 6),
            treatment_mean=round(p_treatment, 6),
            absolute_effect=round(diff, 6),
            relative_effect=round(diff / p_control, 6) if p_control > 0 else 0.0,
            z_score=round(z_score, 6),
            upper_boundary=round(z_boundary, 6),
            lower_boundary=round(-z_boundary, 6),
            decision=decision,
            alpha_spent=round(alpha_spent, 6),
        )

    def _conditional_power(
        self,
        z_current: float,
        t: float,
        alpha: float,
        beta: float,
    ) -> float:
        """
        Estimate conditional power given current z-score and 
        information fraction.
        """
        if t >= 1.0:
            return float(abs(z_current) >= stats.norm.ppf(1 - alpha / 2))
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        drift = z_current * np.sqrt(t)
        remaining = np.sqrt(1 - t)
        cp = 1 - stats.norm.cdf(
            (z_alpha - drift) / remaining
        )
        return cp

    def simulate(
        self,
        true_control_rate: float,
        true_treatment_rate: float,
        n_looks: int = 10,
        seed: int = 42,
    ) -> list:
        """
        Simulate an experiment to see how sequential boundaries behave.
        Useful for visualizing with viz.py.

        Parameters
        ----------
        true_control_rate : float
            True conversion rate for control
        true_treatment_rate : float
            True conversion rate for treatment
        n_looks : int
            Number of interim analyses (default 10)
        seed : int
            Random seed for reproducibility

        Returns
        -------
        list of SequentialTestResult at each look
        """
        rng = np.random.default_rng(seed)
        look_sizes = np.linspace(
            self.max_n // n_looks, self.max_n, n_looks, dtype=int
        )
        results = []

        for n in look_sizes:
            control_conv = int(rng.binomial(n, true_control_rate))
            treatment_conv = int(rng.binomial(n, true_treatment_rate))
            result = self.evaluate(
                control_conversions=control_conv,
                control_n=n,
                treatment_conversions=treatment_conv,
                treatment_n=n,
            )
            results.append(result)
            if result.decision != "continue":
                break

        return results
