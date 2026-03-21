# abforge

> A Python package for the full A/B test lifecycle, from power analysis to sequential monitoring and variance reduction.

---

## Overview

abforge is a statistics library for designing and analyzing randomized controlled experiments. It was built to cover the parts of experiment design that most tutorials skip.

abforge covers four stages of an experiment. 
1. Power analysis determines the sample size needed before data collection starts, so you do not end up with an underpowered experiment or stop too early by accident.
2. Hypothesis testing evaluates whether an observed difference between control and treatment is statistically meaningful.
3. Sequential testing monitors results at planned intervals during the experiment without inflating the false positive rate, which is what happens when you peek at results repeatedly without a stopping rule.
4. CUPED variance reduction uses pre-experiment data to make the outcome metric less noisy, so you get more statistical power out of the same sample size.

The demo notebook applies all four stages to a real e-commerce dataset, framed around a concrete business question: does a promotional banner increase mobile conversion rates at the Google Merchandise Store?

---

## Why abforge?

When working on an experiment, running a t-test is the easy part. Researchers first need to know how many users to collect. During the experiment, checking results without inflating the false positive rate requires a stopping rule. Getting more out of noisy metrics without collecting more data is the next challenge. abforge covers each of these steps.

---

## Features

| Module | What it does |
|---|---|
| `power` | Sample size calculation, MDE estimation, power curves |
| `stats` | Two-proportion z-test, Welch's t-test, chi-square test |
| `sequential` | Sequential testing with O'Brien-Fleming and Pocock alpha spending |
| `cuped` | CUPED variance reduction using pre-experiment covariates |
| `viz` | Plotly visualizations for all of the above |

---

## Quickstart

```python
import abforge

# 1. How many users do I need?
n = abforge.sample_size(
    baseline_rate=0.041,        # current conversion rate
    min_detectable_effect=0.20, # detect 20% relative lift
    alpha=0.05,
    power=0.80,
)
print(f"Required sample size: {n:,} per variant")

# 2. Analyze results
result = abforge.proportions_test(
    control_conversions=417,  control_n=n,
    treatment_conversions=530, treatment_n=n,
)
print(result)

# 3. Monitor safely with sequential testing
test = abforge.SequentialTest(max_n=n, spending='obrien_fleming')
status = test.evaluate(
    control_conversions=417,  control_n=n // 2,
    treatment_conversions=530, treatment_n=n // 2,
)
print(status)

# 4. Reduce variance with CUPED
cuped_result = abforge.cuped(
    control_metric=control_revenue,
    treatment_metric=treatment_revenue,
    control_covariate=control_prior_revenue,
    treatment_covariate=treatment_prior_revenue,
)
print(cuped_result)
```

---

## Installation

```bash
git clone https://github.com/aditiputtur/abforge
cd abforge
pip install -r requirements.txt
```

---

## Demo Notebook

[`notebooks/ecommerce_ab_analysis.ipynb`](notebooks/ecommerce_ab_analysis.ipynb)

A full end-to-end analysis using the [Google Analytics Merchandise Store public dataset](https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=google_analytics_sample) covering 903,653 sessions from August 2016 through August 2017.

**Part 1. Consumer Behavior Analysis**

Half of all sessions bounce after a single page. Overall conversion sits at 1.28%. CPM traffic produces $21 in revenue per session while organic produces $1.03. Desktop users convert at 1.67% compared to 0.41% on mobile. Despite traffic from many countries, 93% of revenue comes from the United States. December shows a conversion spike and Thursday is the strongest day of the week.

**Part 2. A/B Test Simulation with abforge**

The power analysis found that 104,814 sessions per variant are needed to detect a 20% relative lift in mobile conversion. The z-test detected a 27.1% lift at p=0.0002. Pocock sequential boundaries were crossed at look 3 of 10. CUPED is demonstrated using pageviews as a pre-experiment covariate. A 20% mobile conversion lift translates to roughly $11,000 in additional annual revenue at current traffic levels.

---

## Module Reference

### `abforge.power`
```python
sample_size(baseline_rate, min_detectable_effect, alpha=0.05, power=0.80)
minimum_detectable_effect(baseline_rate, n, alpha=0.05, power=0.80)
power_curve(baseline_rate, effects, n, alpha=0.05)
```

### `abforge.stats`
```python
proportions_test(control_conversions, control_n,
                 treatment_conversions, treatment_n, alpha=0.05)
means_test(control_values, treatment_values, alpha=0.05)
chi_square_test(contingency_table, alpha=0.05)
```

### `abforge.sequential`
```python
SequentialTest(max_n, alpha=0.05, spending='obrien_fleming')
    .evaluate(control_conversions, control_n,
              treatment_conversions, treatment_n)
    .simulate(true_control_rate, true_treatment_rate, n_looks=10)
```

Spending functions available: `obrien_fleming`, `pocock`, `linear`

### `abforge.cuped`
```python
cuped(control_metric, treatment_metric,
      control_covariate, treatment_covariate, alpha=0.05)
check_covariate_quality(metric, covariate)
```

### `abforge.viz`
```python
plot_power_curve(baseline_rate, effects, n, alpha, title)
plot_test_result(result, title)
plot_sequential_boundaries(looks_data, title)
plot_cuped_comparison(cuped_result, title)
```

---

## Methodology Notes

Peeking at experiment results before they finish inflates false positives. Sequential testing with alpha spending functions lets you check results at planned intervals without that cost. abforge supports three spending functions: O'Brien-Fleming, which spends very little alpha early and is recommended for most experiments; Pocock, which applies constant boundaries at each look; and linear, which spends proportionally to information fraction.

CUPED removes variance explained by a pre-experiment covariate from the outcome metric. The stronger the correlation between the covariate and the metric, the more variance is removed, and the smaller the sample needed to reach the same power. In practice, prior-period values of the same metric work best as covariates.

> Deng, A., Xu, Y., Kohavi, R., and Walker, T. (2013). Improving the sensitivity of online controlled experiments by utilizing pre-experiment data. WSDM 2013.

---

## Repo Structure

```
abforge/
├── abforge/
│   ├── __init__.py
│   ├── power.py        # Sample size and power calculations
│   ├── stats.py        # Hypothesis tests
│   ├── sequential.py   # Sequential testing and alpha spending
│   ├── cuped.py        # CUPED variance reduction
│   └── viz.py          # Plotly visualizations
├── notebooks/
│   └── ecommerce_ab_analysis.ipynb
└── requirements.txt
```
