import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Optional


def plot_power_curve(
    baseline_rate: float,
    effects: list,
    n: int,
    alpha: float = 0.05,
    title: str = "Power Curve",
) -> go.Figure:
    """
    Plot statistical power across a range of effect sizes.

    Parameters
    ----------
    baseline_rate : float
        Control conversion rate
    effects : list
        List of absolute effect sizes to evaluate
    n : int
        Sample size per variant
    alpha : float
        Significance level
    title : str
        Chart title
    """
    from .power import power_curve, minimum_detectable_effect

    powers = power_curve(baseline_rate, effects, n, alpha)
    mde = minimum_detectable_effect(baseline_rate, n, alpha)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[e * 100 for e in powers.keys()],
        y=list(powers.values()),
        mode='lines+markers',
        line=dict(color='#636EFA', width=2),
        marker=dict(size=4),
        name='Power'
    ))
    fig.add_hline(
        y=0.80, line_dash='dash', line_color='#EF553B',
        annotation_text='80% power threshold'
    )
    fig.add_vline(
        x=mde * 100, line_dash='dash', line_color='#00CC96',
        annotation_text=f'MDE = {mde:.3%}'
    )
    fig.update_layout(
        title=title,
        xaxis_title='Absolute Effect Size (%)',
        yaxis_title='Statistical Power',
        yaxis_tickformat='.0%',
        title_font_size=18,
        height=450,
        template='plotly_white',
    )
    return fig


def plot_test_result(
    result,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Visualize a TestResult with effect size and confidence interval.

    Parameters
    ----------
    result : TestResult
        Output from proportions_test or means_test
    title : str, optional
        Chart title
    """
    title = title or result.test_name
    color = '#00CC96' if result.significant else '#EF553B'
    ci_low, ci_high = result.confidence_interval

    fig = go.Figure()

    # Confidence interval bar
    fig.add_trace(go.Scatter(
        x=[ci_low, ci_high],
        y=['Effect'],
        mode='lines',
        line=dict(color=color, width=4),
        name='95% CI',
        showlegend=True,
    ))

    # Point estimate
    fig.add_trace(go.Scatter(
        x=[result.absolute_effect],
        y=['Effect'],
        mode='markers',
        marker=dict(color=color, size=14, symbol='diamond'),
        name='Absolute effect',
    ))

    # Zero line
    fig.add_vline(x=0, line_dash='dash', line_color='gray', opacity=0.7)

    sig_text = "Significant" if result.significant else "Not significant"
    fig.update_layout(
        title=f"{title}<br><sup>{sig_text} | p={result.p_value:.4f} | "
              f"Relative effect: {result.relative_effect:.2%}</sup>",
        xaxis_title='Absolute Effect Size',
        height=300,
        title_font_size=16,
        template='plotly_white',
    )
    return fig


def plot_sequential_boundaries(
    looks_data: list,
    title: str = "Sequential Test Boundaries",
) -> go.Figure:
    """
    Visualize sequential test z-scores against stopping boundaries.

    Parameters
    ----------
    looks_data : list
        List of SequentialTestResult objects from manual evaluation loop
    title : str
        Chart title
    """
    looks = list(range(1, len(looks_data) + 1))
    z_scores = [r.z_score for r in looks_data]
    upper_bounds = [r.upper_boundary for r in looks_data]
    lower_bounds = [r.lower_boundary for r in looks_data]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=looks, y=upper_bounds,
        mode='lines', name='Upper boundary',
        line=dict(color='#EF553B', dash='dash', width=2),
    ))
    fig.add_trace(go.Scatter(
        x=looks, y=lower_bounds,
        mode='lines', name='Lower boundary',
        line=dict(color='#EF553B', dash='dash', width=2),
        fill='tonexty', fillcolor='rgba(239,85,59,0.1)',
    ))
    fig.add_trace(go.Scatter(
        x=looks, y=z_scores,
        mode='lines+markers', name='Z-score',
        line=dict(color='#636EFA', width=2),
        marker=dict(size=8),
    ))

    fig.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.5)

    # Mark first boundary crossing
    for i, (z, ub, lb) in enumerate(zip(z_scores, upper_bounds, lower_bounds)):
        if z >= ub or z <= lb:
            fig.add_vline(
                x=i + 1,
                line_dash='dot',
                line_color='#00CC96',
                annotation_text="Boundary crossed",
            )
            break

    fig.update_layout(
        title=title,
        xaxis_title='Interim Look #',
        yaxis_title='Z-score',
        title_font_size=18,
        height=450,
        template='plotly_white',
    )
    return fig


def plot_cuped_comparison(
    cuped_result,
    title: str = "CUPED Variance Reduction",
) -> go.Figure:
    """
    Compare p-values and effect estimates before and after CUPED.

    Parameters
    ----------
    cuped_result : CUPEDResult
        Output from cuped()
    title : str
        Chart title
    """
    labels = ['Without CUPED', 'With CUPED']
    p_values = [
        cuped_result.original_result.p_value,
        cuped_result.adjusted_result.p_value,
    ]
    colors = [
        '#00CC96' if cuped_result.original_result.significant else '#EF553B',
        '#00CC96' if cuped_result.adjusted_result.significant else '#EF553B',
    ]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('p-value Comparison', 'Variance Reduction')
    )

    fig.add_trace(go.Bar(
        x=labels,
        y=p_values,
        marker_color=colors,
        text=[f"{p:.4f}" for p in p_values],
        textposition='outside',
        showlegend=False,
    ), row=1, col=1)

    fig.add_hline(
        y=0.05, line_dash='dash', line_color='gray',
        annotation_text='α = 0.05',
        row=1, col=1,
    )

    fig.add_trace(go.Bar(
        x=['Variance Reduction'],
        y=[cuped_result.variance_reduction * 100],
        marker_color='#636EFA',
        text=[f"{cuped_result.variance_reduction:.1%}"],
        textposition='outside',
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        title=f"{title}<br><sup>θ = {cuped_result.theta:.4f} | "
              f"Covariate explains {cuped_result.variance_reduction:.1%} of variance</sup>",
        title_font_size=16,
        height=400,
        template='plotly_white',
    )
    return fig
