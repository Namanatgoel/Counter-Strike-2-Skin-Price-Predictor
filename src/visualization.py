"""
visualization.py — Interactive Visualization Module
====================================================

Purpose
-------
This module produces three interactive HTML charts using Plotly, a Python
library that generates JavaScript-powered visualizations. Unlike static
images (PNG/SVG), these charts are fully interactive — the user can hover
for exact values, zoom in on specific date ranges, toggle series on/off
by clicking the legend, and pan across the timeline.

All charts are saved as standalone `.html` files that open in any web
browser with no additional software required.

Why Plotly?
-----------
- **Interactive**: Hovering, zooming, panning, clicking — out of the box.
- **Standalone HTML**: No server required; the chart is entirely self-contained.
- **Professional styling**: `template='plotly_white'` gives a clean, publication-
  quality look with minimal configuration.
- **Python-native API**: `go.Figure()` and `go.Scatter()` map directly onto
  mental models from financial charting (OHLC, shaded areas, dual axes).

Three Visualizations Produced
------------------------------
1. `forecast_intervals.html` — The primary output chart.
   Shows actual historical prices alongside the probabilistic forecast:
   median prediction (P50), upper bound (P95), and lower bound (P5) as
   a shaded confidence band.

2. `shap_summary.html` — Model explainability chart.
   Shows which features most influenced the model's predictions (SHAP values).
   Each dot is one training observation; color = feature value, x-axis = impact.

3. `volatility_regimes.html` — Risk visualization.
   Overlays the item's price and its 14-day rolling volatility on dual axes,
   with red-shaded regions highlighting periods of abnormally high volatility
   (top 10% of all observed volatility values).

Dependencies
------------
    pandas   — DataFrame indexing for chart data
    numpy    — Array operations for SHAP value processing
    plotly   — Interactive charting library (go.Figure, go.Scatter, etc.)
    shap     — SHAP (SHapley Additive exPlanations) value computation
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap


def plot_forecast_intervals(
    df_actual: pd.DataFrame,
    df_pred: pd.DataFrame,
    item_name: str,
    output_path: str = None
) -> go.Figure:
    """
    Produces the primary forecast visualization: actual historical prices
    overlaid with the model's probabilistic prediction band.

    Chart Components
    ----------------
    - **Black solid line**: Actual observed spot prices over the historical window.
    - **Blue dashed line**: P50 (median) prediction — the model's best guess
      for each future date.
    - **Blue shaded region**: The 90% prediction interval [P5, P95].
      Where this band is narrow, the model is confident. Where it widens,
      the model is uncertain (typically corresponds to high-volatility periods).

    How the Confidence Band is Rendered
    -------------------------------------
    Plotly doesn't have a built-in "shaded area between two lines" primitive.
    Instead, we use the `fill='tonexty'` trick:
    1. Add the P95 line as a trace with invisible color and `showlegend=False`.
    2. Add the P5 line as a trace with `fill='tonexty'`, which fills the area
       BETWEEN this trace and the PREVIOUS trace (the P95 line).
    3. Set `fillcolor='rgba(0,0,255,0.2)'` — blue at 20% opacity.

    Parameters
    ----------
    df_actual : pd.DataFrame
        Historical price data. Must have a DatetimeIndex and a 'spot_price' column.
        Typically the full history of the test item.

    df_pred : pd.DataFrame
        Prediction DataFrame as returned by `QuantileInferenceEngine.predict_with_confidence()`.
        Must have columns: 'P5', 'Prediction_P50', 'P95'.
        The index should be shifted forward by the forecast horizon (done in main.py)
        so predictions plot on their corresponding future dates.

    item_name : str
        Used in the chart title to identify which CS2 skin is being displayed.
        Example: "AK-47 | Redline (Field-Tested)"

    output_path : str or None
        If provided, the figure is saved as an HTML file at this path.
        If None, the figure is returned but not saved (useful for notebook display).

    Returns
    -------
    plotly.graph_objects.Figure
        The complete figure object. Can be shown with `.show()` in Jupyter.
    """
    fig = go.Figure()

    # --- Trace 1: Actual historical price (black, solid, prominent) ---
    fig.add_trace(go.Scatter(
        x=df_actual.index,
        y=df_actual['spot_price'],
        mode='lines',
        name='Actual Price',
        line=dict(color='black', width=2)
    ))

    # --- Trace 2: P50 median forecast (blue, dashed) ---
    # This is the primary prediction — the model's best single estimate
    fig.add_trace(go.Scatter(
        x=df_pred.index,
        y=df_pred['Prediction_P50'],
        mode='lines',
        name='P50 (Median Forecast)',
        line=dict(color='blue', width=2, dash='dash')
    ))

    # --- Trace 3: P95 upper bound (invisible line, used as fill ceiling) ---
    # `line_color='rgba(0,0,255,0)'` = fully transparent; this trace is invisible
    # but necessary as the upper anchor for the fill operation in Trace 4.
    fig.add_trace(go.Scatter(
        x=df_pred.index,
        y=df_pred['P95'],
        fill=None,                          # no fill on this trace
        mode='lines',
        line_color='rgba(0,0,255,0)',        # transparent line
        showlegend=False                     # don't clutter the legend
    ))

    # --- Trace 4: P5 lower bound (fills UP to the P95 trace above) ---
    # `fill='tonexty'` fills the area between THIS trace and the PREVIOUS one.
    # Since the previous trace is P95, this creates the [P5, P95] confidence band.
    fig.add_trace(go.Scatter(
        x=df_pred.index,
        y=df_pred['P5'],
        fill='tonexty',                      # fill area between this and P95 trace
        mode='lines',
        line_color='rgba(0,0,255,0)',         # transparent line (band boundary hidden)
        name='P5–P95 Confidence Band',
        fillcolor='rgba(0,0,255,0.2)'        # semi-transparent blue fill
    ))

    # --- Layout configuration ---
    fig.update_layout(
        title=f'Price Forecast with Confidence Intervals — {item_name}',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',     # show all traces' values on hover at same x
        template='plotly_white'    # clean white background, minimal gridlines
    )

    # Save to HTML if a path was provided
    if output_path:
        fig.write_html(output_path)

    return fig


def plot_shap_summary(
    model,
    X_train: pd.DataFrame,
    output_path: str = None
) -> go.Figure:
    """
    Produces a SHAP summary plot showing which features drive the model's
    predictions and in which direction.

    What are SHAP Values?
    ---------------------
    SHAP (SHapley Additive exPlanations) is a game-theoretic approach to
    explaining individual predictions. For each observation and each feature,
    SHAP computes: "How much did this feature value push the prediction above
    or below the average prediction?"

    - **Positive SHAP value** for a feature → that feature's value on this
      observation pushed the prediction *higher* (more bullish).
    - **Negative SHAP value** → pushed the prediction *lower* (more bearish).
    - **SHAP value near 0** → this feature had little effect on this prediction.

    The summary plot stacks all training observations' SHAP values for the
    top 15 most impactful features, creating a "bee swarm" pattern where:
    - X-axis = SHAP value (impact on prediction)
    - Y-axis = feature name
    - Color = raw feature value (Viridis scale: purple=low, yellow=high)

    Example interpretation:
    "rsi_14 has many dots on the right side colored yellow (high RSI values),
    meaning high RSI strongly pushed up price predictions — consistent with
    overbought conditions driving expected price increases."

    Parameters
    ----------
    model : LGBMRegressor (or other TreeExplainer-compatible model)
        The fitted model to explain. Must be a tree-based model because
        `shap.TreeExplainer` uses the tree structure for exact, fast computation.
        For non-tree models, use `shap.KernelExplainer` (much slower).

    X_train : pd.DataFrame
        Training features. SHAP values are computed for each row in X_train.
        For large datasets, consider using a random sample (e.g., X_train.sample(500))
        to keep computation fast.

    output_path : str or None
        Path to save the HTML file. If None, figure is returned but not saved.

    Returns
    -------
    plotly.graph_objects.Figure

    Notes on TreeExplainer
    ----------------------
    `shap.TreeExplainer(model)` analyzes the model's actual tree structures
    to compute exact SHAP values in O(T × D × 2^D) time, where T = trees
    and D = max depth. For gradient boosting models with 100–300 trees at
    depth 8, this is extremely fast (seconds vs minutes for KernelExplainer).
    """
    # --- Compute SHAP values ---
    # TreeExplainer uses the tree structure for exact computation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    # For multi-output models, shap_values is a list; take the first output
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # --- Select top 15 features by mean absolute SHAP value ---
    # Mean |SHAP| across all observations = "global importance" of each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # argsort returns indices from smallest to largest; [::-1] reverses to largest first
    top_indices = np.argsort(mean_abs_shap)[-15:][::-1]

    # --- Build the bee-swarm scatter plot ---
    fig = go.Figure()

    for i, feature_idx in enumerate(top_indices):
        feature_name = X_train.columns[feature_idx]
        shap_vals    = shap_values[:, feature_idx]          # SHAP values for this feature
        feature_vals = X_train.iloc[:, feature_idx].values  # raw feature values (for coloring)

        fig.add_trace(go.Scatter(
            x=shap_vals,                         # horizontal position = prediction impact
            y=[feature_name] * len(shap_vals),   # all dots for this feature at same y
            mode='markers',
            name=feature_name,
            marker=dict(
                size=5,
                color=feature_vals,              # color encodes the feature's raw value
                colorscale='Viridis',            # purple=low, yellow=high
                showscale=(i == 0)               # only show the colorbar for the first trace
            ),
            hovertemplate=(
                f'<b>{feature_name}</b><br>'
                'SHAP: %{x:.4f}<br>'
                'Feature Value: %{marker.color:.4f}'
                '<extra></extra>'
            )
        ))

    # --- Layout ---
    fig.update_layout(
        title='SHAP Summary Plot — Top 15 Most Influential Features',
        xaxis_title='SHAP Value (Impact on Prediction)',
        yaxis_title='Feature',
        hovermode='closest',
        template='plotly_white',
        showlegend=False   # legend is redundant since features are on y-axis
    )

    if output_path:
        fig.write_html(output_path)

    return fig


def plot_volatility_regimes(
    df_item: pd.DataFrame,
    output_path: str = None
) -> go.Figure:
    """
    Produces a dual-axis chart overlaying price and volatility, with
    red-shaded "high volatility regime" regions.

    This chart helps answer: "When did this item experience abnormal price
    instability, and how did price behave during those periods?"

    High-volatility regimes (top 10% of observed 14-day volatility) are
    shaded in red. These correspond to periods of:
    - Pump-and-dump cycles (rapid price inflation followed by crash)
    - Major CS2 updates or case releases affecting skin demand
    - Thin liquidity creating erratic price movements

    Chart Components
    ----------------
    - **Left Y-axis (blue)**: Spot price over time.
    - **Right Y-axis (orange)**: 14-day rolling volatility.
    - **Red shaded rectangles**: Periods where volatility exceeded the
      90th percentile of all observed volatility (top 10% most volatile days).

    Dual-Axis Design Rationale
    --------------------------
    Price and volatility have entirely different scales. A price of $150 and
    a volatility of 0.04 cannot share the same Y-axis without one of them
    being completely invisible. Plotly's `make_subplots(specs=[[{"secondary_y": True}]])`
    creates a single chart with two independent Y-axes sharing the same X-axis.

    High Volatility Period Detection Algorithm
    ------------------------------------------
    1. Compute the 90th percentile threshold of `volatility_14`.
    2. Create a boolean mask: True where volatility > threshold.
    3. Find where the mask transitions from False→True (starts) using `.diff() == 1`.
    4. Find where mask transitions from True→False (ends) using `.diff() == -1`.
    5. Handle edge cases: if volatility is high at the very first row, manually
       prepend the first timestamp as a "start". Same for the last row.
    6. Draw one `add_vrect()` (vertical rectangle) for each [start, end] pair.

    Parameters
    ----------
    df_item : pd.DataFrame
        DataFrame for a single item. Must have a DatetimeIndex and columns:
        - 'spot_price'    : Daily closing price.
        - 'volatility_14' : 14-day rolling std of daily returns (from features.py).

    output_path : str or None
        Path to save HTML. If None, figure is returned without saving.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    # Create figure with secondary Y-axis support
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # --- Primary axis: spot price ---
    fig.add_trace(
        go.Scatter(
            x=df_item.index,
            y=df_item['spot_price'],
            mode='lines',
            name='Spot Price',
            line=dict(color='darkblue', width=2)
        ),
        secondary_y=False   # maps to left Y-axis
    )

    # --- Secondary axis: 14-day rolling volatility ---
    fig.add_trace(
        go.Scatter(
            x=df_item.index,
            y=df_item['volatility_14'],
            mode='lines',
            name='14-Day Volatility',
            line=dict(color='orange', width=2)
        ),
        secondary_y=True    # maps to right Y-axis
    )

    # --- Red-shaded high-volatility regime detection ---
    if 'volatility_14' in df_item.columns:
        # Threshold: 90th percentile of all observed volatility values
        # (top 10% most volatile periods)
        vol_90 = df_item['volatility_14'].quantile(0.90)

        # Boolean mask: True on high-volatility days
        high_vol_mask = df_item['volatility_14'] > vol_90

        # Detect transitions in the boolean mask
        # .diff() on int gives +1 at rising edges and -1 at falling edges
        changes = high_vol_mask.astype(int).diff()
        starts  = df_item.index[changes == 1].tolist()   # high-vol period begins
        ends    = df_item.index[changes == -1].tolist()  # high-vol period ends

        # Edge case: if first row is already high-volatility, it has no rising edge
        if high_vol_mask.iloc[0]:
            starts.insert(0, df_item.index[0])

        # Edge case: if last row is high-volatility, there's no falling edge
        if high_vol_mask.iloc[-1]:
            ends.append(df_item.index[-1])

        # Draw one vertical rectangle for each contiguous high-volatility period
        for start, end in zip(starts, ends):
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor='red',
                opacity=0.15,       # low opacity so price line remains readable
                layer='below',      # render behind traces
                line_width=0        # no border on the rectangle
            )

    # --- Layout and axis labels ---
    fig.update_layout(
        title='Price and Volatility Regimes (Red = Top 10% Volatility)',
        xaxis_title='Date',
        hovermode='x unified',   # show all traces' values at same x position
        template='plotly_white'
    )
    fig.update_yaxes(title_text='Spot Price (USD)',           secondary_y=False)
    fig.update_yaxes(title_text='14-Day Rolling Volatility',  secondary_y=True)

    if output_path:
        fig.write_html(output_path)

    return fig
