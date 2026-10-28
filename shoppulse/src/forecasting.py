"""
ShopPulse - Demand Forecasting Module
Forecast monthly revenue and product demand using Prophet
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

PLOTLY_TEMPLATE = "plotly_dark"
ACCENT  = "#7C3AED"
ACCENT2 = "#06B6D4"
ACCENT3 = "#F59E0B"
ACCENT4 = "#10B981"


def prepare_monthly_ts(df: pd.DataFrame, freq: str = "ME") -> pd.DataFrame:
    """Prepare monthly time series for Prophet (ds, y columns)."""
    ts = df.resample(freq, on="InvoiceDate")["Revenue"].sum().reset_index()
    ts.columns = ["ds", "y"]
    ts = ts[ts["y"] > 0].copy()
    return ts


def prepare_product_ts(df: pd.DataFrame, product_desc: str, freq: str = "ME") -> pd.DataFrame:
    """Time series for a specific product."""
    subset = df[df["Description"].str.contains(product_desc, case=False, na=False)]
    if subset.empty:
        return pd.DataFrame(columns=["ds", "y"])
    ts = subset.resample(freq, on="InvoiceDate")["Quantity"].sum().reset_index()
    ts.columns = ["ds", "y"]
    ts = ts[ts["y"] > 0]
    return ts


def run_prophet_forecast(ts: pd.DataFrame, periods: int = 6) -> tuple:
    """
    Run Prophet forecast. Returns (model, forecast_df).
    Falls back to simple exponential smoothing if Prophet fails.
    """
    try:
        from prophet import Prophet
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.3,
            seasonality_prior_scale=10.0,
        )
        model.fit(ts)
        future   = model.make_future_dataframe(periods=periods, freq="ME")
        forecast = model.predict(future)
        return model, forecast
    except Exception as e:
        # Fallback: simple linear + noise extrapolation
        print(f"Prophet failed ({e}), using fallback forecasting...")
        return _fallback_forecast(ts, periods)


def _fallback_forecast(ts: pd.DataFrame, periods: int = 6):
    """Simple fallback: linear trend extrapolation."""
    x = np.arange(len(ts))
    y = ts["y"].values
    coeffs = np.polyfit(x, y, 1)
    poly   = np.poly1d(coeffs)

    last_date  = ts["ds"].iloc[-1]
    future_idx = np.arange(len(ts), len(ts) + periods)
    future_vals = poly(future_idx)
    future_vals = np.maximum(future_vals, 0)

    freq = pd.tseries.frequencies.to_offset("ME")
    future_dates = [last_date + freq * (i + 1) for i in range(periods)]

    hist_forecast = pd.DataFrame({
        "ds": ts["ds"],
        "yhat": poly(x),
        "yhat_lower": poly(x) * 0.85,
        "yhat_upper": poly(x) * 1.15,
        "trend": poly(x),
    })
    future_forecast = pd.DataFrame({
        "ds": future_dates,
        "yhat": future_vals,
        "yhat_lower": future_vals * 0.80,
        "yhat_upper": future_vals * 1.20,
        "trend": future_vals,
    })
    forecast = pd.concat([hist_forecast, future_forecast], ignore_index=True)
    return None, forecast


def plot_forecast(ts: pd.DataFrame, forecast: pd.DataFrame,
                  title: str = "📈 Revenue Forecast", unit: str = "£") -> go.Figure:
    """Plot historical data + forecast with confidence interval."""
    fig = go.Figure()

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
        y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
        fill="toself",
        fillcolor=f"rgba(124,58,237,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="Confidence Interval",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"],
        mode="lines",
        line=dict(color=ACCENT, width=3, dash="dash"),
        name="Forecast",
    ))

    # Actuals
    fig.add_trace(go.Scatter(
        x=ts["ds"], y=ts["y"],
        mode="lines+markers",
        line=dict(color=ACCENT2, width=2),
        marker=dict(size=6),
        name="Actual",
    ))

    # Vertical separator (last actual date)
    if not ts.empty:
        last = ts["ds"].iloc[-1]
        fig.add_vline(x=last, line_dash="dot", line_color="rgba(255,255,255,0.3)")

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=title,
        xaxis_title="Date",
        yaxis_title=f"Revenue ({unit})" if unit == "£" else f"Quantity ({unit})",
        legend=dict(orientation="h", y=1.02),
        hovermode="x unified",
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
    )
    return fig


def plot_trend_components(forecast: pd.DataFrame) -> go.Figure:
    """Plot trend component from Prophet forecast."""
    if "trend" not in forecast.columns:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["trend"],
        mode="lines",
        line=dict(color=ACCENT4, width=3),
        fill="tozeroy",
        fillcolor=f"rgba(16,185,129,0.1)",
        name="Trend",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="📉 Revenue Trend Component",
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
    )
    return fig


def get_forecast_summary(ts: pd.DataFrame, forecast: pd.DataFrame) -> dict:
    """Key forecast metrics."""
    hist_avg = ts["y"].mean() if not ts.empty else 0
    future   = forecast[forecast["ds"] > ts["ds"].max()] if not ts.empty else forecast
    forecast_avg = future["yhat"].mean() if not future.empty else 0
    growth   = ((forecast_avg - hist_avg) / hist_avg * 100) if hist_avg > 0 else 0
    return {
        "Historical Avg/Month": f"£{hist_avg:,.0f}",
        "Forecast Avg/Month":   f"£{forecast_avg:,.0f}",
        "Expected Growth":      f"{growth:+.1f}%",
        "Forecast Periods":     f"{len(future)} months",
    }


def get_top_products_for_forecast(df: pd.DataFrame, n: int = 20) -> list[str]:
    """Get top N products by revenue for forecast selection."""
    top = (df.groupby("Description")["Revenue"]
             .sum()
             .nlargest(n)
             .index
             .tolist())
    return top
