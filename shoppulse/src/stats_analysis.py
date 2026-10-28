"""
ShopPulse - Statistical Analysis Module
Descriptive stats: price/quantity/revenue, correlation, revenue by country/month, top products
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


PLOTLY_TEMPLATE = "plotly_dark"
ACCENT  = "#7C3AED"   # Purple
ACCENT2 = "#06B6D4"   # Cyan
ACCENT3 = "#F59E0B"   # Amber
ACCENT4 = "#10B981"   # Emerald
PALETTE = [ACCENT, ACCENT2, ACCENT3, ACCENT4, "#EF4444", "#F97316", "#8B5CF6", "#EC4899"]


# ─── Descriptive Statistics ─────────────────────────────────────────────────

def compute_descriptive_stats(df: pd.DataFrame) -> dict:
    """Return a dict of summary statistics."""
    total_revenue = df["Revenue"].sum()
    avg_order_value = df.groupby("InvoiceNo")["Revenue"].sum().mean()
    return {
        "Total Revenue (£)":        f"£{total_revenue:,.2f}",
        "Total Transactions":       f"{df['InvoiceNo'].nunique():,}",
        "Total Customers":          f"{df['CustomerID'].nunique():,}",
        "Total Products":           f"{df['StockCode'].nunique():,}",
        "Total Countries":          f"{df['Country'].nunique():,}",
        "Avg Order Value (£)":      f"£{avg_order_value:,.2f}",
        "Avg Quantity per Line":    f"{df['Quantity'].mean():.1f}",
        "Median Unit Price (£)":    f"£{df['UnitPrice'].median():.2f}",
        "Date Range":               f"{df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}",
    }


def plot_revenue_over_time(df: pd.DataFrame) -> go.Figure:
    """Monthly revenue trend with rolling average."""
    monthly = df.groupby("YearMonth")["Revenue"].sum().reset_index()
    monthly["YearMonth"] = monthly["YearMonth"].astype(str)
    monthly["Rolling3M"] = monthly["Revenue"].rolling(3, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly["YearMonth"], y=monthly["Revenue"],
        name="Monthly Revenue",
        marker_color=ACCENT,
        opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=monthly["YearMonth"], y=monthly["Rolling3M"],
        name="3-Month Rolling Avg",
        line=dict(color=ACCENT2, width=3),
        mode="lines+markers",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="📈 Monthly Revenue Trend",
        xaxis_title="Month",
        yaxis_title="Revenue (£)",
        legend=dict(orientation="h", y=1.02),
        hovermode="x unified",
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
    )
    return fig


def plot_top_products(df: pd.DataFrame, n: int = 15) -> go.Figure:
    """Top N products by revenue."""
    top = (df.groupby("Description")["Revenue"]
             .sum()
             .nlargest(n)
             .reset_index()
             .sort_values("Revenue"))
    fig = px.bar(
        top, x="Revenue", y="Description", orientation="h",
        color="Revenue", color_continuous_scale=["#3B1F6A", ACCENT, ACCENT2],
        title=f"🏆 Top {n} Products by Revenue",
        template=PLOTLY_TEMPLATE,
        labels={"Revenue": "Revenue (£)", "Description": ""},
    )
    fig.update_layout(
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
        coloraxis_showscale=False,
        yaxis=dict(tickfont=dict(size=10)),
    )
    return fig


def plot_revenue_by_country(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Revenue breakdown by country (excluding UK for a zoomed view)."""
    by_country = df.groupby("Country")["Revenue"].sum().nlargest(top_n).reset_index()
    fig = px.pie(
        by_country, values="Revenue", names="Country",
        title=f"🌍 Revenue Share — Top {top_n} Countries",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=PALETTE,
        hole=0.45,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(paper_bgcolor="#0F0F1A", showlegend=True)
    return fig


def plot_hourly_and_daily(df: pd.DataFrame) -> go.Figure:
    """Orders by hour of day and day of week."""
    hour_counts = df.groupby("Hour")["InvoiceNo"].nunique().reset_index()
    hour_counts.columns = ["Hour", "Orders"]

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_counts = (df.groupby("DayOfWeek")["InvoiceNo"].nunique()
                    .reindex(day_order).reset_index())
    day_counts.columns = ["Day", "Orders"]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Orders by Hour of Day", "Orders by Day of Week"),
    )
    fig.add_trace(go.Bar(x=hour_counts["Hour"], y=hour_counts["Orders"],
                         marker_color=ACCENT2, name="By Hour"), row=1, col=1)
    fig.add_trace(go.Bar(x=day_counts["Day"], y=day_counts["Orders"],
                         marker_color=ACCENT3, name="By Day"), row=1, col=2)
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="🕐 Order Timing Patterns",
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
        showlegend=False,
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation heatmap for numeric features."""
    numeric_cols = ["Quantity", "UnitPrice", "Revenue"]
    corr = df[numeric_cols].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=[[0, "#1E1B4B"], [0.5, ACCENT], [1, ACCENT2]],
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="🔗 Correlation Heatmap",
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
    )
    return fig


def plot_price_distribution(df: pd.DataFrame) -> go.Figure:
    """Unit price distribution (log scale)."""
    prices = df[df["UnitPrice"] < df["UnitPrice"].quantile(0.99)]["UnitPrice"]
    fig = px.histogram(
        prices, nbins=60,
        title="💰 Unit Price Distribution",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[ACCENT],
        labels={"value": "Unit Price (£)", "count": "Count"},
    )
    fig.update_layout(paper_bgcolor="#0F0F1A", plot_bgcolor="#0F0F1A")
    return fig


def plot_quantity_distribution(df: pd.DataFrame) -> go.Figure:
    """Quantity per line distribution."""
    qty = df[df["Quantity"] < df["Quantity"].quantile(0.98)]["Quantity"]
    fig = px.histogram(
        qty, nbins=50,
        title="📦 Quantity per Line Distribution",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=[ACCENT2],
        labels={"value": "Quantity", "count": "Count"},
    )
    fig.update_layout(paper_bgcolor="#0F0F1A", plot_bgcolor="#0F0F1A")
    return fig


def get_underperforming_products(df: pd.DataFrame, bottom_n: int = 15) -> pd.DataFrame:
    """Bottom N products by revenue with low purchase frequency."""
    prod_stats = df.groupby("Description").agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique"),
        AvgPrice=("UnitPrice", "mean"),
    ).reset_index()
    # Only products bought by at least 2 invoices
    prod_stats = prod_stats[prod_stats["Orders"] >= 2]
    return prod_stats.nsmallest(bottom_n, "Revenue").reset_index(drop=True)
