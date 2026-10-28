"""
ShopPulse - Market Basket Analysis
Apriori / FP-Growth: "Customers who bought X also bought Y"
"""
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px
import plotly.graph_objects as go

PLOTLY_TEMPLATE = "plotly_dark"
ACCENT  = "#7C3AED"
ACCENT2 = "#06B6D4"
ACCENT3 = "#F59E0B"


def build_basket_matrix(df: pd.DataFrame, country: str = "United Kingdom") -> pd.DataFrame:
    """Build invoice × product boolean matrix for the specified country."""
    subset = df if country == "All" else df[df["Country"] == country]

    basket = (subset
              .groupby(["InvoiceNo", "Description"])["Quantity"]
              .sum()
              .unstack(fill_value=0))

    # Encode: True/False
    basket = basket > 0
    return basket


def run_apriori(basket: pd.DataFrame,
                min_support: float = 0.02,
                min_confidence: float = 0.3,
                min_lift: float = 1.0,
                use_fpgrowth: bool = True) -> pd.DataFrame:
    """
    Run Apriori or FP-Growth and extract association rules.
    Returns a DataFrame of rules sorted by lift.
    """
    algo = fpgrowth if use_fpgrowth else apriori
    freq_items = algo(basket.astype(bool), min_support=min_support, use_colnames=True)

    if freq_items.empty:
        return pd.DataFrame()

    rules = association_rules(freq_items, metric="lift", min_threshold=min_lift)
    rules = rules[rules["confidence"] >= min_confidence]

    # Clean up frozensets
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    cols  = ["antecedents", "consequents", "support", "confidence", "lift", "leverage", "conviction"]
    return rules[[c for c in cols if c in rules.columns]]


# ─── Visualizations ──────────────────────────────────────────────────────────

def plot_rules_scatter(rules: pd.DataFrame) -> go.Figure:
    """Support vs Confidence scatter, sized by lift."""
    if rules.empty:
        fig = go.Figure()
        fig.update_layout(title="No rules found — try lower thresholds", template=PLOTLY_TEMPLATE)
        return fig

    fig = px.scatter(
        rules,
        x="support", y="confidence",
        size="lift", color="lift",
        color_continuous_scale=["#1E1B4B", ACCENT, ACCENT2],
        hover_data=["antecedents", "consequents", "lift"],
        title="🛒 Association Rules: Support vs Confidence",
        template=PLOTLY_TEMPLATE,
        labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"},
        size_max=20,
    )
    fig.update_layout(paper_bgcolor="#0F0F1A", plot_bgcolor="#0F0F1A")
    return fig


def plot_top_rules_bar(rules: pd.DataFrame, n: int = 15) -> go.Figure:
    """Bar chart of top N rules by lift."""
    if rules.empty:
        fig = go.Figure()
        fig.update_layout(title="No rules found", template=PLOTLY_TEMPLATE)
        return fig

    top  = rules.head(n).copy()
    top["rule"] = top["antecedents"] + " → " + top["consequents"]

    fig = px.bar(
        top.sort_values("lift"),
        x="lift", y="rule", orientation="h",
        color="confidence",
        color_continuous_scale=["#3B1F6A", ACCENT, ACCENT2],
        title=f"🔗 Top {n} Association Rules by Lift",
        template=PLOTLY_TEMPLATE,
        labels={"lift": "Lift", "rule": ""},
    )
    fig.update_layout(
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
        yaxis=dict(tickfont=dict(size=9)),
    )
    return fig


def plot_rules_heatmap(rules: pd.DataFrame, n: int = 12) -> go.Figure:
    """Heatmap of confidence between top antecedents and consequents."""
    if rules.empty:
        return go.Figure()

    top_rules = rules.head(n * 2)
    ants = top_rules["antecedents"].unique()[:n]
    cons = top_rules["consequents"].unique()[:n]

    matrix = pd.DataFrame(index=ants, columns=cons, dtype=float).fillna(0)
    for _, row in top_rules.iterrows():
        if row["antecedents"] in matrix.index and row["consequents"] in matrix.columns:
            matrix.loc[row["antecedents"], row["consequents"]] = row["confidence"]

    fig = go.Figure(data=go.Heatmap(
        z=matrix.values,
        x=list(cons),
        y=list(ants),
        colorscale=[[0, "#0F0F1A"], [0.5, ACCENT], [1, ACCENT2]],
        text=matrix.round(2).values,
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(title="Confidence"),
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="🔥 Rule Confidence Heatmap",
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
        xaxis=dict(tickangle=-30, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
    )
    return fig


def get_product_recommendations(rules: pd.DataFrame, product: str, top_n: int = 5) -> pd.DataFrame:
    """Get top_n product recommendations for a given product."""
    if rules.empty:
        return pd.DataFrame()
    mask = rules["antecedents"].str.contains(product, case=False, na=False)
    recs = rules[mask].head(top_n)[["consequents", "confidence", "lift", "support"]]
    return recs.reset_index(drop=True)
