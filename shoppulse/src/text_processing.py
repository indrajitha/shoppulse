"""
ShopPulse - Text Processing Module
Product description word frequency, category extraction, keyword clustering
"""
import re
import pandas as pd
import numpy as np
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go

PLOTLY_TEMPLATE = "plotly_dark"
ACCENT  = "#7C3AED"
ACCENT2 = "#06B6D4"
ACCENT3 = "#F59E0B"

# Common stop words for product descriptions
STOP_WORDS = {
    "set", "of", "the", "and", "in", "a", "to", "for", "with", "large",
    "small", "mini", "big", "pack", "box", "bag", "tin", "red", "blue",
    "green", "white", "black", "pink", "purple", "yellow", "orange",
    "vintage", "retro", "design", "pattern", "print", "new", "old",
    "cm", "x", "mm", "or", "no", "on", "an", "is", "it", "be",
}

# Keyword → Category mapping
CATEGORY_MAP = {
    "KITCHEN":    ["mug", "cup", "plate", "bowl", "jar", "jug", "kettle", "cake", "baking", "kitchen", "popcorn", "egg"],
    "HOME DECOR": ["candle", "lantern", "frame", "light", "holder", "vase", "cushion", "mirror", "clock", "wreath"],
    "STORAGE":    ["bag", "box", "tin", "basket", "tote", "storage", "jumbo", "bin", "container"],
    "CRAFT":      ["ribbon", "button", "felt", "sewing", "craft", "knitting", "wool"],
    "SEASONAL":   ["christmas", "halloween", "easter", "birthday", "party", "bunting", "balloon", "decoration"],
    "GIFT":       ["heart", "love", "gift", "present", "sweet", "charm"],
    "STATIONERY": ["pencil", "pen", "notebook", "pad", "card", "book", "stamp"],
    "CLOTHING":   ["apron", "gloves", "scarf", "bag", "tote", "warmer"],
    "GARDEN":     ["bird", "flower", "garden", "outdoor", "plant"],
    "TOYS":       ["doll", "game", "toy", "puzzle", "ball", "children"],
}


def tokenize_descriptions(df: pd.DataFrame) -> list[str]:
    """Clean and tokenize all product descriptions."""
    descriptions = df["Description"].dropna().unique()
    tokens = []
    for desc in descriptions:
        words = re.findall(r"[a-zA-Z]+", str(desc).lower())
        tokens.extend([w for w in words if w not in STOP_WORDS and len(w) > 2])
    return tokens


def get_word_frequencies(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Top N words in product descriptions."""
    tokens = tokenize_descriptions(df)
    counts = Counter(tokens)
    df_freq = pd.DataFrame(counts.most_common(top_n), columns=["Word", "Count"])
    return df_freq


def assign_category(description: str) -> str:
    """Assign a product category based on keyword matching."""
    desc_lower = str(description).lower()
    for category, keywords in CATEGORY_MAP.items():
        if any(kw in desc_lower for kw in keywords):
            return category
    return "OTHER"


def add_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'Category' column to the dataframe."""
    df = df.copy()
    df["Category"] = df["Description"].apply(assign_category)
    return df


def get_category_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue breakdown by inferred product category."""
    df_cat = add_categories(df)
    cat_rev = df_cat.groupby("Category").agg(
        Revenue  = ("Revenue",   "sum"),
        Orders   = ("InvoiceNo", "nunique"),
        Products = ("StockCode", "nunique"),
        Quantity = ("Quantity",  "sum"),
    ).reset_index().sort_values("Revenue", ascending=False)
    return cat_rev


# ─── Visualizations ──────────────────────────────────────────────────────────

def plot_word_frequency_bar(df: pd.DataFrame, top_n: int = 25) -> go.Figure:
    """Horizontal bar chart of top N words."""
    freq = get_word_frequencies(df, top_n=top_n).sort_values("Count")
    fig = px.bar(
        freq, x="Count", y="Word", orientation="h",
        color="Count", color_continuous_scale=["#3B1F6A", ACCENT, ACCENT2],
        title=f"📝 Top {top_n} Words in Product Descriptions",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(
        paper_bgcolor="#0F0F1A", plot_bgcolor="#0F0F1A",
        coloraxis_showscale=False,
        yaxis=dict(tickfont=dict(size=10)),
    )
    return fig


def plot_category_revenue(df: pd.DataFrame) -> go.Figure:
    """Treemap of revenue by category."""
    cat_rev = get_category_revenue(df)
    fig = px.treemap(
        cat_rev,
        path=["Category"],
        values="Revenue",
        color="Revenue",
        color_continuous_scale=["#1E1B4B", ACCENT, ACCENT2],
        title="📦 Revenue by Product Category (Keyword Extraction)",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(paper_bgcolor="#0F0F1A")
    return fig


def plot_category_breakdown(df: pd.DataFrame) -> go.Figure:
    """Multi-metric bar for each category."""
    cat_rev = get_category_revenue(df)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cat_rev["Category"], y=cat_rev["Revenue"],
        name="Revenue (£)", marker_color=ACCENT,
    ))
    fig.add_trace(go.Bar(
        x=cat_rev["Category"], y=cat_rev["Orders"],
        name="Orders", marker_color=ACCENT2, visible="legendonly",
    ))
    fig.add_trace(go.Bar(
        x=cat_rev["Category"], y=cat_rev["Products"],
        name="Unique Products", marker_color=ACCENT3, visible="legendonly",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="🏷️ Category Analysis — Revenue / Orders / Products",
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
        barmode="group",
        xaxis_tickangle=-30,
    )
    return fig


def generate_wordcloud_data(df: pd.DataFrame, top_n: int = 80) -> pd.DataFrame:
    """Return word + count data for use in a word cloud visualization."""
    freq = get_word_frequencies(df, top_n=top_n)
    freq["Size"] = (freq["Count"] / freq["Count"].max() * 60 + 10).astype(int)
    return freq


def plot_wordcloud_scatter(df: pd.DataFrame, top_n: int = 60) -> go.Figure:
    """Simulate a word cloud with a scatter plot (works in Streamlit without extra deps)."""
    freq = generate_wordcloud_data(df, top_n=top_n)
    rng = np.random.default_rng(42)
    freq["x"] = rng.uniform(0, 1, len(freq))
    freq["y"] = rng.uniform(0, 1, len(freq))

    fig = go.Figure()
    for _, row in freq.iterrows():
        fig.add_annotation(
            x=row["x"], y=row["y"],
            text=row["Word"],
            showarrow=False,
            font=dict(size=int(row["Size"] * 0.5 + 8),
                      color=f"hsl({int(row['x']*360)},70%,65%)"),
        )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="☁️ Product Description Word Cloud",
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1]),
        height=450,
    )
    return fig
