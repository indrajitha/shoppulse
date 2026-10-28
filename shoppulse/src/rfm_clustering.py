"""
ShopPulse - RFM Analysis + K-Means Customer Segmentation
Segments customers into: Champions, Loyal, At-Risk, One-Time, Big Spenders, etc.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PLOTLY_TEMPLATE = "plotly_dark"
ACCENT  = "#7C3AED"
ACCENT2 = "#06B6D4"
ACCENT3 = "#F59E0B"
ACCENT4 = "#10B981"

SEGMENT_COLORS = {
    "Champions":       "#10B981",
    "Loyal":           "#06B6D4",
    "Potential Loyal": "#8B5CF6",
    "New Customers":   "#F59E0B",
    "At-Risk":         "#EF4444",
    "Lost":            "#6B7280",
    "Big Spenders":    "#F97316",
    "One-Time":        "#EC4899",
}


# ─── RFM Calculation ─────────────────────────────────────────────────────────

def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary for each customer."""
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency   = ("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency = ("InvoiceNo",   "nunique"),
        Monetary  = ("Revenue",     "sum"),
    ).reset_index()

    # RFM Scores (1-5, 5=best)
    rfm["R_Score"] = pd.qcut(rfm["Recency"],   5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"),  5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["RFM_Score"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
    rfm["RFM_Total"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    return rfm


def assign_segment(row: pd.Series) -> str:
    r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
    rfm = row["RFM_Total"]
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif r >= 3 and f >= 3 and m >= 3:
        return "Loyal"
    elif r >= 3 and f <= 2:
        return "Potential Loyal"
    elif r == 5 and f == 1:
        return "New Customers"
    elif r <= 2 and f >= 3:
        return "At-Risk"
    elif r <= 2 and f <= 2 and m <= 2:
        return "Lost"
    elif m == 5:
        return "Big Spenders"
    else:
        return "One-Time"


def add_rfm_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    rfm = rfm.copy()
    rfm["Segment"] = rfm.apply(assign_segment, axis=1)
    return rfm


# ─── K-Means Clustering ──────────────────────────────────────────────────────

def run_kmeans(rfm: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    """Run K-Means on log-transformed RFM features."""
    rfm = rfm.copy()
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()

    # Log transform to reduce skewness
    features["Recency"]   = np.log1p(features["Recency"])
    features["Frequency"] = np.log1p(features["Frequency"])
    features["Monetary"]  = np.log1p(features["Monetary"])

    scaler  = StandardScaler()
    scaled  = scaler.fit_transform(features)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
    rfm["Cluster"] = km.fit_predict(scaled)
    rfm["Cluster"] = rfm["Cluster"].astype(str)

    return rfm, km, scaler


def find_optimal_k(rfm: pd.DataFrame, k_range=range(2, 9)) -> tuple[list, list]:
    """Elbow + silhouette analysis to find optimal K."""
    features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    features = np.log1p(features)
    scaler   = StandardScaler()
    scaled   = scaler.fit_transform(features)

    inertias    = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(scaled, labels))

    return list(k_range), inertias, silhouettes


# ─── Visualizations ──────────────────────────────────────────────────────────

def plot_elbow(k_range, inertias, silhouettes) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Elbow Method (Inertia)", "Silhouette Score"),
    )
    fig.add_trace(go.Scatter(x=list(k_range), y=inertias,
                             mode="lines+markers",
                             line=dict(color=ACCENT, width=3),
                             marker=dict(size=8),
                             name="Inertia"), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(k_range), y=silhouettes,
                             mode="lines+markers",
                             line=dict(color=ACCENT2, width=3),
                             marker=dict(size=8),
                             name="Silhouette"), row=1, col=2)
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="📊 Optimal K Analysis",
        paper_bgcolor="#0F0F1A",
        plot_bgcolor="#0F0F1A",
        showlegend=False,
    )
    return fig


def plot_rfm_segments(rfm_segmented: pd.DataFrame) -> go.Figure:
    """Segment distribution donut."""
    seg_counts = rfm_segmented["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]
    colors = [SEGMENT_COLORS.get(s, "#888888") for s in seg_counts["Segment"]]

    fig = go.Figure(data=[go.Pie(
        labels=seg_counts["Segment"],
        values=seg_counts["Count"],
        hole=0.5,
        marker_colors=colors,
        textinfo="percent+label",
    )])
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="👥 Customer Segments",
        paper_bgcolor="#0F0F1A",
    )
    return fig


def plot_rfm_3d(rfm: pd.DataFrame, color_col: str = "Segment") -> go.Figure:
    """3D scatter of RFM space colored by segment or cluster."""
    palette = list(SEGMENT_COLORS.values())
    fig = px.scatter_3d(
        rfm,
        x="Recency", y="Frequency", z="Monetary",
        color=color_col,
        title="🧊 RFM 3D Space",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=palette,
        opacity=0.7,
        labels={"Recency": "Recency (days)", "Frequency": "Frequency", "Monetary": "Revenue (£)"},
        hover_data=["CustomerID", "RFM_Score"] if "RFM_Score" in rfm.columns else ["CustomerID"],
    )
    fig.update_layout(paper_bgcolor="#0F0F1A")
    return fig


def plot_segment_profiles(rfm_segmented: pd.DataFrame) -> go.Figure:
    """Radar chart showing RFM mean by segment."""
    seg_means = rfm_segmented.groupby("Segment").agg(
        Recency   = ("Recency",   "mean"),
        Frequency = ("Frequency", "mean"),
        Monetary  = ("Monetary",  "mean"),
    ).reset_index()

    # Normalize 0-1
    for col in ["Recency", "Frequency", "Monetary"]:
        col_min = seg_means[col].min()
        col_max = seg_means[col].max()
        if col == "Recency":
            # Lower recency = better, so invert
            seg_means[f"{col}_norm"] = 1 - (seg_means[col] - col_min) / (col_max - col_min + 1e-9)
        else:
            seg_means[f"{col}_norm"] = (seg_means[col] - col_min) / (col_max - col_min + 1e-9)

    categories = ["Recency_norm", "Frequency_norm", "Monetary_norm"]
    cat_labels  = ["Recency", "Frequency", "Monetary"]

    fig = go.Figure()
    colors = list(SEGMENT_COLORS.values())
    for i, row in seg_means.iterrows():
        vals = [row[c] for c in categories]
        vals += [vals[0]]  # close polygon
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=cat_labels + [cat_labels[0]],
            fill="toself",
            name=row["Segment"],
            line_color=colors[i % len(colors)],
            opacity=0.6,
        ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="🎯 Segment RFM Profiles (Radar)",
        paper_bgcolor="#0F0F1A",
    )
    return fig


def plot_cluster_scatter(rfm: pd.DataFrame) -> go.Figure:
    """2D scatter: Recency vs Monetary colored by cluster."""
    fig = px.scatter(
        rfm, x="Recency", y="Monetary",
        color="Cluster" if "Cluster" in rfm.columns else "Segment",
        size="Frequency",
        title="🔵 K-Means Clusters: Recency vs Revenue",
        template=PLOTLY_TEMPLATE,
        color_discrete_sequence=list(SEGMENT_COLORS.values()),
        hover_data=["CustomerID"],
        opacity=0.7,
        labels={"Recency": "Recency (days)", "Monetary": "Revenue (£)"},
    )
    fig.update_layout(paper_bgcolor="#0F0F1A", plot_bgcolor="#0F0F1A")
    return fig


def get_segment_summary(rfm_segmented: pd.DataFrame) -> pd.DataFrame:
    """Summary table of each segment."""
    summary = rfm_segmented.groupby("Segment").agg(
        Customers        = ("CustomerID", "count"),
        Avg_Recency_Days = ("Recency",    "mean"),
        Avg_Orders       = ("Frequency",  "mean"),
        Avg_Revenue      = ("Monetary",   "mean"),
        Total_Revenue    = ("Monetary",   "sum"),
    ).reset_index()
    summary["Avg_Recency_Days"] = summary["Avg_Recency_Days"].round(0).astype(int)
    summary["Avg_Orders"]       = summary["Avg_Orders"].round(1)
    summary["Avg_Revenue"]      = summary["Avg_Revenue"].round(2)
    summary["Total_Revenue"]    = summary["Total_Revenue"].round(2)
    summary = summary.sort_values("Total_Revenue", ascending=False)
    return summary.reset_index(drop=True)
