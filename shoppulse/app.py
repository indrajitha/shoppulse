"""
ShopPulse — Customer Behaviour & Product Intelligence Dashboard
Streamlit app entry point
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
import numpy as np

# ── Page config MUST be first ────────────────────────────────────────────────
st.set_page_config(
    page_title="ShopPulse | E-Commerce Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* === Base === */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0A0A14 0%, #0F0F1A 50%, #080812 100%);
    color: #E2E8F0;
}

/* === Sidebar === */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D0D1F 0%, #130F2A 100%) !important;
    border-right: 1px solid rgba(124,58,237,0.2);
}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p {
    color: #94A3B8 !important;
}

/* === Hero Header === */
.hero-header {
    background: linear-gradient(135deg, #1E1B4B 0%, #312E81 30%, #3B1F6A 70%, #1A1035 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(124,58,237,0.3);
    box-shadow: 0 20px 60px rgba(124,58,237,0.2), 0 0 0 1px rgba(124,58,237,0.1);
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(124,58,237,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #E2E8F0, #7C3AED, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    color: #94A3B8;
    font-size: 1.05rem;
    margin-top: 0.5rem;
    font-weight: 400;
}
.hero-badges {
    display: flex;
    gap: 10px;
    margin-top: 1.2rem;
    flex-wrap: wrap;
}
.badge {
    background: rgba(124,58,237,0.15);
    border: 1px solid rgba(124,58,237,0.35);
    border-radius: 100px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: #A78BFA;
    font-weight: 500;
}

/* === KPI Cards === */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 2rem;
}
.kpi-card {
    background: linear-gradient(135deg, rgba(30,27,75,0.8), rgba(19,15,42,0.9));
    border: 1px solid rgba(124,58,237,0.25);
    border-radius: 16px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.kpi-card:hover {
    border-color: rgba(124,58,237,0.55);
    box-shadow: 0 8px 30px rgba(124,58,237,0.2);
    transform: translateY(-2px);
}
.kpi-label {
    font-size: 0.72rem;
    color: #64748B;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-size: 1.75rem;
    font-weight: 800;
    background: linear-gradient(135deg, #E2E8F0, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* === Section Headers === */
.section-header {
    font-size: 1.35rem;
    font-weight: 700;
    color: #E2E8F0;
    padding: 0.5rem 0;
    border-left: 4px solid #7C3AED;
    padding-left: 1rem;
    margin: 1.5rem 0 1rem 0;
}

/* === Tables === */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(124,58,237,0.2) !important;
}

/* === Sidebar nav items === */
.nav-item {
    background: rgba(124,58,237,0.1);
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin: 0.3rem 0;
    cursor: pointer;
    color: #C4B5FD;
    font-weight: 500;
    transition: all 0.2s;
}

/* === Status indicators === */
.status-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 600;
}
.status-good  { background: rgba(16,185,129,0.15); color: #34D399; border: 1px solid rgba(16,185,129,0.3); }
.status-warn  { background: rgba(245,158,11,0.15); color: #FCD34D; border: 1px solid rgba(245,158,11,0.3); }
.status-bad   { background: rgba(239,68,68,0.15);  color: #FCA5A5; border: 1px solid rgba(239,68,68,0.3); }

/* === Info boxes === */
.info-box {
    background: rgba(6,182,212,0.08);
    border: 1px solid rgba(6,182,212,0.25);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    color: #7DD3FC;
    font-size: 0.9rem;
}

/* === Tabs === */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(15,15,26,0.8);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #64748B;
    font-weight: 500;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: rgba(124,58,237,0.3) !important;
    color: #A78BFA !important;
}

/* Plotly chart background */
.js-plotly-plot {
    border-radius: 12px !important;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ── Imports ───────────────────────────────────────────────────────────────────
from data_loader       import get_clean_data
from stats_analysis    import (compute_descriptive_stats, plot_revenue_over_time,
                                plot_top_products, plot_revenue_by_country,
                                plot_hourly_and_daily, plot_correlation_heatmap,
                                plot_price_distribution, plot_quantity_distribution,
                                get_underperforming_products)
from rfm_clustering    import (compute_rfm, add_rfm_segments, run_kmeans,
                                find_optimal_k, plot_elbow, plot_rfm_segments,
                                plot_rfm_3d, plot_segment_profiles, plot_cluster_scatter,
                                get_segment_summary)
from market_basket     import (build_basket_matrix, run_apriori, plot_rules_scatter,
                                plot_top_rules_bar, plot_rules_heatmap,
                                get_product_recommendations)
from text_processing   import (plot_word_frequency_bar, plot_category_revenue,
                                plot_category_breakdown, plot_wordcloud_scatter,
                                get_category_revenue, add_categories)
from forecasting       import (prepare_monthly_ts, prepare_product_ts,
                                run_prophet_forecast, plot_forecast, plot_trend_components,
                                get_forecast_summary, get_top_products_for_forecast)


# ─── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="⚡ Loading & cleaning data...")
def load_data(force_refresh: bool = False):
    return get_clean_data(force_refresh=force_refresh)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
        <div style="font-size:2.5rem">🛒</div>
        <div style="font-size:1.2rem; font-weight:800; color:#A78BFA;">ShopPulse</div>
        <div style="font-size:0.7rem; color:#4B5563; letter-spacing:2px;">E-COMMERCE INTELLIGENCE</div>
    </div>
    <hr style="border-color: rgba(124,58,237,0.2); margin: 1rem 0;">
    """, unsafe_allow_html=True)

    page = st.selectbox(
        "📍 Navigate",
        options=[
            "🏠 Overview",
            "📊 Statistical Analysis",
            "👥 RFM Segmentation",
            "🔬 K-Means Clustering",
            "🛒 Market Basket Analysis",
            "📝 Text & Category Analysis",
            "🔮 Demand Forecasting",
        ],
        key="page_select",
    )

    st.markdown("---")
    st.markdown("**⚙️ Global Filters**")

    # Load data first to get filter options
    with st.spinner("Loading dataset..."):
        df_raw = load_data()

    all_countries = ["All"] + sorted(df_raw["Country"].unique().tolist())
    selected_country = st.selectbox("🌍 Country", all_countries, key="country_filter")

    # Filter data
    df = df_raw if selected_country == "All" else df_raw[df_raw["Country"] == selected_country]

    min_date = df["InvoiceDate"].min().date()
    max_date = df["InvoiceDate"].max().date()
    date_range = st.date_input(
        "📅 Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="date_range",
    )
    if len(date_range) == 2:
        df = df[(df["InvoiceDate"].dt.date >= date_range[0]) &
                (df["InvoiceDate"].dt.date <= date_range[1])]

    st.markdown("---")
    st.markdown(f"""
    <div style="font-size:0.75rem; color:#4B5563; text-align:center;">
        <div>📦 {len(df):,} transactions</div>
        <div>👤 {df['CustomerID'].nunique():,} customers</div>
        <div>🏷️ {df['StockCode'].nunique():,} products</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔄 Refresh Data Cache", width='stretch'):
        st.cache_data.clear()
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">🛒 ShopPulse</h1>
        <p class="hero-subtitle">Customer Behaviour & Product Intelligence Platform<br>
        Powered by UCI Online Retail Dataset (~540K real UK transactions)</p>
        <div class="hero-badges">
            <span class="badge">📊 Statistical Analysis</span>
            <span class="badge">👥 RFM Segmentation</span>
            <span class="badge">🔬 K-Means Clustering</span>
            <span class="badge">🛒 Market Basket</span>
            <span class="badge">📝 Text Mining</span>
            <span class="badge">🔮 Demand Forecasting</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards
    stats = compute_descriptive_stats(df)
    kpi_items = [
        ("💰", "Total Revenue", stats["Total Revenue (£)"]),
        ("🧾", "Transactions",  stats["Total Transactions"]),
        ("👤", "Customers",     stats["Total Customers"]),
        ("📦", "Products",      stats["Total Products"]),
        ("🌍", "Countries",     stats["Total Countries"]),
        ("🛒", "Avg Order",     stats["Avg Order Value (£)"]),
        ("📏", "Avg Qty/Line",  stats["Avg Quantity per Line"]),
        ("💲", "Median Price",  stats["Median Unit Price (£)"]),
    ]

    cols = st.columns(4)
    for i, (icon, label, val) in enumerate(kpi_items):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="kpi-card">
                <div style="font-size:1.8rem">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="info-box">
        📅 <strong>Date Range:</strong> {stats['Date Range']}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        🌍 <strong>Filter:</strong> {selected_country}
        &nbsp;&nbsp;|&nbsp;&nbsp;
        📦 <strong>Rows loaded:</strong> {len(df):,}
    </div>
    """, unsafe_allow_html=True)

    # Charts
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(plot_revenue_over_time(df), width='stretch')
    with col2:
        st.plotly_chart(plot_revenue_by_country(df), width='stretch')

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(plot_top_products(df), width='stretch')
    with col4:
        st.plotly_chart(plot_hourly_and_daily(df), width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Statistical Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Statistical Analysis":
    st.markdown('<div class="section-header">📊 Statistical Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Revenue Trends", "🏆 Products", "💰 Distributions", "🔗 Correlations"
    ])

    with tab1:
        st.plotly_chart(plot_revenue_over_time(df), width='stretch')
        st.plotly_chart(plot_hourly_and_daily(df), width='stretch')
        st.plotly_chart(plot_revenue_by_country(df), width='stretch')

    with tab2:
        n_top = st.slider("Number of top products", 10, 30, 15, key="top_prod_n")
        st.plotly_chart(plot_top_products(df, n=n_top), width='stretch')

        st.markdown('<div class="section-header">⚠️ Underperforming Products</div>', unsafe_allow_html=True)
        n_bottom = st.slider("Number of bottom products", 5, 20, 10, key="bot_prod_n")
        under = get_underperforming_products(df, bottom_n=n_bottom)
        st.dataframe(
            under.style.background_gradient(subset=["Revenue"], cmap="Purples"),
            width='stretch',
        )

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_price_distribution(df), width='stretch')
        with col2:
            st.plotly_chart(plot_quantity_distribution(df), width='stretch')

        # Revenue per customer distribution
        rev_per_cust = df.groupby("CustomerID")["Revenue"].sum()
        import plotly.express as px
        fig_cust = px.histogram(
            rev_per_cust[rev_per_cust < rev_per_cust.quantile(0.97)],
            nbins=60,
            title="💎 Revenue Distribution per Customer",
            template="plotly_dark",
            color_discrete_sequence=["#F59E0B"],
        )
        fig_cust.update_layout(paper_bgcolor="#0F0F1A", plot_bgcolor="#0F0F1A")
        st.plotly_chart(fig_cust, width='stretch')

    with tab4:
        st.plotly_chart(plot_correlation_heatmap(df), width='stretch')

        # Descriptive stats table
        st.markdown('<div class="section-header">📋 Descriptive Statistics</div>', unsafe_allow_html=True)
        numeric_cols = ["Quantity", "UnitPrice", "Revenue"]
        desc_df = df[numeric_cols].describe().round(3)
        st.dataframe(desc_df.style.background_gradient(cmap="Purples"), width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RFM Segmentation
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👥 RFM Segmentation":
    st.markdown('<div class="section-header">👥 RFM Customer Segmentation</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>RFM Analysis</strong> segments customers by: 
        <strong>Recency</strong> (how recently they bought),
        <strong>Frequency</strong> (how often), and
        <strong>Monetary</strong> (how much they spent).
        Each dimension is scored 1–5, then customers are classified into strategic segments.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🧮 Computing RFM scores..."):
        rfm = compute_rfm(df)
        rfm_segmented = add_rfm_segments(rfm)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🥧 Segments", "📊 Profiles", "🧊 3D Space", "📋 Customer Table"
    ])

    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(plot_rfm_segments(rfm_segmented), width='stretch')
        with col2:
            summary = get_segment_summary(rfm_segmented)
            st.markdown("**Segment Summary**")
            st.dataframe(
                summary.style.background_gradient(subset=["Total_Revenue"], cmap="Purples"),
                width='stretch',
                hide_index=True,
            )

    with tab2:
        st.plotly_chart(plot_segment_profiles(rfm_segmented), width='stretch')

    with tab3:
        color_choice = st.radio(
            "Color by:", ["Segment", "R_Score", "F_Score", "M_Score"],
            horizontal=True, key="rfm_color"
        )
        st.plotly_chart(plot_rfm_3d(rfm_segmented, color_col=color_choice), width='stretch')

    with tab4:
        seg_filter = st.multiselect(
            "Filter by Segment",
            options=sorted(rfm_segmented["Segment"].unique()),
            default=list(rfm_segmented["Segment"].unique()),
            key="rfm_seg_filter",
        )
        disp = rfm_segmented[rfm_segmented["Segment"].isin(seg_filter)]
        cols_show = ["CustomerID", "Recency", "Frequency", "Monetary",
                     "R_Score", "F_Score", "M_Score", "RFM_Score", "Segment"]
        st.dataframe(
            disp[cols_show].sort_values("Monetary", ascending=False)
                           .reset_index(drop=True)
                           .style.background_gradient(subset=["Monetary"], cmap="Purples"),
            width='stretch',
        )

        csv = disp[cols_show].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download RFM Table", csv, "rfm_segments.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: K-Means Clustering
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 K-Means Clustering":
    st.markdown('<div class="section-header">🔬 K-Means Customer Clustering</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        K-Means groups customers into clusters based on their RFM values.
        Use the <strong>Elbow method</strong> + <strong>Silhouette score</strong> to find the optimal K.
        Features are <strong>log-transformed</strong> and <strong>standardized</strong> before clustering.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🧮 Computing RFM..."):
        rfm = compute_rfm(df)

    col1, col2 = st.columns([1, 2])
    with col1:
        n_clusters = st.slider("Number of clusters (K)", 2, 8, 5, key="kmeans_k")

    with st.spinner("📊 Analyzing optimal K..."):
        k_range, inertias, silhouettes = find_optimal_k(rfm)
    st.plotly_chart(plot_elbow(k_range, inertias, silhouettes), width='stretch')

    with st.spinner(f"🔬 Running K-Means (K={n_clusters})..."):
        rfm_clustered, km_model, scaler = run_kmeans(rfm, n_clusters=n_clusters)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_cluster_scatter(rfm_clustered), width='stretch')
    with col2:
        st.plotly_chart(plot_rfm_3d(rfm_clustered, color_col="Cluster"), width='stretch')

    # Cluster profiles table
    st.markdown('<div class="section-header">📋 Cluster Profiles</div>', unsafe_allow_html=True)
    cluster_summary = rfm_clustered.groupby("Cluster").agg(
        Customers        = ("CustomerID", "count"),
        Avg_Recency_Days = ("Recency",    "mean"),
        Avg_Frequency    = ("Frequency",  "mean"),
        Avg_Monetary     = ("Monetary",   "mean"),
        Total_Revenue    = ("Monetary",   "sum"),
    ).round(1).reset_index()
    st.dataframe(
        cluster_summary.style.background_gradient(subset=["Total_Revenue"], cmap="Purples"),
        width='stretch', hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Market Basket Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🛒 Market Basket Analysis":
    st.markdown('<div class="section-header">🛒 Market Basket Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>Association Rule Mining</strong> using FP-Growth algorithm discovers 
        products frequently bought together. Generates rules of the form:
        <em>"Customers who bought X also bought Y"</em>.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        min_support    = st.slider("Min Support", 0.01, 0.15, 0.02, 0.005, key="mba_supp",
                                    help="Fraction of transactions containing the itemset")
    with col2:
        min_confidence = st.slider("Min Confidence", 0.1, 0.9, 0.3, 0.05, key="mba_conf",
                                    help="P(consequent | antecedent)")
    with col3:
        min_lift       = st.slider("Min Lift", 1.0, 5.0, 1.0, 0.1, key="mba_lift",
                                    help="Values > 1 indicate positive association")

    mba_country = st.selectbox(
        "Country for basket analysis",
        ["United Kingdom"] + [c for c in sorted(df["Country"].unique()) if c != "United Kingdom"],
        key="mba_country",
    )

    with st.spinner("🔄 Building basket matrix & mining rules..."):
        basket = build_basket_matrix(df, country=mba_country)
        rules  = run_apriori(basket, min_support=min_support,
                              min_confidence=min_confidence, min_lift=min_lift)

    if rules.empty:
        st.warning("⚠️ No rules found with current thresholds. Try lowering Support or Confidence.")
    else:
        st.success(f"✅ Found **{len(rules):,}** association rules!")
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Scatter", "🏆 Top Rules", "🔥 Heatmap", "🔍 Product Lookup"
        ])
        with tab1:
            st.plotly_chart(plot_rules_scatter(rules), width='stretch')
        with tab2:
            n_rules = st.slider("Top N rules to show", 5, 30, 15, key="mba_top_n")
            st.plotly_chart(plot_top_rules_bar(rules, n=n_rules), width='stretch')
        with tab3:
            st.plotly_chart(plot_rules_heatmap(rules), width='stretch')
        with tab4:
            all_products = sorted(df["Description"].dropna().unique().tolist())
            selected_product = st.selectbox("🔍 Select a product", all_products, key="mba_product")
            recs = get_product_recommendations(rules, selected_product)
            if recs.empty:
                st.info("No recommendations found for this product in current rules.")
            else:
                st.markdown(f"**Customers who buy '{selected_product}' also buy:**")
                st.dataframe(recs.style.background_gradient(subset=["lift"], cmap="Purples"),
                              width='stretch', hide_index=True)

        st.markdown('<div class="section-header">📋 All Rules</div>', unsafe_allow_html=True)
        st.dataframe(
            rules.style.background_gradient(subset=["lift", "confidence"], cmap="Purples"),
            width='stretch',
        )
        csv = rules.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Rules CSV", csv, "association_rules.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Text & Category Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📝 Text & Category Analysis":
    st.markdown('<div class="section-header">📝 Text Processing & Category Analysis</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        Analyzes product <strong>description text</strong> to extract word frequencies,
        infer product categories using keyword matching, and understand the product portfolio structure.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["☁️ Word Analysis", "📦 Categories", "🌳 Treemap"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            top_n_words = st.slider("Top N words", 15, 60, 25, key="text_word_n")
        st.plotly_chart(plot_word_frequency_bar(df, top_n=top_n_words), width='stretch')
        st.plotly_chart(plot_wordcloud_scatter(df, top_n=60), width='stretch')

    with tab2:
        st.plotly_chart(plot_category_breakdown(df), width='stretch')

        cat_rev = get_category_revenue(df)
        st.markdown("**Category Revenue Summary**")
        st.dataframe(
            cat_rev.style.background_gradient(subset=["Revenue"], cmap="Purples"),
            width='stretch', hide_index=True,
        )

    with tab3:
        st.plotly_chart(plot_category_revenue(df), width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Demand Forecasting
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Demand Forecasting":
    st.markdown('<div class="section-header">🔮 Demand Forecasting</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        Uses <strong>Meta Prophet</strong> (additive time series model) to forecast 
        monthly revenue and individual product demand.
        Prophet handles seasonality, holidays, and trend changes automatically.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📈 Revenue Forecast", "📦 Product Forecast"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            periods = st.slider("Forecast horizon (months)", 3, 12, 6, key="fc_periods")

        with st.spinner("🔮 Running forecast..."):
            ts = prepare_monthly_ts(df)
            if len(ts) < 3:
                st.warning("⚠️ Not enough data for forecasting. At least 3 months of data required.")
            else:
                _, forecast = run_prophet_forecast(ts, periods=periods)

                summary = get_forecast_summary(ts, forecast)
                col1, col2, col3, col4 = st.columns(4)
                for (label, val), col in zip(summary.items(), [col1, col2, col3, col4]):
                    with col:
                        icon = "📉" if "Historical" in label else "🔮" if "Forecast" in label else "📈" if "Growth" in label else "📅"
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div style="font-size:1.5rem">{icon}</div>
                            <div class="kpi-label">{label}</div>
                            <div class="kpi-value" style="font-size:1.2rem">{val}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.plotly_chart(plot_forecast(ts, forecast, title="📈 Monthly Revenue Forecast"),
                                width='stretch')
                st.plotly_chart(plot_trend_components(forecast), width='stretch')

    with tab2:
        top_products = get_top_products_for_forecast(df, n=30)
        product = st.selectbox("📦 Select Product", top_products, key="fc_product")
        periods_prod = st.slider("Forecast horizon (months)", 3, 12, 4, key="fc_prod_periods")

        with st.spinner(f"🔮 Forecasting demand for '{product}'..."):
            ts_prod = prepare_product_ts(df, product_desc=product)
            if len(ts_prod) < 3:
                st.warning("⚠️ Insufficient data for this product. Select another.")
            else:
                _, forecast_prod = run_prophet_forecast(ts_prod, periods=periods_prod)
                st.plotly_chart(
                    plot_forecast(ts_prod, forecast_prod,
                                  title=f"📦 Demand Forecast — {product}", unit="units"),
                    width='stretch',
                )
