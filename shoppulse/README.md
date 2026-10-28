# 🛒 ShopPulse — E-Commerce Customer Behaviour & Product Intelligence

[![Python](https://img.shields.io/badge/Python-3.10+-7C3AED?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-06B6D4?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20Online%20Retail-F59E0B?style=flat-square)](https://archive.ics.uci.edu/dataset/352/online+retail)

> A full-stack data analytics platform for online retail intelligence — built on the UCI Online Retail dataset (~540K real UK transactions, 2010–2011).

---

## 🗺️ Project Pipeline

| Stage | What it does |
|-------|-------------|
| **Data Preprocessing** | Clean transactions: missing CustomerIDs, cancelled orders, negative quantities, duplicates, date parsing |
| **Statistical Analysis** | Descriptive stats on price/quantity/revenue, correlation, revenue by country/month, top products |
| **Pattern Mining** | Market basket analysis (FP-Growth) → "Customers who bought X also bought Y" |
| **Classification/Clustering** | RFM + K-Means clustering → customer segments (Champions, Loyal, At-Risk, etc.) |
| **Text Processing** | Product description field → word frequency, category extraction, keyword clustering |
| **Visualization & Interpretation** | Dashboards: sales trends, top products, segment profiles, basket rules, demand forecast |

---

## 📁 Project Structure

```
shoppulse/
├── app.py                   # 🎛️  Streamlit dashboard (entry point)
├── requirements.txt         # 📦 Dependencies
├── .streamlit/
│   └── config.toml          # 🎨 Dark purple theme
├── src/
│   ├── data_loader.py       # 📂 Data loading, cleaning, synthetic generation
│   ├── stats_analysis.py    # 📊 Statistical analysis & visualizations
│   ├── rfm_clustering.py    # 👥 RFM analysis + K-Means segmentation
│   ├── market_basket.py     # 🛒 Association rule mining (FP-Growth/Apriori)
│   ├── text_processing.py   # 📝 Text mining, word frequency, categories
│   └── forecasting.py       # 🔮 Prophet demand forecasting
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_statistical_analysis.ipynb
│   ├── 03_market_basket.ipynb
│   ├── 04_rfm_clustering.ipynb
│   ├── 05_text_processing.ipynb
│   └── 06_forecasting.ipynb
└── data/
    ├── README.md            # Dataset instructions
    └── online_retail_cleaned.parquet  # Auto-generated cache
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Add the real dataset
Download from [UCI](https://archive.ics.uci.edu/dataset/352/online+retail) or [Kaggle](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) and place as:
```
data/online_retail.xlsx   # or online_retail.csv
```
> ⚠️ If no dataset is found, the app auto-generates a realistic **synthetic dataset** (~25,000 invoices) with the same schema.

### 3. Launch the dashboard
```bash
streamlit run app.py
```

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| **🏠 Overview** | KPI cards, revenue trend, top products, country breakdown |
| **📊 Statistical Analysis** | Revenue trends, distributions, correlations, underperforming products |
| **👥 RFM Segmentation** | Recency-Frequency-Monetary scoring, segment profiles, 3D visualization |
| **🔬 K-Means Clustering** | Elbow + silhouette analysis, cluster scatter plots |
| **🛒 Market Basket** | FP-Growth association rules, confidence heatmap, product recommendations |
| **📝 Text Analysis** | Word frequency, word cloud, category treemap |
| **🔮 Demand Forecasting** | Prophet-based revenue + product demand forecast with confidence intervals |

---

## 🧪 Analytics Methods

- **RFM Analysis**: Score customers 1–5 on Recency, Frequency, Monetary dimensions
- **K-Means Clustering**: Log-transformed, standardized RFM features; elbow + silhouette for K selection
- **FP-Growth**: Fast frequent pattern mining with configurable support/confidence/lift thresholds
- **Prophet**: Facebook's additive time series model for trend + seasonality decomposition
- **Text Mining**: Regex tokenization, stop-word removal, TF-style frequency analysis

---

## 📦 Dataset Info

| Field | Value |
|-------|-------|
| Source | UCI Machine Learning Repository |
| Records | ~541,909 transactions |
| Customers | ~4,373 unique |
| Products | ~4,070 unique |
| Countries | 38 |
| Period | Dec 2010 – Dec 2011 |

---

## 🎨 Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Streamlit** | Interactive web dashboard |
| **Pandas + NumPy** | Data manipulation |
| **Scikit-learn** | K-Means clustering, preprocessing |
| **MLxtend** | Apriori / FP-Growth association rules |
| **Prophet** | Time series forecasting |
| **Plotly** | Interactive visualizations |
| **WordCloud** | Text visualization |
