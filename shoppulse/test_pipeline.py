import sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, 'src')
from data_loader import get_clean_data
from rfm_clustering import compute_rfm, add_rfm_segments, run_kmeans
from text_processing import get_category_revenue
from market_basket import build_basket_matrix, run_apriori

df = get_clean_data()

# Test RFM
rfm = compute_rfm(df)
rfm_seg = add_rfm_segments(rfm)
print("RFM Segments:")
print(rfm_seg["Segment"].value_counts().to_string())

# Test K-Means
rfm_cl, km, scaler = run_kmeans(rfm, n_clusters=5)
print("K-Means clusters:", rfm_cl["Cluster"].value_counts().to_string())

# Test Categories
cats = get_category_revenue(df)
print("Top categories:")
for _, row in cats.head(4).iterrows():
    print(f"  {row['Category']}: {row['Revenue']:.0f}")

# Test Market Basket
basket = build_basket_matrix(df, country="All")
print("Basket shape:", basket.shape)
rules = run_apriori(basket, min_support=0.05, min_confidence=0.3, min_lift=1.0)
print("Association rules found:", len(rules))
print("ALL PIPELINE TESTS PASSED")
