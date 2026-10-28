"""
ShopPulse - Data Loader
Handles downloading and loading the Online Retail dataset
"""
import os
import io
import hashlib
import zipfile
import requests
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CACHE_PATH = os.path.join(DATA_DIR, "online_retail_cleaned.parquet")
RAW_XLSX = os.path.join(DATA_DIR, "online_retail.xlsx")
RAW_CSV = os.path.join(DATA_DIR, "online_retail.csv")
SYNTHETIC_PATH = os.path.join(DATA_DIR, "synthetic_retail.parquet")


def generate_synthetic_data(n_customers: int = 4000, n_invoices: int = 25000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic Online Retail dataset when real data is unavailable."""
    rng = np.random.default_rng(seed)

    products = [
        ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.55),
        ("71053",  "WHITE METAL LANTERN", 3.39),
        ("84406B", "CREAM CUPID HEARTS COAT HANGER", 2.75),
        ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE", 3.39),
        ("84029E", "RED WOOLLY HOTTIE WHITE HEART.", 3.39),
        ("22752",  "SET 7 BABUSHKA NESTING BOXES", 7.65),
        ("21730",  "GLASS STAR FROSTED T-LIGHT HOLDER", 4.25),
        ("22633",  "HAND WARMER UNION JACK", 1.85),
        ("22632",  "HAND WARMER RED POLKA DOT", 1.85),
        ("47566",  "PARTY BUNTING", 4.95),
        ("85099B", "JUMBO BAG RED RETROSPOT", 1.65),
        ("21212",  "PACK OF 72 RETROSPOT CAKE CASES", 0.55),
        ("23298",  "SPOTTY BUNTING", 4.95),
        ("22961",  "JAM MAKING SET WITH JARS", 4.25),
        ("20725",  "LUNCH BAG RED RETROSPOT", 1.65),
        ("22469",  "HEART OF WICKER SMALL", 1.65),
        ("21232",  "STRAWBERRY CERAMIC TRINKET BOX", 1.25),
        ("22197",  "SMALL POPCORN HOLDER", 0.72),
        ("21915",  "RED HARMONICA IN BOX", 1.25),
        ("22111",  "SCOTTIE DOG HOT WATER BOTTLE", 4.95),
        ("85099F", "JUMBO BAG VINTAGE DOILY", 1.65),
        ("23483",  "SET OF 4 KNICK KNACK TINS DOILY", 4.95),
        ("22423",  "REGENCY CAKESTAND 3 TIER", 10.95),
        ("47566B", "TEA TIME PARTY BUNTING", 4.95),
        ("21928",  "JUMBO STORAGE BAG SUKI", 2.08),
        ("23320",  "SET OF 3 BUTTERFLY COOKIE CUTTERS", 1.25),
        ("22748",  "FELTCRAFT PRINCESS CHARLOTTE DOLL", 3.75),
        ("22090",  "PAPER BUNTING RETROSPOT", 2.55),
        ("21977",  "PACK OF 60 MUSHROOM CAKE CASES", 0.55),
        ("22551",  "PLASTERS IN TIN CIRCUS PARADE", 1.65),
        ("POST",   "POSTAGE", 18.00),
        ("22386",  "JUMBO BAG PINK POLKADOT", 1.65),
        ("21258",  "VICTORIAN SEWING BOX LARGE", 9.95),
        ("22899",  "CHILDREN'S APRON DOLLY GIRL", 2.08),
        ("23355",  "HOT WATER BOTTLE KEEP CALM", 4.95),
        ("22629",  "SPACEBOY LUNCH BOX", 1.65),
        ("22659",  "LUNCH BOX WITH CUTLERY RETROSPOT", 2.55),
        ("21479",  "WHITE ROUND PHOTO FRAME", 2.49),
        ("22113",  "MAGIC DRAWING BOARD", 1.85),
        ("22410",  "LAVENDER SCENTED CANDLE", 1.85),
    ]
    stock_codes   = [p[0] for p in products]
    descriptions  = [p[1] for p in products]
    base_prices   = np.array([p[2] for p in products])
    # Popularity weights (some products sell much more)
    weights = rng.dirichlet(np.ones(len(products)) * 0.5) 

    countries = ["United Kingdom"] * 70 + [
        "Germany", "France", "EIRE", "Netherlands", "Belgium", "Spain",
        "Switzerland", "Portugal", "Australia", "Norway", "Cyprus",
        "Sweden", "Japan", "Denmark", "Singapore", "Finland",
    ]
    country_weights = [70/100] + [(30/100) / 15] * 15 + [0]

    customer_ids = list(range(12000, 12000 + n_customers))
    # Customer country mapping
    cust_countries = rng.choice(countries[:16], size=n_customers,
                                p=[0.70] + [0.30/15]*15)

    # Build invoices
    rows = []
    start_date = pd.Timestamp("2010-12-01")
    end_date   = pd.Timestamp("2011-12-09")
    date_range = (end_date - start_date).days

    invoice_counter = 536365

    for _ in range(n_invoices):
        # ~4% cancellations
        is_cancel = rng.random() < 0.04
        inv_prefix = "C" if is_cancel else ""
        invoice_no = f"{inv_prefix}{invoice_counter}"
        invoice_counter += 1

        cust_idx  = rng.integers(0, n_customers)
        cust_id   = customer_ids[cust_idx]
        country   = cust_countries[cust_idx]

        n_lines   = rng.integers(1, 12)
        items     = rng.choice(len(products), size=n_lines, replace=False, p=weights)

        inv_date  = start_date + pd.Timedelta(days=int(rng.integers(0, date_range)),
                                              hours=int(rng.integers(8, 18)),
                                              minutes=int(rng.integers(0, 59)))

        for item_idx in items:
            qty  = int(rng.integers(1, 25)) * (-1 if is_cancel else 1)
            price = float(round(base_prices[item_idx] * rng.uniform(0.95, 1.05), 2))
            rows.append({
                "InvoiceNo":   invoice_no,
                "StockCode":   stock_codes[item_idx],
                "Description": descriptions[item_idx],
                "Quantity":    qty,
                "InvoiceDate": inv_date,
                "UnitPrice":   price,
                "CustomerID":  float(cust_id),
                "Country":     country,
            })

    df = pd.DataFrame(rows)
    # Add a handful of missing CustomerIDs  (~25% to mirror real data)
    mask = rng.random(len(df)) < 0.25
    df.loc[mask, "CustomerID"] = np.nan
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def load_raw_data() -> pd.DataFrame:
    """Load raw data from file or generate synthetic data."""
    if os.path.exists(RAW_XLSX):
        print("[INFO] Loading Online Retail .xlsx ...")
        df = pd.read_excel(RAW_XLSX, engine="openpyxl", dtype={"CustomerID": str})
        return df
    if os.path.exists(RAW_CSV):
        print("[INFO] Loading Online Retail .csv ...")
        # Try utf-8 first, fall back to latin-1 for older exports
        for enc in ("utf-8", "latin-1"):
            try:
                df = pd.read_csv(RAW_CSV, encoding=enc, low_memory=False)
                print(f"[INFO] CSV decoded with encoding: {enc}")
                return df
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode CSV with utf-8 or latin-1")
    # Fall back to synthetic
    print("[WARN] No dataset found. Generating synthetic data (~25,000 invoices)...")
    df = generate_synthetic_data()
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_parquet(SYNTHETIC_PATH, index=False)
    print(f"[OK] Synthetic data saved to {SYNTHETIC_PATH}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Full data cleaning pipeline matching the Online Retail schema."""
    df = df.copy()

    # ── 1. Normalize column names (handle Online Retail I & II schemas) ────────
    df.columns = df.columns.str.strip()

    # Online Retail II uses: Invoice, Price, Customer ID
    # Online Retail I uses:  InvoiceNo, UnitPrice, CustomerID
    rename_map = {}
    if "Invoice" in df.columns and "InvoiceNo" not in df.columns:
        rename_map["Invoice"] = "InvoiceNo"
    if "Price" in df.columns and "UnitPrice" not in df.columns:
        rename_map["Price"] = "UnitPrice"
    if "Customer ID" in df.columns and "CustomerID" not in df.columns:
        rename_map["Customer ID"] = "CustomerID"
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"  [INFO] Renamed columns for Online Retail II schema: {rename_map}")

    # ── 2. Drop duplicates ───────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    print(f"  [CLEAN] Dropped {before - len(df):,} duplicate rows")

    # ── 3. Parse InvoiceDate ─────────────────────────────────────────────────
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")

    # ── 4. Filter cancelled orders (InvoiceNo starts with 'C') ───────────────
    df["IsCancelled"] = df["InvoiceNo"].astype(str).str.startswith("C")
    cancelled_count   = df["IsCancelled"].sum()
    print(f"  [CLEAN] Flagged {cancelled_count:,} cancelled transactions")
    df_clean = df[~df["IsCancelled"]].copy()

    # ── 5. Drop rows with missing CustomerID ─────────────────────────────────
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=["CustomerID"])
    print(f"  [CLEAN] Dropped {before - len(df_clean):,} rows with missing CustomerID")

    # ── 6. Remove negative/zero quantities & prices ──────────────────────────
    before = len(df_clean)
    df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] > 0)]
    print(f"  [CLEAN] Dropped {before - len(df_clean):,} rows with bad Quantity/Price")

    # ── 7. Convert types ─────────────────────────────────────────────────────
    df_clean["CustomerID"] = df_clean["CustomerID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    df_clean["Quantity"]   = df_clean["Quantity"].astype(int)
    df_clean["UnitPrice"]  = df_clean["UnitPrice"].astype(float)

    # ── 8. Derived columns ───────────────────────────────────────────────────
    df_clean["Revenue"]    = df_clean["Quantity"] * df_clean["UnitPrice"]
    df_clean["YearMonth"]  = df_clean["InvoiceDate"].dt.to_period("M")
    df_clean["Year"]       = df_clean["InvoiceDate"].dt.year
    df_clean["Month"]      = df_clean["InvoiceDate"].dt.month
    df_clean["DayOfWeek"]  = df_clean["InvoiceDate"].dt.day_name()
    df_clean["Hour"]       = df_clean["InvoiceDate"].dt.hour

    print(f"  [OK] Clean dataset: {len(df_clean):,} rows, {df_clean['CustomerID'].nunique():,} customers")
    return df_clean.reset_index(drop=True)


def get_clean_data(force_refresh: bool = False) -> pd.DataFrame:
    """Load and cache clean data. Returns cached parquet if available."""
    if not force_refresh and os.path.exists(CACHE_PATH):
        print("[CACHE] Loading from cache ...")
        return pd.read_parquet(CACHE_PATH)

    print("[INFO] Processing raw data ...")
    raw = load_raw_data()
    clean = clean_data(raw)
    os.makedirs(DATA_DIR, exist_ok=True)
    clean.to_parquet(CACHE_PATH, index=False)
    print(f"[OK] Cached to {CACHE_PATH}")
    return clean
