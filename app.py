"""
SalesIQ – AI-Powered Sales Forecasting
========================================
A Streamlit dashboard for exploring retail sales data and generating
demand forecasts with a pre-trained Random Forest model.

Enhancements over the original version:
  - Defensive data loading with column validation & clear user feedback
  - Hash-safe caching for uploaded files (no re-parsing on every rerun)
  - Missing-value / data-quality summary on the Dashboard
  - Feature-importance chart (pulled live from the model when available)
  - Forecast page shows historical context + a simple confidence band
  - Downloadable single-prediction CSV
  - Centralized constants, type hints, and small helper functions
  - Graceful degradation when optional columns/model are missing
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import copy

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "rf_model.joblib"
FEATURES_PATH = BASE_DIR / "models" / "feature_columns.json"
DEFAULT_DATA_PATH = BASE_DIR / "data" / "train.csv"

REQUIRED_COLUMNS = {"date", "store_nbr", "family", "sales", "onpromotion"}
WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

NOTEBOOK_METRICS = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "RMSE": [888.496620341228, 486.3318826660714],
    "R²": [0.5741, 0.8723907649751279],
})

PALETTE = ["#818CF8", "#F472B6", "#22D3EE", "#FBBF24", "#4ADE80", "#A78BFA", "#FB7185", "#2DD4BF"]

# A single vibrant, transparent Plotly theme applied to every chart in the app,
# so bars/lines/pies inherit the same colorful, "glassy" look automatically.
_theme = copy.deepcopy(pio.templates["plotly_dark"])
_theme.layout.paper_bgcolor = "rgba(0,0,0,0)"
_theme.layout.plot_bgcolor = "rgba(0,0,0,0)"
_theme.layout.colorway = PALETTE
_theme.layout.font = dict(family="Inter, Sora, sans-serif", color="#E5E7EB", size=13)
_theme.layout.title = dict(font=dict(size=18, family="Sora, sans-serif", color="#F1F5F9"))
_theme.layout.legend = dict(bgcolor="rgba(0,0,0,0)")
_theme.layout.xaxis = dict(gridcolor="rgba(148,163,184,.15)", zerolinecolor="rgba(148,163,184,.15)")
_theme.layout.yaxis = dict(gridcolor="rgba(148,163,184,.15)", zerolinecolor="rgba(148,163,184,.15)")
_theme.layout.margin = dict(l=10, r=10, t=55, b=10)
pio.templates["salesiq"] = _theme
pio.templates.default = "salesiq"
px.defaults.template = "salesiq"
px.defaults.color_discrete_sequence = PALETTE
px.defaults.color_continuous_scale = "Plasma"

st.set_page_config(
    page_title="SalesIQ – AI-Powered Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
    h1, h2, h3, .hero h1 {font-family: 'Sora', sans-serif;}

    .stApp {
        background: radial-gradient(circle at 10% 10%, rgba(99,102,241,.08), transparent 40%),
                    radial-gradient(circle at 90% 20%, rgba(236,72,153,.08), transparent 40%),
                    radial-gradient(circle at 50% 90%, rgba(6,182,212,.07), transparent 45%);
    }
    .block-container {padding-top: 1.3rem; padding-bottom: 2rem;}

    /* ---------- Hero ---------- */
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.4rem 2.6rem;
        border-radius: 26px;
        background: linear-gradient(120deg, #0f172a 0%, #1d4ed8 30%, #7c3aed 65%, #db2777 100%);
        background-size: 300% 300%;
        animation: gradientShift 10s ease infinite;
        color: white;
        margin-bottom: 1.3rem;
        box-shadow: 0 20px 55px rgba(76, 29, 149, .35);
    }
    @keyframes gradientShift {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .hero h1 {
        font-size: 3.1rem; margin-bottom: .4rem; font-weight: 800;
        background: linear-gradient(90deg, #ffffff, #e0e7ff, #ffffff);
        background-size: 200% auto;
        -webkit-background-clip: text; background-clip: text; color: transparent;
        animation: shine 4s linear infinite;
    }
    @keyframes shine {
        to {background-position: 200% center;}
    }
    .hero p {font-size: 1.1rem; opacity: .94; max-width: 640px;}
    .live-pill {
        display:inline-flex; align-items:center; gap:.45rem;
        background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.28);
        padding: .3rem .8rem; border-radius: 999px; font-size: .78rem; font-weight: 600;
        margin-bottom: .9rem; backdrop-filter: blur(6px);
    }
    .live-dot {
        width: 8px; height: 8px; border-radius: 50%; background: #4ade80;
        box-shadow: 0 0 0 rgba(74, 222, 128, .6);
        animation: pulse 1.6s infinite;
    }
    @keyframes pulse {
        0% {box-shadow: 0 0 0 0 rgba(74,222,128,.65);}
        70% {box-shadow: 0 0 0 9px rgba(74,222,128,0);}
        100% {box-shadow: 0 0 0 0 rgba(74,222,128,0);}
    }

    /* ---------- KPI cards ---------- */
    .kpi {
        position: relative;
        padding: 1.1rem 1.2rem;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,.08);
        background: linear-gradient(160deg, rgba(255,255,255,.05), rgba(255,255,255,.015));
        min-height: 112px;
        overflow: hidden;
        transition: transform .25s ease, box-shadow .25s ease;
    }
    .kpi:hover {transform: translateY(-5px) scale(1.015); box-shadow: 0 14px 30px rgba(0,0,0,.22);}
    .kpi::before {
        content:""; position:absolute; inset:0; border-radius:20px; padding:1px;
        background: var(--kpi-accent, linear-gradient(135deg,#6366F1,#EC4899));
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor; mask-composite: exclude;
        opacity: .55;
    }
    .kpi-label {font-size:.82rem; opacity:.75; font-weight:500;}
    .kpi-value {font-size:1.7rem; font-weight:800; margin-top:.3rem;}

    .section-title {
        font-size:1.5rem; font-weight:800; margin:1.1rem 0 .8rem;
        background: linear-gradient(90deg, #818cf8, #f472b6);
        -webkit-background-clip: text; background-clip: text; color: transparent;
        display:inline-block;
    }

    /* ---------- Forecast result ---------- */
    .result {
        padding: 1.8rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(99,102,241,.20), rgba(236,72,153,.20), rgba(6,182,212,.18));
        background-size: 200% 200%;
        animation: gradientShift 6s ease infinite;
        border: 1px solid rgba(255,255,255,.14);
        text-align:center;
        box-shadow: 0 18px 45px rgba(76,29,149,.25);
    }
    .result-value {
        font-size:3rem; font-weight:900;
        background: linear-gradient(90deg, #a5b4fc, #f9a8d4, #67e8f9);
        -webkit-background-clip: text; background-clip: text; color: transparent;
    }
    .muted {opacity:.72;}

    /* ---------- Badges ---------- */
    .badge {
        display:inline-flex; align-items:center; gap:.35rem;
        padding:.28rem .75rem; border-radius:999px;
        font-size:.78rem; font-weight:700;
    }
    .badge-ok {background:rgba(34,197,94,.16); color:#22c55e; border:1px solid rgba(34,197,94,.35);}
    .badge-warn {background:rgba(234,179,8,.16); color:#eab308; border:1px solid rgba(234,179,8,.35);}

    /* ---------- Misc polish ---------- */
    div[data-testid="stMetric"] {
        background: linear-gradient(160deg, rgba(255,255,255,.05), rgba(255,255,255,.01));
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 16px; padding: .8rem .9rem;
        transition: transform .2s ease;
    }
    div[data-testid="stMetric"]:hover {transform: translateY(-3px);}
    .stButton>button, .stDownloadButton>button {
        border-radius: 14px !important; font-weight: 700 !important;
        transition: transform .15s ease, box-shadow .15s ease !important;
    }
    .stButton>button:hover, .stDownloadButton>button:hover {
        transform: translateY(-2px); box-shadow: 0 10px 24px rgba(99,102,241,.35) !important;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(99,102,241,.07), rgba(0,0,0,0) 30%);
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained model + its expected feature column order.

    Returns (model, feature_columns, error_message). Any of the first two
    may be None if loading failed; error_message explains why.
    """
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        return None, None, "Model artifacts not found at models/rf_model.joblib and models/feature_columns.json."
    try:
        model = joblib.load(MODEL_PATH)
        feature_columns = json.loads(FEATURES_PATH.read_text())
        if not isinstance(feature_columns, list) or not feature_columns:
            return None, None, "feature_columns.json is empty or malformed."
        return model, feature_columns, None
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        return None, None, f"Failed to load model artifacts: {exc}"


def _prepare_dates(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_dataset_from_path(path_str: str) -> Optional[pd.DataFrame]:
    path = Path(path_str)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return _prepare_dates(df)


@st.cache_data(show_spinner=False)
def load_dataset_from_bytes(file_hash: str, file_bytes: bytes) -> pd.DataFrame:
    # file_hash is part of the cache key so identical re-uploads hit the cache
    # instead of re-parsing, while a genuinely new file always re-parses.
    df = pd.read_csv(pd.io.common.BytesIO(file_bytes))
    return _prepare_dates(df)


def validate_columns(df: pd.DataFrame) -> list[str]:
    """Return a list of required columns missing from df."""
    return sorted(REQUIRED_COLUMNS - set(df.columns))


def fmt_num(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1_000_000_000:
        return f"{sign}{x/1_000_000_000:.2f}B"
    if x >= 1_000_000:
        return f"{sign}{x/1_000_000:.2f}M"
    if x >= 1_000:
        return f"{sign}{x/1_000:.2f}K"
    return f"{sign}{x:,.2f}"


def make_prediction_row(date_value, store_nbr, family, onpromotion, feature_columns) -> pd.DataFrame:
    d = pd.Timestamp(date_value)
    row = pd.DataFrame([{
        "store_nbr": int(store_nbr),
        "onpromotion": int(onpromotion),
        "year": d.year,
        "month": d.month,
        "day": d.day,
        "day_of_week": d.dayofweek,
        "family": family,
    }])
    row = pd.get_dummies(row, columns=["family"], drop_first=True)
    row = row.reindex(columns=feature_columns, fill_value=0)
    return row


def get_feature_importance(model, feature_columns) -> Optional[pd.DataFrame]:
    """Pull feature_importances_ off the model if it's tree-based."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None or len(importances) != len(feature_columns):
        return None
    return (
        pd.DataFrame({"feature": feature_columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("📈 SalesIQ")
st.sidebar.caption("AI-Powered Sales Forecasting")
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Sales Forecast", "Analytics", "Data Explorer",
     "Model Performance", "Business Insights", "About Project"],
)

st.sidebar.divider()

model, feature_columns, model_error = load_model()
if model is not None:
    st.sidebar.markdown('<span class="badge badge-ok">Model loaded</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<span class="badge badge-warn">Model unavailable</span>', unsafe_allow_html=True)
st.sidebar.caption("Random Forest Regressor · 50 trees · max depth 10")

uploaded = st.sidebar.file_uploader("Optional: upload train.csv", type=["csv"])

if st.sidebar.button("🔄 Clear cache & reload", use_container_width=True):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
df: Optional[pd.DataFrame] = None
data_error: Optional[str] = None

try:
    if uploaded is not None:
        raw_bytes = uploaded.getvalue()
        file_hash = hashlib.md5(raw_bytes).hexdigest()
        df = load_dataset_from_bytes(file_hash, raw_bytes)
    elif DEFAULT_DATA_PATH.exists():
        with st.spinner("Loading sales data..."):
            df = load_dataset_from_path(str(DEFAULT_DATA_PATH))
except Exception as exc:  # noqa: BLE001
    data_error = f"Could not read the CSV file: {exc}"

if df is not None:
    missing_cols = validate_columns(df)
    if missing_cols:
        data_error = f"Dataset is missing required column(s): {', '.join(missing_cols)}"
        df = None
    elif df["date"].isna().all():
        data_error = "The 'date' column could not be parsed for any row."
        df = None

if data_error:
    st.sidebar.error(data_error)

# --------------------------------------------------------------------------
# Dashboard
# --------------------------------------------------------------------------
if page == "Dashboard":
    st.markdown(f"""
    <div class="hero">
        <div class="live-pill"><span class="live-dot"></span> LIVE DASHBOARD &nbsp;·&nbsp; {pd.Timestamp.now().strftime('%b %d, %Y — %H:%M')}</div>
        <h1>Predict Sales with AI 🚀</h1>
        <p>SalesIQ turns historical retail data into practical sales forecasts and business insights using machine learning.</p>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.warning("Dataset not found or invalid. Put the Kaggle train.csv at data/train.csv, or upload a valid file from the sidebar.")
    else:
        total_sales = df["sales"].sum()
        avg_sales = df["sales"].mean()
        stores = df["store_nbr"].nunique()
        families = df["family"].nunique()
        promo_rows = (df["onpromotion"] > 0).sum()

        cols = st.columns(5)
        cards = [
            ("💰 Total Sales", fmt_num(total_sales)),
            ("📊 Average Sale", fmt_num(avg_sales)),
            ("🏬 Stores", f"{stores:,}"),
            ("🛍️ Product Families", f"{families:,}"),
            ("🏷️ Promotion Records", f"{promo_rows:,}"),
        ]
        for col, (label, value), color in zip(cols, cards, PALETTE):
            with col:
                st.markdown(
                    f'<div class="kpi" style="--kpi-accent: linear-gradient(135deg,{color},transparent);">'
                    f'<div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown('<div class="section-title">📈 Sales Trend Preview</div>', unsafe_allow_html=True)
        daily = df.dropna(subset=["date"]).groupby("date", as_index=False)["sales"].sum()
        fig = px.line(daily, x="date", y="sales", title="Daily Total Sales")
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("📋 Data quality summary"):
            n_missing_dates = df["date"].isna().sum()
            dup_rows = df.duplicated().sum()
            neg_sales = (df["sales"] < 0).sum()
            q1, q2, q3 = st.columns(3)
            q1.metric("Rows", f"{len(df):,}")
            q2.metric("Unparseable dates", f"{n_missing_dates:,}")
            q3.metric("Duplicate rows", f"{dup_rows:,}")
            if neg_sales:
                st.caption(f"⚠️ {neg_sales:,} rows have negative sales values (returns/adjustments).")

# --------------------------------------------------------------------------
# Sales Forecast
# --------------------------------------------------------------------------
elif page == "Sales Forecast":
    st.title("🔮 Sales Forecast")
    st.caption("Generate a prediction using the saved Random Forest model and the same feature logic used in the notebook.")

    if model is None or feature_columns is None:
        st.error(
            f"The trained model artifact is missing or invalid ({model_error}). "
            "Run train_model.py once to create models/rf_model.joblib and models/feature_columns.json. "
            "No prediction is generated without the trained model."
        )
    else:
        if df is not None:
            min_date = df["date"].min().date()
            max_date = df["date"].max().date()
            stores = sorted(df["store_nbr"].dropna().unique().tolist())
            families = sorted(df["family"].dropna().astype(str).unique().tolist())
            default_store = int(stores[0]) if stores else 1
            default_family = families[0] if families else "AUTOMOTIVE"
        else:
            min_date, max_date = pd.Timestamp("2013-01-01").date(), pd.Timestamp("2017-08-15").date()
            default_store, default_family = 1, "AUTOMOTIVE"
            stores, families = list(range(1, 55)), []

        c1, c2 = st.columns(2)
        with c1:
            forecast_date = st.date_input("Forecast Date", value=max_date, min_value=min_date, max_value=max_date)
            store_nbr = st.selectbox("Store Number", stores, index=0)
        with c2:
            family = st.selectbox("Product Family", families if families else ["AUTOMOTIVE"])
            onpromotion = st.number_input("Number of Items on Promotion", min_value=0, value=0, step=1)

        if st.button("✨ Generate Forecast", type="primary", use_container_width=True):
            try:
                with st.spinner("Running Random Forest prediction..."):
                    X_input = make_prediction_row(forecast_date, store_nbr, family, onpromotion, feature_columns)
                    prediction = float(model.predict(X_input)[0])

                    # Simple uncertainty estimate from individual tree predictions,
                    # when the underlying model exposes its trees (e.g. RandomForest).
                    tree_preds = None
                    estimators = getattr(model, "estimators_", None)
                    if estimators is not None:
                        try:
                            tree_preds = np.array([t.predict(X_input)[0] for t in estimators])
                        except Exception:
                            tree_preds = None

                st.markdown(f"""
                <div class="result">
                    <div class="muted">🔮 Predicted Sales</div>
                    <div class="result-value">{max(prediction, 0):,.2f}</div>
                    <div class="muted">Model: Random Forest Regressor</div>
                </div>
                """, unsafe_allow_html=True)
                st.balloons()

                st.write("")
                a, b, c, d = st.columns(4)
                a.metric("Date", str(forecast_date))
                b.metric("Store", store_nbr)
                c.metric("Family", family)
                d.metric("Promotion", int(onpromotion))

                if tree_preds is not None:
                    lo, hi = np.percentile(tree_preds, [10, 90])
                    st.caption(f"Approximate 80% range across trees: **{max(lo,0):,.0f} – {hi:,.0f}**")

                # Historical context for the same store/family, if data is available
                if df is not None:
                    hist = df[(df["store_nbr"] == store_nbr) & (df["family"].astype(str) == str(family))]
                    if not hist.empty:
                        hist_avg = hist["sales"].mean()
                        delta = prediction - hist_avg
                        st.info(
                            f"Historical average for Store {store_nbr} / {family}: "
                            f"**{hist_avg:,.2f}** ({'above' if delta >= 0 else 'below'} average by {abs(delta):,.2f})."
                        )

                st.info("The prediction uses year, month, day and day_of_week derived from the selected date, plus store, promotion and the one-hot encoded family features. The final row is aligned to the saved training feature order.")

                pred_df = pd.DataFrame([{
                    "date": forecast_date, "store_nbr": store_nbr, "family": family,
                    "onpromotion": onpromotion, "predicted_sales": round(prediction, 2),
                }])
                st.download_button(
                    "⬇️ Download this prediction as CSV",
                    pred_df.to_csv(index=False).encode("utf-8"),
                    "salesiq_prediction.csv", "text/csv",
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# --------------------------------------------------------------------------
# Analytics
# --------------------------------------------------------------------------
elif page == "Analytics":
    st.title("📊 Analytics")
    if df is None:
        st.warning("Load the dataset from data/train.csv or upload train.csv.")
    else:
        clean = df.dropna(subset=["date"])
        tab1, tab2, tab3 = st.tabs(["Time Trends", "Products & Stores", "Promotions"])

        with tab1:
            daily = clean.groupby("date", as_index=False)["sales"].sum()
            monthly = (
                clean.assign(month_period=clean["date"].dt.to_period("M").astype(str))
                .groupby("month_period", as_index=False)["sales"].sum()
            )
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.line(daily, x="date", y="sales", title="Daily Sales"), use_container_width=True)
            with c2:
                st.plotly_chart(px.bar(monthly, x="month_period", y="sales", title="Monthly Sales"), use_container_width=True)

            weekday = clean.assign(day_name=clean["date"].dt.day_name()).groupby("day_name", as_index=False)["sales"].mean()
            weekday["order"] = weekday["day_name"].map({x: i for i, x in enumerate(WEEKDAY_ORDER)})
            weekday = weekday.sort_values("order")
            st.plotly_chart(px.bar(weekday, x="day_name", y="sales", title="Average Sales by Weekday"), use_container_width=True)

        with tab2:
            family_sales = df.groupby("family", as_index=False)["sales"].sum().sort_values("sales", ascending=False).head(15)
            store_sales = df.groupby("store_nbr", as_index=False)["sales"].sum().sort_values("sales", ascending=False).head(20)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.bar(family_sales, x="sales", y="family", orientation="h", title="Top Product Families"), use_container_width=True)
            with c2:
                st.plotly_chart(px.bar(store_sales, x="store_nbr", y="sales", title="Top Stores"), use_container_width=True)

        with tab3:
            promo = df.assign(promotion=df["onpromotion"].gt(0).map({True: "Promotion", False: "No Promotion"}))
            promo_sales = promo.groupby("promotion", as_index=False)["sales"].mean()
            st.plotly_chart(px.bar(promo_sales, x="promotion", y="sales", title="Average Sales: Promotion vs No Promotion"), use_container_width=True)

# --------------------------------------------------------------------------
# Data Explorer
# --------------------------------------------------------------------------
elif page == "Data Explorer":
    st.title("🔎 Data Explorer")
    if df is None:
        st.warning("Load the dataset from data/train.csv or upload train.csv.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            min_d, max_d = df["date"].min().date(), df["date"].max().date()
            date_range = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
        with c2:
            store_filter = st.multiselect("Stores", sorted(df["store_nbr"].dropna().unique().tolist()), max_selections=10)
        with c3:
            family_filter = st.multiselect("Product Families", sorted(df["family"].dropna().astype(str).unique().tolist()), max_selections=10)

        filtered = df
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
            filtered = filtered[(filtered["date"] >= start) & (filtered["date"] <= end)]
        if store_filter:
            filtered = filtered[filtered["store_nbr"].isin(store_filter)]
        if family_filter:
            filtered = filtered[filtered["family"].isin(family_filter)]

        st.caption(f"Showing up to 500 rows from {len(filtered):,} matching records.")
        preview = filtered.head(500)
        st.dataframe(preview, use_container_width=True, height=500)

        csv = preview.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download visible rows as CSV", csv, "salesiq_filtered_data.csv", "text/csv")

# --------------------------------------------------------------------------
# Model Performance
# --------------------------------------------------------------------------
elif page == "Model Performance":
    st.title("🤖 Model Performance")
    st.caption("Evaluation values below come from the project's notebook/report.")
    metrics = NOTEBOOK_METRICS

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(metrics, x="Model", y="RMSE", title="RMSE (Lower is Better)"), use_container_width=True)
    with c2:
        st.plotly_chart(px.bar(metrics, x="Model", y="R²", title="R² Score (Higher is Better)"), use_container_width=True)

    st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.info("MAPE is not shown as a primary metric because the project contains many zero-sales observations, which makes percentage-based error unstable.")

    if model is not None and feature_columns is not None:
        importance_df = get_feature_importance(model, feature_columns)
        if importance_df is not None:
            st.markdown('<div class="section-title">🧠 Top Feature Importances</div>', unsafe_allow_html=True)
            fig = px.bar(
                importance_df.sort_values("importance"),
                x="importance", y="feature", orientation="h",
                title="Random Forest Feature Importance (top 15)",
            )
            fig.update_layout(height=450, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Feature importances are not available for this model type.")

# --------------------------------------------------------------------------
# Business Insights
# --------------------------------------------------------------------------
elif page == "Business Insights":
    st.title("💡 Business Insights")
    insights = [
        ("📦 Inventory Planning", "Use forecasted high-demand periods to improve stock availability and reduce stockout risk."),
        ("🎯 Promotion Planning", "Use demand patterns to identify lower-demand periods where promotions may help stabilize sales."),
        ("🛒 Product Strategy", "Compare product-family demand and focus planning resources on consistently strong categories."),
        ("👥 Workforce Planning", "Use expected demand to support staffing and scheduling decisions."),
        ("💰 Financial Planning", "Forecasted sales can provide a data-driven reference for budgeting and cash-flow planning."),
    ]
    for title, text in insights:
        st.markdown(f"### {title}")
        st.write(text)

# --------------------------------------------------------------------------
# About
# --------------------------------------------------------------------------
elif page == "About Project":
    st.title("ℹ️ About SalesIQ")
    st.markdown("""
    **Sales Forecasting Using Machine Learning** was developed during the internship at **INTERNSHIP STUDIO**.

    **Workflow**

    Dataset → Data Cleaning → EDA → Feature Engineering → Chronological 80/20 Train/Test Split → Linear Regression Baseline → Random Forest → Evaluation → Sales Prediction → Business Insights

    **Feature engineering**
    - `year` = date year
    - `month` = date month
    - `day` = date day
    - `day_of_week` = pandas day-of-week value
    - `family` = one-hot encoded with `drop_first=True`
    - `sales`, `date`, and `id` are excluded from X
    - final training matrix = 38 features

    **Random Forest**
    - n_estimators = 50
    - max_depth = 10
    - random_state = 42
    - n_jobs = -1

    **Evaluation**
    - Linear Regression: RMSE 888.50, R² 0.5741
    - Random Forest: RMSE 486.33, R² 0.8724
    """)