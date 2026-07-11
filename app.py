import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  dayfirst=True)
    return df

df = load_data()

page = st.sidebar.selectbox("Navigate to", [
    "Sales Overview",
    "Forecast Explorer",
    "Anomaly Report",
    "Product Demand Segments"
])

# ── Page 1: Sales Overview ─────────────────────────────────────
if page == "Sales Overview":
    st.title("Sales Overview Dashboard")

    region_filter   = st.selectbox("Filter by Region",   ["All"] + list(df["Region"].unique()))
    category_filter = st.selectbox("Filter by Category", ["All"] + list(df["Category"].unique()))

    filtered = df.copy()
    if region_filter   != "All": filtered = filtered[filtered["Region"]   == region_filter]
    if category_filter != "All": filtered = filtered[filtered["Category"] == category_filter]

    yearly = filtered.groupby(filtered["Order Date"].dt.year)["Sales"].sum()
    st.subheader("Total Sales by Year")
    st.bar_chart(yearly)

    monthly = filtered.resample("ME", on="Order Date")["Sales"].sum()
    st.subheader("Monthly Sales Trend")
    st.line_chart(monthly)

    st.subheader("Sales by Region")
    st.bar_chart(filtered.groupby("Region")["Sales"].sum())

# ── Page 2: Forecast Explorer ──────────────────────────────────
elif page == "Forecast Explorer":
    st.title("Forecast Explorer")

    segment_type  = st.selectbox("Select Segment Type", ["Category", "Region"])
    segment_value = st.selectbox("Select Value",
        list(df["Category"].unique()) if segment_type == "Category"
        else list(df["Region"].unique()))
    horizon = st.slider("Forecast Horizon (months)", 1, 3, 3)

    seg_df  = df[df[segment_type] == segment_value]
    monthly = seg_df.resample("ME", on="Order Date")["Sales"].sum().reset_index()
    monthly.columns = ["ds", "y"]

    m = Prophet(yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False)
    m.fit(monthly)
    future   = m.make_future_dataframe(periods=horizon, freq="ME")
    forecast = m.predict(future)

    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(monthly["ds"], monthly["y"], label="Actual")
    ax.plot(forecast["ds"].iloc[-horizon:],
            forecast["yhat"].iloc[-horizon:],
            "r--o", label="Forecast")
    ax.set_title(f"Forecast — {segment_value}")
    ax.legend()
    st.pyplot(fig)

    st.metric("MAE",  "See notebook Task 3")
    st.metric("RMSE", "See notebook Task 3")

# ── Page 3: Anomaly Report ─────────────────────────────────────
elif page == "Anomaly Report":
    from sklearn.ensemble import IsolationForest

    st.title("Anomaly Report")

    weekly = df.resample("W", on="Order Date")["Sales"].sum().to_frame(name="Sales")
    iso    = IsolationForest(contamination=0.05, random_state=42)
    weekly["anomaly"] = iso.fit_predict(weekly[["Sales"]])

    fig, ax = plt.subplots(figsize=(14,5))
    ax.plot(weekly.index, weekly["Sales"], label="Weekly Sales")
    anom = weekly[weekly["anomaly"] == -1]
    ax.scatter(anom.index, anom["Sales"],
               color="red", zorder=5, label="Anomaly", s=80)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Detected Anomaly Dates")
    st.dataframe(anom.reset_index()[["Order Date","Sales"]])

# ── Page 4: Product Demand Segments ───────────────────────────
elif page == "Product Demand Segments":
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    st.title("Product Demand Segments")

    subcat = df.groupby("Sub-Category")["Sales"].agg(
        total_sales="sum", avg_order_value="mean", volatility="std").reset_index()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(subcat[["total_sales","avg_order_value","volatility"]])
    km       = KMeans(n_clusters=4, random_state=42, n_init=10)
    subcat["Cluster"] = km.fit_predict(X_scaled)

    pca       = PCA(n_components=2)
    coords    = pca.fit_transform(X_scaled)
    subcat["PC1"] = coords[:,0]
    subcat["PC2"] = coords[:,1]

    fig, ax = plt.subplots(figsize=(10,6))
    for c in subcat["Cluster"].unique():
        mask = subcat["Cluster"] == c
        ax.scatter(subcat[mask]["PC1"], subcat[mask]["PC2"], label=f"Cluster {c}", s=100)
        for _, row in subcat[mask].iterrows():
            ax.annotate(row["Sub-Category"], (row["PC1"], row["PC2"]), fontsize=8)
    ax.legend()
    st.pyplot(fig)

    st.subheader("Sub-Category Cluster Assignments")
    st.dataframe(subcat[["Sub-Category","Cluster","total_sales","avg_order_value","volatility"]])