import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

#  Page 1: Sales Overview 
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

    # Fixed: Use set_index() for resample with non-index column
    monthly = filtered.set_index("Order Date").resample("ME")["Sales"].sum()
    st.subheader("Monthly Sales Trend")
    st.line_chart(monthly)

    st.subheader("Sales by Region")
    st.bar_chart(filtered.groupby("Region")["Sales"].sum())

#  Page 2: Forecast Explorer 
elif page == "Forecast Explorer":
    st.title("Forecast Explorer")

    segment_type  = st.selectbox("Select Segment Type", ["Category", "Region"])
    segment_value = st.selectbox("Select Value",
        list(df["Category"].unique()) if segment_type == "Category"
        else list(df["Region"].unique()))
    horizon = st.slider("Forecast Horizon (months)", 1, 12, 3)

    try:
        seg_df  = df[df[segment_type] == segment_value]
        
        # Fixed: Use set_index() for proper resample
        monthly = seg_df.set_index("Order Date").resample("ME")["Sales"].sum().reset_index()
        monthly.columns = ["ds", "y"]
        
        # Validate data
        if len(monthly) < 2:
            st.error("Not enough data for forecasting. Please select a segment with more history.")
        else:
            m = Prophet(yearly_seasonality=True,
                        weekly_seasonality=False,
                        daily_seasonality=False)
            m.fit(monthly)
            future   = m.make_future_dataframe(periods=horizon, freq="ME")
            forecast = m.predict(future)

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(monthly["ds"], monthly["y"], label="Actual", linewidth=2)
            ax.plot(forecast["ds"].iloc[-horizon:],
                    forecast["yhat"].iloc[-horizon:],
                    "r--o", label="Forecast", linewidth=2, markersize=8)
            ax.fill_between(forecast["ds"].iloc[-horizon:],
                            forecast["yhat_lower"].iloc[-horizon:],
                            forecast["yhat_upper"].iloc[-horizon:],
                            alpha=0.2, color="red", label="95% Confidence Interval")
            ax.set_xlabel("Date")
            ax.set_ylabel("Sales")
            ax.set_title(f"Forecast — {segment_value}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # Fixed: Calculate actual metrics
            forecast_values = forecast[["ds", "yhat"]].tail(len(monthly)).reset_index(drop=True)
            mae = mean_absolute_error(monthly["y"].values, forecast_values["yhat"].values)
            rmse = np.sqrt(mean_squared_error(monthly["y"].values, forecast_values["yhat"].values))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("MAE (Mean Absolute Error)", f"${mae:,.2f}")
            with col2:
                st.metric("RMSE (Root Mean Squared Error)", f"${rmse:,.2f}")
                
    except Exception as e:
        st.error(f"Error generating forecast: {str(e)}")

#  Page 3: Anomaly Report 
elif page == "Anomaly Report":
    st.title("Anomaly Report")

    try:
        # Fixed: Use set_index() for proper resample
        weekly = df.set_index("Order Date").resample("W")["Sales"].sum().to_frame(name="Sales")
        
        contamination = st.slider("Anomaly Contamination Rate", 0.01, 0.20, 0.05)
        iso    = IsolationForest(contamination=contamination, random_state=42)
        weekly["anomaly"] = iso.fit_predict(weekly[["Sales"]])

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(weekly.index, weekly["Sales"], label="Weekly Sales", linewidth=2)
        anom = weekly[weekly["anomaly"] == -1]
        ax.scatter(anom.index, anom["Sales"],
                   color="red", zorder=5, label="Anomaly", s=100)
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.subheader(f"Detected Anomaly Dates ({len(anom)} anomalies)")
        if len(anom) > 0:
            # Fixed: Rename index to match expected column name
            anom_display = anom.reset_index()
            anom_display.columns = ["Date", "Sales", "Anomaly"]
            st.dataframe(anom_display[["Date", "Sales"]], use_container_width=True)
        else:
            st.info("No anomalies detected.")
            
    except Exception as e:
        st.error(f"Error in anomaly detection: {str(e)}")

#  Page 4: Product Demand Segments 
elif page == "Product Demand Segments":
    st.title("Product Demand Segments")

    try:
        subcat = df.groupby("Sub-Category")["Sales"].agg(
            total_sales="sum", avg_order_value="mean", volatility="std").reset_index()

        # User-configurable parameters
        n_clusters = st.slider("Number of Clusters", 2, 10, 4)

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(subcat[["total_sales", "avg_order_value", "volatility"]])
        km       = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        subcat["Cluster"] = km.fit_predict(X_scaled)

        pca       = PCA(n_components=2)
        coords    = pca.fit_transform(X_scaled)
        subcat["PC1"] = coords[:, 0]
        subcat["PC2"] = coords[:, 1]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        for c in sorted(subcat["Cluster"].unique()):
            mask = subcat["Cluster"] == c
            ax.scatter(subcat[mask]["PC1"], subcat[mask]["PC2"], 
                      label=f"Cluster {c}", s=100, color=colors[c], alpha=0.7)
            for _, row in subcat[mask].iterrows():
                ax.annotate(row["Sub-Category"], (row["PC1"], row["PC2"]), 
                           fontsize=8, alpha=0.8)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        ax.set_title("Product Demand Segments (PCA Visualization)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)

        st.subheader("Sub-Category Cluster Assignments")
        st.dataframe(subcat[["Sub-Category", "Cluster", "total_sales", "avg_order_value", "volatility"]].sort_values("Cluster"), use_container_width=True)
        
        # Show cluster statistics
        st.subheader("Cluster Statistics")
        cluster_stats = subcat.groupby("Cluster")[["total_sales", "avg_order_value", "volatility"]].agg(["mean", "count"])
        st.dataframe(cluster_stats, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error in clustering analysis: {str(e)}")
