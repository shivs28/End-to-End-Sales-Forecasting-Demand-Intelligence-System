"""
ðŸ“Š Sales Forecasting & Demand Intelligence Dashboard
Streamlit App â€” Interactive dashboard for sales analysis, forecasting,
anomaly detection, and product demand segmentation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Resolve all paths relative to this script's directory
BASE_DIR = Path(__file__).resolve().parent

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Page Configuration
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence",
    page_icon="ðŸ“Š",
    layout="wide",
    initial_sidebar_state="expanded"
)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Custom CSS
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1a1a2e, #16213e, #0f3460);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.9;
        margin-top: 0.3rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Data Loading
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
@st.cache_data
def load_data():
    """Load all required data files."""
    csv_path = BASE_DIR / 'train.csv'
    df = pd.read_csv(csv_path, encoding='latin1')
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y')
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Month_Name'] = df['Order Date'].dt.strftime('%b')
    df['Quarter'] = df['Order Date'].dt.quarter
    return df


@st.cache_data
def load_model_data():
    """Load pre-computed model artifacts."""
    data = {}
    
    model_dir = BASE_DIR / 'models'
    
    if os.path.exists(f'{model_dir}/monthly_sales.csv'):
        ms = pd.read_csv(f'{model_dir}/monthly_sales.csv', index_col=0, parse_dates=True)
        data['monthly_sales'] = ms
    
    if os.path.exists(f'{model_dir}/weekly_anomalies.csv'):
        wa = pd.read_csv(f'{model_dir}/weekly_anomalies.csv', index_col=0, parse_dates=True)
        data['weekly_anomalies'] = wa
    
    if os.path.exists(f'{model_dir}/anomalies_isolation_forest.csv'):
        aif = pd.read_csv(f'{model_dir}/anomalies_isolation_forest.csv', index_col=0, parse_dates=True)
        data['anomalies_if'] = aif
    
    if os.path.exists(f'{model_dir}/anomalies_zscore.csv'):
        azs = pd.read_csv(f'{model_dir}/anomalies_zscore.csv', index_col=0, parse_dates=True)
        data['anomalies_zs'] = azs
    
    if os.path.exists(f'{model_dir}/cluster_info.csv'):
        ci = pd.read_csv(f'{model_dir}/cluster_info.csv')
        data['cluster_info'] = ci
    
    if os.path.exists(f'{model_dir}/cluster_features.csv'):
        cf = pd.read_csv(f'{model_dir}/cluster_features.csv')
        data['cluster_features'] = cf
    
    if os.path.exists(f'{model_dir}/model_comparison.csv'):
        mc = pd.read_csv(f'{model_dir}/model_comparison.csv')
        data['model_comparison'] = mc
    
    if os.path.exists(f'{model_dir}/segment_forecasts.json'):
        with open(f'{model_dir}/segment_forecasts.json', 'r') as f:
            data['segment_forecasts'] = json.load(f)
    
    return data


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Inline Forecasting (if model files not yet generated)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def generate_forecast_inline(df, category=None, region=None, horizon=3):
    """Generate a forecast using Prophet directly in the app."""
    try:
        from prophet import Prophet
        
        mask = pd.Series(True, index=df.index)
        if category and category != 'All':
            mask = mask & (df['Category'] == category)
        if region and region != 'All':
            mask = mask & (df['Region'] == region)
        
        filtered = df[mask].copy()
        monthly = filtered.groupby(filtered['Order Date'].dt.to_period('M'))['Sales'].sum()
        monthly.index = monthly.index.to_timestamp()
        monthly = monthly.sort_index()
        
        prophet_df = monthly.reset_index()
        prophet_df.columns = ['ds', 'y']
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(prophet_df)
        
        future = model.make_future_dataframe(periods=horizon, freq='MS')
        forecast = model.predict(future)
        
        # Calculate metrics on last 3 months
        if len(monthly) > 3:
            test_actual = monthly.values[-3:]
            test_pred = forecast[forecast['ds'].isin(monthly.index[-3:])]['yhat'].values[-3:]
            if len(test_pred) == 3:
                mae = np.mean(np.abs(test_actual - test_pred))
                rmse = np.sqrt(np.mean((test_actual - test_pred) ** 2))
            else:
                mae, rmse = None, None
        else:
            mae, rmse = None, None
        
        return forecast, monthly, mae, rmse
    
    except Exception as e:
        st.error(f"Forecast error: {e}")
        return None, None, None, None


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Load Data
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
try:
    df = load_data()
    model_data = load_model_data()
    data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Make sure `train.csv` is in the same directory as this app.")
    data_loaded = False

if data_loaded:
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Sidebar
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    with st.sidebar:
        st.markdown("## ðŸ“Š Navigation")
        page = st.radio(
            "Go to",
            ["ðŸ  Sales Overview", "ðŸ”® Forecast Explorer", "ðŸš¨ Anomaly Report", "ðŸ“¦ Demand Segments"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### ðŸ“… Data Range")
        st.markdown(f"**{df['Order Date'].min().strftime('%b %Y')}** to **{df['Order Date'].max().strftime('%b %Y')}**")
        st.markdown(f"**{len(df):,}** total transactions")
        st.markdown(f"**${df['Sales'].sum():,.0f}** total revenue")

    # ==============================================================
    # PAGE 1: Sales Overview Dashboard
    # ==============================================================
    if page == "ðŸ  Sales Overview":
        st.markdown('<h1 class="main-header">ðŸ“Š Sales Overview Dashboard</h1>', unsafe_allow_html=True)
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            selected_region = st.multiselect("Filter by Region", df['Region'].unique(), default=df['Region'].unique())
        with col2:
            selected_category = st.multiselect("Filter by Category", df['Category'].unique(), default=df['Category'].unique())
        
        filtered = df[df['Region'].isin(selected_region) & df['Category'].isin(selected_category)]
        
        # KPI Metrics
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.metric("Total Revenue", f"${filtered['Sales'].sum():,.0f}")
        with m2:
            st.metric("Avg Order Value", f"${filtered['Sales'].mean():,.0f}")
        with m3:
            st.metric("Total Orders", f"{len(filtered):,}")
        with m4:
            yoy = filtered.groupby('Year')['Sales'].sum()
            if len(yoy) >= 2:
                growth = ((yoy.iloc[-1] - yoy.iloc[-2]) / yoy.iloc[-2]) * 100
                st.metric("YoY Growth", f"{growth:.1f}%")
            else:
                st.metric("YoY Growth", "N/A")
        
        st.markdown("---")
        
        # Charts
        c1, c2 = st.columns(2)
        
        with c1:
            # Total sales by year
            yearly = filtered.groupby('Year')['Sales'].sum().reset_index()
            fig = px.bar(
                yearly, x='Year', y='Sales',
                title='Total Sales by Year',
                color_discrete_sequence=['#667eea'],
                text_auto='${:,.0f}'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter'),
                yaxis_title='Sales ($)',
                xaxis_title='Year',
                showlegend=False
            )
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, width='stretch')
        
        with c2:
            # Sales by category
            cat_sales = filtered.groupby('Category')['Sales'].sum().reset_index()
            fig = px.pie(
                cat_sales, values='Sales', names='Category',
                title='Revenue Distribution by Category',
                color_discrete_sequence=['#667eea', '#764ba2', '#f093fb'],
                hole=0.4
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter')
            )
            st.plotly_chart(fig, width='stretch')
        
        # Monthly trend line
        monthly = filtered.groupby(filtered['Order Date'].dt.to_period('M'))['Sales'].sum()
        monthly.index = monthly.index.to_timestamp()
        monthly = monthly.sort_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly.index, y=monthly.values,
            mode='lines+markers',
            line=dict(color='#667eea', width=2),
            marker=dict(size=5),
            fill='tonexty',
            fillcolor='rgba(102, 126, 234, 0.1)',
            name='Monthly Sales'
        ))
        fig.update_layout(
            title='Monthly Sales Trend',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter'),
            yaxis_title='Sales ($)',
            xaxis_title='Date',
            hovermode='x unified'
        )
        st.plotly_chart(fig, width='stretch')
        
        # Sales by region
        c3, c4 = st.columns(2)
        with c3:
            region_sales = filtered.groupby('Region')['Sales'].sum().reset_index().sort_values('Sales', ascending=True)
            fig = px.bar(
                region_sales, x='Sales', y='Region',
                orientation='h',
                title='Sales by Region',
                color='Sales',
                color_continuous_scale='Viridis',
                text_auto='${:,.0f}'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter'),
                showlegend=False,
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, width='stretch')
        
        with c4:
            # Top sub-categories
            subcat_sales = filtered.groupby('Sub-Category')['Sales'].sum().nlargest(10).reset_index()
            fig = px.bar(
                subcat_sales, x='Sales', y='Sub-Category',
                orientation='h',
                title='Top 10 Sub-Categories by Sales',
                color='Sales',
                color_continuous_scale='Plasma',
                text_auto='${:,.0f}'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter'),
                showlegend=False,
                coloraxis_showscale=False,
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig, width='stretch')

    # ==============================================================
    # PAGE 2: Forecast Explorer
    # ==============================================================
    elif page == "ðŸ”® Forecast Explorer":
        st.markdown('<h1 class="main-header">ðŸ”® Forecast Explorer</h1>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            forecast_type = st.selectbox("Segment Type", ["Category", "Region"])
        
        with col2:
            if forecast_type == "Category":
                segment = st.selectbox("Select Segment", ["All"] + sorted(df['Category'].unique().tolist()))
            else:
                segment = st.selectbox("Select Segment", ["All"] + sorted(df['Region'].unique().tolist()))
        
        with col3:
            horizon = st.slider("Forecast Horizon (Months)", 1, 3, 3)
        
        st.markdown("---")
        
        with st.spinner("Generating forecast..."):
            if forecast_type == "Category":
                forecast, actual, mae, rmse = generate_forecast_inline(df, category=segment, horizon=horizon)
            else:
                forecast, actual, mae, rmse = generate_forecast_inline(df, region=segment, horizon=horizon)
        
        if forecast is not None and actual is not None:
            # Plot
            fig = go.Figure()
            
            # Actual data
            fig.add_trace(go.Scatter(
                x=actual.index, y=actual.values,
                mode='lines+markers',
                name='Actual Sales',
                line=dict(color='#2c3e50', width=2),
                marker=dict(size=4)
            ))
            
            # Forecast
            future_mask = forecast['ds'] > actual.index[-1]
            future_fc = forecast[future_mask].head(horizon)
            
            fig.add_trace(go.Scatter(
                x=future_fc['ds'], y=future_fc['yhat'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='#e74c3c', width=2, dash='dash'),
                marker=dict(size=8, symbol='diamond')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=pd.concat([future_fc['ds'], future_fc['ds'][::-1]]),
                y=pd.concat([future_fc['yhat_upper'], future_fc['yhat_lower'][::-1]]),
                fill='toself',
                fillcolor='rgba(231, 76, 60, 0.1)',
                line=dict(color='rgba(231, 76, 60, 0)'),
                name='95% Confidence'
            ))
            
            title_segment = segment if segment != "All" else f"All {forecast_type}s"
            fig.update_layout(
                title=f'Sales Forecast â€” {title_segment} ({horizon}-Month Horizon)',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter'),
                yaxis_title='Sales ($)',
                xaxis_title='Date',
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            st.plotly_chart(fig, width='stretch')
            
            # Metrics
            if mae is not None and rmse is not None:
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("Mean Absolute Error (MAE)", f"${mae:,.0f}")
                with m2:
                    st.metric("Root Mean Squared Error (RMSE)", f"${rmse:,.0f}")
            
            # Forecast table
            st.markdown("### ðŸ“‹ Forecast Values")
            fc_table = future_fc[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            fc_table.columns = ['Date', 'Forecast ($)', 'Lower Bound ($)', 'Upper Bound ($)']
            fc_table['Date'] = fc_table['Date'].dt.strftime('%B %Y')
            st.dataframe(
                fc_table.style.format({
                    'Forecast ($)': '${:,.0f}',
                    'Lower Bound ($)': '${:,.0f}',
                    'Upper Bound ($)': '${:,.0f}'
                }),
                width='stretch',
                hide_index=True
            )
            
            # Model comparison (if available)
            if 'model_comparison' in model_data:
                st.markdown("### ðŸ“Š Model Comparison (Overall)")
                mc = model_data['model_comparison']
                st.dataframe(mc, width='stretch', hide_index=True)

    # ==============================================================
    # PAGE 3: Anomaly Report
    # ==============================================================
    elif page == "ðŸš¨ Anomaly Report":
        st.markdown('<h1 class="main-header">ðŸš¨ Anomaly Detection Report</h1>', unsafe_allow_html=True)
        
        if 'weekly_anomalies' in model_data:
            wa = model_data['weekly_anomalies']
            
            # Tab for different methods
            tab1, tab2 = st.tabs(["ðŸŒ² Isolation Forest", "ðŸ“ Z-Score Method"])
            
            with tab1:
                anomalies_if = model_data.get('anomalies_if', wa[wa['IF_Anomaly'] == -1])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=wa.index, y=wa['Sales'],
                    mode='lines',
                    name='Weekly Sales',
                    line=dict(color='#2c3e50', width=1),
                    opacity=0.7
                ))
                fig.add_trace(go.Scatter(
                    x=anomalies_if.index, y=anomalies_if['Sales'],
                    mode='markers',
                    name=f'Anomalies ({len(anomalies_if)})',
                    marker=dict(color='#e74c3c', size=10, line=dict(width=1, color='black'))
                ))
                fig.update_layout(
                    title='Anomaly Detection â€” Isolation Forest',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter'),
                    yaxis_title='Weekly Sales ($)',
                    xaxis_title='Date',
                    hovermode='x unified'
                )
                st.plotly_chart(fig, width='stretch')
                
                st.markdown("### ðŸ“‹ Detected Anomalies (Isolation Forest)")
                if len(anomalies_if) > 0:
                    anomaly_table = anomalies_if[['Sales']].copy()
                    anomaly_table.index.name = 'Date'
                    anomaly_table = anomaly_table.reset_index()
                    anomaly_table['Date'] = pd.to_datetime(anomaly_table['Date']).dt.strftime('%Y-%m-%d')
                    anomaly_table['Sales'] = anomaly_table['Sales'].apply(lambda x: f'${x:,.0f}')
                    st.dataframe(anomaly_table, width='stretch', hide_index=True)
            
            with tab2:
                if 'ZS_Anomaly' in wa.columns:
                    anomalies_zs = model_data.get('anomalies_zs', wa[wa['ZS_Anomaly'] == 1])
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=wa.index, y=wa['Sales'],
                        mode='lines', name='Weekly Sales',
                        line=dict(color='#2c3e50', width=1), opacity=0.7
                    ))
                    if 'Rolling_Mean' in wa.columns:
                        fig.add_trace(go.Scatter(
                            x=wa.index, y=wa['Rolling_Mean'],
                            mode='lines', name='Rolling Mean',
                            line=dict(color='#3498db', width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=wa.index, y=wa['Rolling_Mean'] + 2 * wa['Rolling_Std'],
                            mode='lines', name='+2Ïƒ',
                            line=dict(color='#3498db', width=1, dash='dash'), opacity=0.5
                        ))
                        fig.add_trace(go.Scatter(
                            x=wa.index, y=wa['Rolling_Mean'] - 2 * wa['Rolling_Std'],
                            mode='lines', name='-2Ïƒ',
                            line=dict(color='#3498db', width=1, dash='dash'), opacity=0.5,
                            fill='tonexty', fillcolor='rgba(52, 152, 219, 0.08)'
                        ))
                    
                    fig.add_trace(go.Scatter(
                        x=anomalies_zs.index, y=anomalies_zs['Sales'],
                        mode='markers', name=f'Anomalies ({len(anomalies_zs)})',
                        marker=dict(color='#e67e22', size=10, line=dict(width=1, color='black'))
                    ))
                    fig.update_layout(
                        title='Anomaly Detection â€” Z-Score Method (|Z| > 2)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(family='Inter'),
                        yaxis_title='Weekly Sales ($)',
                        xaxis_title='Date'
                    )
                    st.plotly_chart(fig, width='stretch')
                    
                    st.markdown("### ðŸ“‹ Detected Anomalies (Z-Score)")
                    if len(anomalies_zs) > 0:
                        zs_table = anomalies_zs[['Sales']].copy()
                        if 'Z_Score' in anomalies_zs.columns:
                            zs_table['Z_Score'] = anomalies_zs['Z_Score'].round(2)
                        zs_table.index.name = 'Date'
                        zs_table = zs_table.reset_index()
                        zs_table['Date'] = pd.to_datetime(zs_table['Date']).dt.strftime('%Y-%m-%d')
                        st.dataframe(zs_table, width='stretch', hide_index=True)
        else:
            st.warning("âš ï¸ Anomaly data not found. Please run the analysis notebook first to generate model artifacts.")
            st.info("Run all cells in `analysis.ipynb` to generate the required data files in the `models/` directory.")

    # ==============================================================
    # PAGE 4: Product Demand Segments
    # ==============================================================
    elif page == "ðŸ“¦ Demand Segments":
        st.markdown('<h1 class="main-header">ðŸ“¦ Product Demand Segmentation</h1>', unsafe_allow_html=True)
        
        if 'cluster_features' in model_data:
            cf = model_data['cluster_features']
            
            # Cluster scatter plot
            if 'PCA_1' in cf.columns and 'PCA_2' in cf.columns:
                fig = px.scatter(
                    cf, x='PCA_1', y='PCA_2',
                    color='Cluster_Label',
                    text='Sub-Category',
                    size='Total_Sales',
                    title='Product Demand Segments (PCA Projection)',
                    color_discrete_sequence=['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'],
                    hover_data=['Total_Sales', 'Growth_Rate', 'Volatility']
                )
                fig.update_traces(
                    textposition='top center',
                    marker=dict(line=dict(width=1, color='black'))
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Inter'),
                    xaxis_title='Principal Component 1',
                    yaxis_title='Principal Component 2',
                    height=600
                )
                st.plotly_chart(fig, width='stretch')
            
            # Cluster membership table
            st.markdown("### ðŸ“‹ Sub-Category Cluster Assignments")
            
            if 'Cluster_Label' in cf.columns:
                display_cols = ['Sub-Category', 'Cluster_Label', 'Total_Sales', 'Growth_Rate', 'Volatility', 'Avg_Order_Value']
                available_cols = [c for c in display_cols if c in cf.columns]
                display_df = cf[available_cols].copy()
                display_df = display_df.sort_values('Cluster_Label')
                
                # Format numbers
                format_dict = {}
                if 'Total_Sales' in display_df.columns:
                    format_dict['Total_Sales'] = '${:,.0f}'
                if 'Growth_Rate' in display_df.columns:
                    format_dict['Growth_Rate'] = '{:.1f}%'
                if 'Volatility' in display_df.columns:
                    format_dict['Volatility'] = '${:,.0f}'
                if 'Avg_Order_Value' in display_df.columns:
                    format_dict['Avg_Order_Value'] = '${:,.2f}'
                
                st.dataframe(
                    display_df.style.format(format_dict),
                    width='stretch',
                    hide_index=True,
                    height=500
                )
            
            # Cluster summary metrics
            if 'Cluster_Label' in cf.columns:
                st.markdown("### ðŸ“Š Cluster Summary Statistics")
                
                numeric_cols_available = [c for c in ['Total_Sales', 'Growth_Rate', 'Volatility', 'Avg_Order_Value'] if c in cf.columns]
                summary = cf.groupby('Cluster_Label')[numeric_cols_available].mean().round(2)
                summary['Count'] = cf.groupby('Cluster_Label').size()
                st.dataframe(summary, width='stretch')
        
        else:
            st.warning("âš ï¸ Clustering data not found. Please run the analysis notebook first.")
            st.info("Run all cells in `analysis.ipynb` to generate the required data files in the `models/` directory.")

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Footer
    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #888; font-size: 0.8rem;'>"
        "Sales Forecasting & Demand Intelligence Dashboard | Built with Streamlit & Python</p>",
        unsafe_allow_html=True
    )
