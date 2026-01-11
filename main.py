"""
Market Immune System Dashboard
Real-time risk monitoring for market fragility and crash detection.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from market_immune_system import MarketImmuneSystem, SignalStatus


# Page configuration
st.set_page_config(
    page_title="Market Immune System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FAFAFA;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E1E2E 0%, #2D2D3F 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #333;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    .signal-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    .contributor-item {
        background: #1E1E2E;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        display: flex;
        justify-content: space-between;
        border-left: 3px solid #00C853;
    }
    .divider {
        border-top: 1px solid #333;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_market_data():
    """Load and cache market data for 1 hour."""
    mis = MarketImmuneSystem()
    returns = mis.fetch_data()
    return returns


from scipy import stats

def create_health_monitor_chart(
    dates: pd.DatetimeIndex,
    spy_cumulative: pd.Series,
    turbulence: pd.Series
) -> go.Figure:
    """Create dual-axis Market Health Monitor chart."""
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Calculate thresholds based on degrees of freedom
    df = 99 
    t75 = stats.chi2.ppf(0.75, df)
    t95 = stats.chi2.ppf(0.95, df)
    t99 = stats.chi2.ppf(0.99, df)
    
    # SPY Cumulative Returns (left axis)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=spy_cumulative,
            name="SPY Cumulative Return",
            line=dict(color="#00C853", width=2),
            fill=None
        ),
        secondary_y=False
    )
    
    # Turbulence Score (right axis) as area
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=turbulence,
            name="Turbulence (Mahalanobis)",
            line=dict(color="#FF5252", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255, 82, 82, 0.15)"
        ),
        secondary_y=True
    )
    
    # Add Threshold Lines
    fig.add_hline(y=t95, line_dash="dash", line_color="#FF9800", opacity=0.7, secondary_y=True, annotation_text="95th %", annotation_position="top right")
    fig.add_hline(y=t99, line_dash="dot", line_color="#FF5252", opacity=0.9, secondary_y=True, annotation_text="99th %", annotation_position="top right")
    
    # Layout updates
    fig.update_layout(
        title=dict(
            text="📊 Market Health Monitor",
            font=dict(size=20, color="#FAFAFA")
        ),
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=450,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    fig.update_xaxes(
        title_text="Date",
        gridcolor="#333",
        showgrid=True
    )
    
    fig.update_yaxes(
        title_text="SPY Cumulative Return (%)",
        secondary_y=False,
        gridcolor="#333",
        showgrid=True
    )
    
    fig.update_yaxes(
        title_text="Turbulence Score",
        secondary_y=True,
        showgrid=False
    )
    
    return fig


def create_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Create fragility correlation heatmap."""
    
    # Sample if too many assets for readability
    if len(corr_matrix.columns) > 30:
        # Select a diverse subset
        sample_tickers = list(corr_matrix.columns[:15]) + list(corr_matrix.columns[-15:])
        corr_matrix = corr_matrix.loc[sample_tickers, sample_tickers]
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale=[
            [0, "#1E88E5"],    # Negative correlation (blue)
            [0.5, "#0E1117"],  # Zero correlation (black)
            [1, "#FF5252"]     # Positive correlation (red)
        ],
        zmin=-1,
        zmax=1,
        colorbar=dict(
            title="Correlation",
            tickvals=[-1, -0.5, 0, 0.5, 1]
        )
    ))
    
    fig.update_layout(
        title=dict(
            text="🔥 30-Day Rolling Correlation Heatmap",
            font=dict(size=20, color="#FAFAFA")
        ),
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=500,
        margin=dict(l=100, r=60, t=80, b=100),
        xaxis=dict(tickangle=45)
    )
    
    return fig


def get_signal_color(signal: SignalStatus) -> str:
    """Get the color for a signal status."""
    return signal.value[1]


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ Market Immune System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time detection of market fragility and crash signals</p>', unsafe_allow_html=True)
    
    # Initialize the system
    mis = MarketImmuneSystem()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Controls")
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Date range info
        st.markdown("### 📅 Analysis Period")
        st.info(f"Effective lookback: {mis.effective_lookback}-day window")
        
        st.markdown("---")
    
    # Load data with spinner
    with st.spinner("📡 Fetching market data..."):
        try:
            returns = load_market_data()
        except Exception as e:
            st.error(f"❌ Failed to load market data: {str(e)}")
            st.info("Please check your internet connection and try again.")
            return
    
    # Calculate metrics
    with st.spinner("🧮 Calculating market metrics..."):
        try:
            metrics = mis.get_current_metrics(returns)
            turbulence_series = mis.calculate_rolling_turbulence(returns)
            spy_cumulative = mis.get_spy_cumulative_returns(returns)
            corr_matrix = mis.calculate_rolling_correlation(returns)
        except Exception as e:
            st.error(f"❌ Error calculating metrics: {str(e)}")
            return
    
    # Align series for charting
    common_dates = turbulence_series.index.intersection(spy_cumulative.index)
    turbulence_aligned = turbulence_series.loc[common_dates]
    spy_aligned = spy_cumulative.loc[common_dates]
    
    # Sidebar: Top Contributors
    with st.sidebar:
        st.markdown("### 📊 Top Turbulence Contributors")
        for ticker, contribution in metrics.top_contributors:
            st.markdown(f"""
            <div class="contributor-item">
                <span><strong>{ticker}</strong></span>
                <span style="color: #FF5252;">+{contribution:.1f}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Asset count
        n_assets = len(returns.columns)
        st.markdown(f"**Active Assets:** {n_assets}")
        st.markdown(f"**Data Points:** {len(returns):,}")
        st.markdown(f"**Last Updated:** {returns.index[-1].strftime('%Y-%m-%d')}")
    
    # Main Panel - Metrics Row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Determine threshold for color
        df_assets = len(returns.columns)
        t75 = stats.chi2.ppf(0.75, df_assets)
        t95 = stats.chi2.ppf(0.95, df_assets)
        
        color = "#00C853"
        if metrics.turbulence_score > t95:
            color = "#FF5252" # Red
        elif metrics.turbulence_score > t75:
            color = "#FF9800" # Orange
            
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Turbulence Score</div>
            <div class="metric-value" style="color: {color}">
                {metrics.turbulence_score:.1f}
            </div>
            <div style="color: #888; font-size: 0.85rem;">(Raw Distance)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Absorption Ratio</div>
            <div class="metric-value" style="color: {'#FF9800' if metrics.absorption_ratio > 800 else '#00C853'}">
                {metrics.absorption_ratio:.0f}
            </div>
            <div style="color: #888; font-size: 0.85rem;">/ 1000</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        signal_color = get_signal_color(metrics.signal)
        signal_name = metrics.signal.value[0]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Market Signal</div>
            <div class="signal-badge" style="background-color: {signal_color}; color: {'#FFF' if signal_color != '#212121' else '#FFF'};">
                {signal_name.upper()}
            </div>
            <div style="color: #AAA; font-size: 0.8rem; margin-top: 0.5rem;">
                SPY: {metrics.spy_return:+.2f}%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Signal message
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, {signal_color}22 0%, transparent 100%); 
                padding: 1rem; border-radius: 8px; margin: 1.5rem 0; 
                border-left: 4px solid {signal_color};">
        <strong style="color: {signal_color};">📢 {metrics.signal_message}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Charts
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Chart 1: Market Health Monitor
    health_chart = create_health_monitor_chart(
        common_dates, spy_aligned, turbulence_aligned
    )
    st.plotly_chart(health_chart, use_container_width=True)
    
    # Chart 2: Correlation Heatmap
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    heatmap = create_correlation_heatmap(corr_matrix)
    st.plotly_chart(heatmap, use_container_width=True)
    
    # Footer
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.85rem; padding: 1rem;">
        <strong>Market Immune System</strong> | Risk Dashboard for Fragility Detection<br>
        Data sourced from Yahoo Finance | Calculations use Ledoit-Wolf shrinkage for covariance estimation
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
