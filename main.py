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
from scipy import stats

from market_immune_system import MarketImmuneSystem, SignalStatus, MarketContext
import textwrap


# Page configuration
st.set_page_config(
    page_title="Market Immune System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern dark theme
st.markdown(textwrap.dedent("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00C853 !important;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #AAA;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1E1E2E;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        text-align: center;
        border: 1px solid #333;
    }
    .metric-value {
        font-size: 2rem;
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
    .summary-card {
        background-color: #FDFBF0; /* Beige/Paper feel from screenshot */
        color: #212121;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 5px solid #333;
        font-family: 'Courier New', monospace;
        margin-bottom: 2rem;
    }
    .summary-header {
        font-weight: 700;
        border-bottom: 2px solid #555;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""").strip(), unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_market_data():
    """Load and cache market data for 1 hour."""
    mis = MarketImmuneSystem()
    returns = mis.fetch_data()
    return returns

@st.cache_data(ttl=3600)
def load_context_data():
    """Load auxiliary context data (VIX, SPX)."""
    mis = MarketImmuneSystem()
    return mis.fetch_market_context_data()

def render_executive_summary(metrics: any, context: MarketContext, signal_color: str, turbulence_norm: float):
    """Render the text-based executive summary."""
    
    # Determine Status String
    status = "HEALTHY"
    if metrics.signal == SignalStatus.ORANGE:
        status = "ELEVATED"
    elif metrics.signal in [SignalStatus.RED, SignalStatus.BLACK]:
        status = "CRITICAL"
    elif metrics.signal == SignalStatus.BLUE:
        status = "OPPORTUNITY"
        
    # Interpretations
    interpretations = []
    
    # Turbulence Interpretation (0-1000)
    if turbulence_norm < 750:
         interpretations.append("✓ Market showing normal stress levels")
    elif turbulence_norm > 950:
        interpretations.append(f"⚠️ **Critical turbulence ({turbulence_norm:.0f}/1000)**: Market behavior is statistically extreme (Top 5%).")
    else:
        interpretations.append(f"⚠️ Elevated turbulence ({turbulence_norm:.0f}/1000)")
    
    # Divergence Check (The "Low VIX" Trap)
    if metrics.signal == SignalStatus.RED:
        interpretations.append("⚠️ **DIVERGENCE DETECTED**: VIX is Low (Calm) but Turbulence is High (Fragile). This mismatch often precedes crash events.")

    if metrics.absorption_ratio > 800:
        interpretations.append("⚠️ High asset correlation (fragile structure - contagion risk)")
    else:
        interpretations.append("✓ Absorption Rate is healthy (Diversification working)")
        
    if context.spx_level > context.spx_50d_ma:
        interpretations.append("✓ Risk-on environment (Price > 50d MA)")
    else:
        interpretations.append("⚠️ Risk-off potential (Price < 50d MA)")

    # Recommendations
    actions = []
    if status == "HEALTHY":
        actions = ["Normal portfolio operations", "Monitor daily but no immediate action needed", "Next check: Tomorrow's update"]
    elif status == "ELEVATED":
        actions = ["Review leverage and tight stops", "Monitor for persistence > 3 days", "Prepare hedging strategy"]
    elif status == "CRITICAL":
        actions = ["Reduce risk exposure immediately", "Hedge downside risk", "Wait for turbulence to subside"]
    elif status == "OPPORTUNITY":
        actions = ["Look for high-quality entries", "Confirm with price action", "Scale in slowly"]

    # AI Context
    ai_msg = "AI SECTOR SHOWING RELATIVE STRENGTH" if context.ai_market_ratio < 1.0 else "AI SECTOR LEADING STRESS"

    # Important: No indentation in the HTML string to prevent Markdown code block rendering
    html_content = f"""
<div class="summary-card">
<div class="summary-header">📋 CURRENT IMMUNE SYSTEM STATUS - {datetime.now().strftime('%Y-%m-%d')}</div>
<div style="margin-bottom: 1rem;">
<strong>WARNING LEVEL:</strong> <span style="background-color: {signal_color}; color: white; padding: 2px 8px; border-radius: 4px;">{status}</span>
</div>
<div style="margin-bottom: 1rem;">
<strong>📉 Current Metrics:</strong>
<ul style="margin-top: 5px;">
<li>Market Turbulence: {turbulence_norm:.0f}/1000 <span style="color:#888; font-size:0.8em;">(Raw: {metrics.turbulence_score:.1f})</span></li>
<li>Days Elevated: {context.days_elevated} days</li>
<li>SPX Level: {context.spx_level:.2f} ({'ABOVE' if context.spx_level > context.spx_50d_ma else 'BELOW'} 50-day MA)</li>
<li>VIX Level: {context.vix_level:.2f}</li>
<li>Divergence Active: {'Yes' if metrics.signal == SignalStatus.RED else 'No'}</li>
</ul>
</div>
<div style="margin-bottom: 1rem;">
<strong>🧠 Interpretation:</strong>
<ul style="margin-top: 5px;">
{''.join([f'<li>{i}</li>' for i in interpretations])}
</ul>
</div>
<div style="margin-bottom: 1rem;">
<strong>🛡️ RECOMMENDED ACTIONS:</strong>
<ul style="margin-top: 5px;">
{''.join([f'<li>{a}</li>' for a in actions])}
</ul>
</div>
<div>
<strong>🤖 AI SECTOR CONTEXT:</strong>
<ul style="margin-top: 5px;">
<li>AI Turbulence: {context.ai_turbulence:.1f} (Raw)</li>
<li>Market Score: {turbulence_norm:.0f}/1000</li>
<li>Ratio: {context.ai_market_ratio:.2f}x</li>
<br>
<li>ℹ️ {ai_msg}</li>
</ul>
</div>
</div>
"""
    st.markdown(html_content, unsafe_allow_html=True)


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

@st.cache_data(ttl=3600)
def load_market_data():
    """Load and cache market data for 1 hour."""
    mis = MarketImmuneSystem()
    return mis.fetch_data() # Returns (returns, prices, volumes)

@st.cache_data(ttl=3600)
def load_context_data():
    """Load auxiliary context data (VIX, SPX)."""
    mis = MarketImmuneSystem()
    return mis.fetch_market_context_data()

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
    
    # Load data
    with st.spinner("📡 Fetching market data..."):
        try:
            # Unpack the tuple
            returns, prices, volumes = load_market_data()
            spx, spx_ma, vix = load_context_data()
        except Exception as e:
            st.error(f"❌ Failed to load market data: {str(e)}")
            return
    
    # Calculate metrics
    with st.spinner("🧮 Calculating market metrics..."):
        try:
            # Pass prices and volumes for advanced metrics
            metrics = mis.get_current_metrics(returns, prices, volumes)
            turbulence_series = mis.calculate_rolling_turbulence(returns)
            spy_cumulative = mis.get_spy_cumulative_returns(returns)
            corr_matrix = mis.calculate_rolling_correlation(returns)
            
            # Context Calculations
            df_assets = len(returns.columns)
            
            # Normalize Turbulence to 0-1000 Scale (CDF)
            turbulence_raw = metrics.turbulence_score
            turbulence_norm = stats.chi2.cdf(turbulence_raw, df_assets) * 1000
            
            t75 = stats.chi2.ppf(0.75, df_assets)
            
            days_elevated = mis.calculate_days_elevated(turbulence_series, t75)
            ai_turbulence = mis.calculate_sector_turbulence(returns, "AI & Growth")
            
            # Create Context Object
            market_context = MarketContext(
                spx_level=spx,
                spx_50d_ma=spx_ma,
                vix_level=vix,
                days_elevated=days_elevated,
                ai_turbulence=ai_turbulence,
                ai_market_ratio=ai_turbulence / (metrics.turbulence_score + 1e-6)
            )
            
        except Exception as e:
            st.error(f"❌ Error calculating metrics: {str(e)}")
            st.exception(e)
            return

    # Detail Status Verification
    signal_color = get_signal_color(metrics.signal)
    report = mis.get_detailed_report(metrics)
    with st.expander(f"🛡️ Verify Status: Why is it {report['current_state']}?"):
        st.markdown(f"""
        **1. Definitions**
        - **Healthy Condition**: {report['definition_of_healthy']}
        - **Thresholds**: {report['thresholds']}
        
        **2. Current Reality**
        - **Status**: <span style='color: {signal_color}; font-weight: bold;'>{report['current_state']}</span>
        - **Math Verification**: {report['verification_math']}
        
        **3. Recommended Action**
        - {report['recommended_action']}
        """, unsafe_allow_html=True)

    # Render Executive Summary
    render_executive_summary(metrics, market_context, signal_color, turbulence_norm)

    # Turbulence Attribution (Why is it high?)
    drivers = mis.get_turbulence_drivers(returns)
    with st.expander("🔍 Why is Turbulence High? (Top Contributors)"):
        st.markdown(
            "These assets showed the most extreme moves relative to their own history (Z-Score), "
            "driving the aggregate turbulence score higher."
        )
        
        driver_cols = st.columns(5)
        for i, d in enumerate(drivers):
            with driver_cols[i]:
                st.metric(
                    d['ticker'],
                    f"{d['return']:+.2f}%",
                    f"{d['z_score']:+.1f}σ"
                )

    # Institutional Macro Analysis
    macro_signals = mis.get_macro_signals(returns)
    vix_term = mis.get_vix_term_structure_signal(prices)
    
    st.markdown("### ⚡ Advanced Quant Signals")
    
    # New Quant Row
    q1, q2, q3 = st.columns(3)
    with q1:
        st.metric(
            "Math Super-Signal",
            metrics.advanced_signal,
            help="Integrated Logic:\n\n"
                 "- **FRAGILE**: Hurst > 0.75 (Crowded) + Liquidity Z > 2.0 (Thin).\n"
                 "- **CRASH**: Turbulence > 900.\n"
                 "- **BUY**: Turbulence Fading + Liquidity Restored + Mean Reversion (Hurst < 0.5)."
        )
    with q2:
        st.metric(
            "Hurst Exponent (Fractal)",
            f"{metrics.hurst_exponent:.2f}",
            help="**The Hurst Exponent ($H$)**: A robust variance ratio test.\n\n"
                 "- $H \\approx 0.5$: Random Walk (Healthy).\n"
                 "- $H > 0.75$: Strong Trend (Greed/FOMO). Prone to reversal.\n"
                 "- $H \\to 1.0$: Supercritical State. Slightest shock triggers cascade."
        )
    with q3:
        st.metric(
            "Liquidity Stress (Alihud Z)",
            f"{metrics.liquidity_z:.1f}σ",
            delta="Normal" if metrics.liquidity_z < 1.0 else "Thin",
            delta_color="inverse",
            help="**Amihud Illiquidity ($ILLIQ$)**: $|Return_t| / (Volume_t \\times Price_t)$.\n\n"
                 "When this spikes (Z-Score > 2), it takes less money to crash the market. The 'Floor' is gone."
        )

    if macro_signals:
        st.markdown("### 🏦 Institutional Macro Ratios")
        st.caption("Strategic recommendations based on relative asset flows (20-day trend).")
        
        macro_cols = st.columns(2)
        for i, sig in enumerate(macro_signals):
            col_idx = i % 2
            with macro_cols[col_idx]:
                # Style the card
                bg_color = "#1E1E2E"
                border_color = "#00C853" if sig['trend'] == "Rising" else "#FF9800"
                
                st.markdown(f"""
                <div style="background-color: {bg_color}; border-left: 4px solid {border_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <strong style="color: #AAA; font-size: 0.8rem;">{sig['pair']} ({sig['trend']})</strong><br>
                    <span style="font-size: 1rem; font-weight: 500;">{sig['signal']}</span>
                </div>
                """, unsafe_allow_html=True)
    
    # Methodology & Research
    with st.expander("📚 Methodology, Lead Times & Research Sources"):
        st.markdown("""
        ### 1. How is the Warning Level Calculated?
        The **"Critical"** warning is not arbitrary. It is derived from the **Mahalanobis Distance**, a statistical measure of how "strange" the current return vector is compared to the historical covariance matrix.
        - **Formula**: $D^2 = (r - \mu)^T \Sigma^{-1} (r - \mu)$
        - **Normalization**: The raw $D^2$ score is mapped to a Chi-Squared cumulative distribution function (CDF) with degrees of freedom equal to the number of assets (~99).
        - **Thresholds**: 
            - **95th Percentile**: "Elevated" (Occurs once a month)
            - **99th Percentile**: "Critical" (Occurs few times a year)

        ### 2. Lead Time Limitations & Reality
        **Turbulence is a Lead Indicator, but not a Crystal Ball.**
        - **Lead Time**: Historically, structural breaks (correlation breakdowns) occur **1 to 3 weeks** before significant price capitulation.
        - **Why?**: Institutional liquidity dries up first ( widening spreads, increased turbulence) before retailers panic-sell.
        - **Action**: A "Critical" signal means **fragility is high**. A small catalyst can now cause a large crash.

        ### 3. Recommended Actions Logic
        The dashboard infers actions by combining the **Turbulence** (Shock) with the **Macro Ratios**:
        - If **EEM/SPY** is Rising → "Capital is rotating to Emerging Markets (Risk On / US Peak)".
        - If **SPY/TLT** is Falling → "Capital is fleeing to Bonds (Risk Off)".
        
        ### 4. Sources & Datasets
        - **Core Paper**: Kritzman, M., & Li, Y. (2010). ["Skulls, Financial Turbulence, and Risk Management"](https://www.tandfonline.com/doi/abs/10.2469/faj.v66.n5.3). *Financial Analysts Journal*.
        - **Data Source**: Real-time OHLCV data from **Yahoo Finance** (`yfinance`), covering 99 global assets.
        - **Other Useful Datasets (Not currently integrated)**:
            - **FRED API**: For 10y-2y Yield Curve (Recession signal).
            - **Options Flow (CBOE)**: For Put/Call ratios (Sentiment).
        """)

    # Main Panel - Metrics Row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        color = "#00C853"
        if turbulence_norm > 950:
            color = "#FF5252" # Red
        elif turbulence_norm > 750:
            color = "#FF9800" # Orange
            
        st.metric(
            "Turbulence Score", 
            f"{turbulence_norm:.0f}/1000", 
            f"Raw: {metrics.turbulence_score:.1f}",
            delta_color="off",
            help="**Turbulence (0-1000)**: Measures the statistical 'bounciness' or shock level of the market structure.\n\n"
                 "- **0-750 (Healthy)**: Normal noise.\n"
                 "- **750-950 (Elevated)**: Unusual volatility clustering.\n"
                 "- **950+ (Critical)**: Extreme structural stress (2+ Sigma event).\n\n"
                 "*Logic*: Based on Mahalanobis Distance (Raw Score available in delta)."
        )
    
    with col2:
        st.metric(
            "Absorption Ratio",
            f"{metrics.absorption_ratio:.0f}/1000",
            help="Measures market unification (using PCA). High values (>800) mean assets are moving in lockstep, indicating fragility and risk of contagion.",
            delta_color="off"
        )
    
    with col3:
        st.metric(
            "Market Signal",
            metrics.signal.value[0].upper(),
            f"SPY {metrics.spy_return:+.2f}%",
            delta_color="normal",
            help="Combined signal based on Turbulence, Absorption, and Price Price action. Green=Normal, Orange=Elevated, Red=Divergence, Black=Crash."
        )

    # Align series for charting
    common_dates = turbulence_series.index.intersection(spy_cumulative.index)
    turbulence_aligned = turbulence_series.loc[common_dates]
    spy_aligned = spy_cumulative.loc[common_dates]

    # Charts
    st.markdown("### 🔍 Visual Analysis")
    
    # Chart 1: Market Health Monitor
    st.markdown("**1. Market Health Monitor**")
    st.caption(
        "Red spikes (Turbulence) represent structural stress. "
        "Historically, these spikes often **precede** price drops by days or weeks (Lead Indicator)."
    )
    health_chart = create_health_monitor_chart(
        common_dates, spy_aligned, turbulence_aligned
    )
    st.plotly_chart(health_chart, use_container_width=True)
    
    # Chart 2: Correlation Heatmap
    st.markdown("**2. Fragility Heatmap**")
    st.caption("Rolling 30-day correlation matrix. Red = Assets moving together (Danger). Blue = Diversified (Safe).")
    with st.expander("ℹ️ How to read this heatmap"):
        st.markdown("""
        - **Axes**: List of 30 representative assets from the portfolio.
        - **Color**:
            - 🟥 **Red (+1.0)**: Perfect positive correlation. Assets move up/down together. High red density = Fragile.
            - 🟦 **Blue (-1.0)**: Negative correlation. Assets move in opposite directions (hedging).
            - ⬛ **Black (0.0)**: Uncorrelated.
        - **Goal**: You want to see a mix of colors. A 'sea of red' indicates panic/liquidation events.
        """)
        
    heatmap = create_correlation_heatmap(corr_matrix)
    st.plotly_chart(heatmap, use_container_width=True)
    
    # Footer
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.85rem; padding: 1rem;">
        <strong>Market Immune System</strong> | Risk Dashboard<br>
        <br>
        <strong>Data Sources:</strong> Finance data sourced via <code>yfinance</code> (Yahoo Finance API).<br>
        <strong>Assets:</strong> Tracks 99 liquid assets including SPY, QQQ, GLD, VIX, Crypto (BTC/ETH), and AI Sector Leaders (NVDA, AMD).<br>
        <strong>Methodology:</strong> Statistical Turbulence (Kritzman & Li) + Absorption Ratio (Kritzman et al.).<br>
        <em>Data is delayed by 15-20 minutes. Not investment advice.</em>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
