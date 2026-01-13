"""
Market Immune System Dashboard
Real-time risk monitoring for market fragility and crash detection.
"""

import warnings
warnings.filterwarnings('ignore')

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

def render_executive_summary(metrics: any, context: MarketContext, signal_color: str, turbulence_norm: float, analysis_date: str):
    """Render the text-based executive summary."""
    
    # Determine Status String
    status = "HEALTHY"
    if metrics.signal == SignalStatus.ORANGE:
        status = "ELEVATED"
    elif metrics.signal in [SignalStatus.RED, SignalStatus.BLACK]:
        status = "CRITICAL"
    elif metrics.signal == SignalStatus.BLUE:
        status = "OPPORTUNITY"
        
    # 2. [CRITICAL FIX] Absorption Override
    # If the market is locked up (Absorption > 850), it is NEVER "Healthy".
    if metrics.absorption_ratio > 850 and status in ["HEALTHY", "OPPORTUNITY"]:
        status = "FRAGILE"
        signal_color = "#FF9800" # Orange override
        
    # Interpretations
    interpretations = []
    
    # Turbulence Interpretation (Calibrated Scale: Warning=180, Critical=370)
    if turbulence_norm < 180:
         interpretations.append("✓ Market showing normal stress levels (< 180)")
    elif turbulence_norm > 370:
        interpretations.append(f"⚠️ **Critical turbulence ({turbulence_norm:.0f}/1000)**: Market behavior is extreme (> 99th Percentile).")
    else:
        interpretations.append(f"⚠️ Elevated turbulence ({turbulence_norm:.0f}/1000): Above Warning Level (180).")
    
    # Divergence Check (The "Low VIX" Trap)
    if metrics.signal == SignalStatus.RED:
        interpretations.append("⚠️ **DIVERGENCE DETECTED**: Market rising on broken structure (Price > 50MA + Turb > 180).")

    # Absorption Check
    if metrics.absorption_ratio > 850:
        interpretations.append("⚠️ **FRAGILITY CRITICAL**: Absorption Ratio > 850. Assets moving in lockstep. Diversification failing.")
    else:
        interpretations.append("✓ Absorption Rate is healthy (Diversification working)")
        
    if context.spx_level > context.spx_50d_ma:
        interpretations.append("✓ Price Trend: Bullish (Above 50-day MA)")
    else:
        interpretations.append("⚠️ Price Trend: Bearish (Below 50-day MA)")

    # Recommendations
    actions = []
    
    # Check for Fragility override in Advanced Signal
    is_fragile = "FRAGILE" in metrics.advanced_signal or "CAUTION" in metrics.advanced_signal
    
    if is_fragile:
        actions = [
            "**CAUTION: MELT-UP REGIME.** Prices rising on thin ice.",
            "Diversification is failing (Absorption Critical).",
            "Keep tight trailing stops.",
            "Do not add aggressive leverage."
        ]
    elif status == "HEALTHY":
        actions = ["Normal portfolio operations", "Monitor daily", "System functioning normally"]
    elif status == "ELEVATED":
        actions = ["Review leverage", "Monitor for persistence > 3 days", "Prepare hedges"]
    elif status == "CRITICAL":
        actions = ["Reduce risk exposure immediately", "Cash preservation", "Wait for turbulence to subside"]
    elif status == "OPPORTUNITY":
        actions = ["Look for high-quality entries", "Confirm with price action", "Scale in slowly"]

    # AI Context
    if context.spx_level > context.spx_50d_ma: # Bull Market
        ai_msg = "AI LAGGING VOLATILITY (Defensive)" if context.ai_market_ratio < 1.0 else "AI LEADING VOLATILITY (Aggressive)"
    else: # Bear Market
        ai_msg = "AI SHOWING RELATIVE STRENGTH" if context.ai_market_ratio < 1.0 else "AI LEADING THE DROP"

    # Important: No indentation in the HTML string to prevent Markdown code block rendering
    header_title = f"HISTORICAL IMMUNE SYSTEM STATUS - {analysis_date}" if analysis_date != datetime.now().strftime('%Y-%m-%d') else f"CURRENT IMMUNE SYSTEM STATUS - {analysis_date}"
    
    html_content = f"""
<div class="summary-card">
<div class="summary-header">📋 {header_title}</div>
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
    turbulence: pd.Series,
    spx_price: pd.Series,
    spx_ma: pd.Series
) -> go.Figure:
    """Create dual-axis Market Health Monitor chart with Divergence shading."""
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Thresholds for Calibrated 0-1000 Scale (P99 = 370)
    T_WARNING = 180.0
    T_CRITICAL = 370.0
    
    # 1. Turbulence Score (Left Axis) - Red Line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=turbulence,
            name="Turbulence (0-1000)",
            line=dict(color="#FF5252", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255, 82, 82, 0.1)"
        ),
        secondary_y=False
    )
    
    # 2. SPX Price (Right Axis) - Dark Blue Solid Line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=spx_price,
            name="S&P 500 Level",
            line=dict(color="#1565C0", width=2.5),
            fill=None
        ),
        secondary_y=True
    )
    
    # 3. SPX 50-Day MA (Right Axis) - Light Blue Dashed Line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=spx_ma,
            name="50-Day Moving Average",
            line=dict(color="#4FC3F7", width=1.5, dash="dash"),
            fill=None
        ),
        secondary_y=True
    )
    
    # 4. Green Vertical Shading for Divergence (Warning Signal)
    # Logic: Price Rising (Price > MA) AND High Turbulence (Turb > 180)
    divergence_mask = (turbulence > T_WARNING) & (spx_price > spx_ma)
    
    in_block = False
    start_date = None
    
    for date, is_divergent in divergence_mask.items():
        if is_divergent and not in_block:
            in_block = True
            start_date = date
        elif not is_divergent and in_block:
            in_block = False
            fig.add_vrect(
                x0=start_date, x1=date,
                fillcolor="rgba(0, 200, 83, 0.25)", # Green Shading
                layer="below", line_width=0,
                annotation_text="Divergence" if (date - start_date).days > 7 else None,
                annotation_position="top left",
                annotation_font_color="#00E676"
            )
            
    if in_block:
        fig.add_vrect(
            x0=start_date, x1=dates[-1],
            fillcolor="rgba(0, 200, 83, 0.25)",
            layer="below", line_width=0
        )

    # Threshold Lines (Left Axis)
    fig.add_hline(y=T_WARNING, line_dash="dash", line_color="#FF9800", opacity=0.8, secondary_y=False, annotation_text="Warning (18%)", annotation_position="top right")
    fig.add_hline(y=T_CRITICAL, line_dash="dot", line_color="#FF5252", opacity=0.9, secondary_y=False, annotation_text="Critical (37%)", annotation_position="bottom right")

    # Layout updates
    fig.update_layout(
        title=dict(
            text="📊 Market Health Monitor",
            font=dict(size=20, color="#FAFAFA")
        ),
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=60, t=80, b=60)
    )
    
    fig.update_xaxes(title_text="Date", gridcolor="#333", showgrid=True)
    
    # Left Y-Axis: Turbulence (0-1000)
    fig.update_yaxes(
        title_text="Turbulence Score (0-1000)",
        secondary_y=False,
        gridcolor="#333",
        showgrid=True,
        range=[0, 1000] # Fixed per PRD
    )
    
    # Right Y-Axis: SPX Price (Scaled to Price)
    fig.update_yaxes(
        title_text="S&P 500 Level",
        secondary_y=True,
        showgrid=False,
        range=[min(3500, spx_price.min() * 0.9), max(6000, spx_price.max() * 1.05)]
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

def render_catalyst_watch():
    """Display upcoming high-impact economic/corporate events in the sidebar."""
    # Manual high-impact list (Update quarterly)
    catalysts = {
        "2026-01-15": "CPI Inflation Data",
        "2026-01-28": "FOMC Rate Decision",
        "2026-02-06": "Non-Farm Payrolls",
        "2026-02-20": "NVDA Earnings (AI Proxy)"
    }
    
    st.markdown("### 📅 Macro Catalyst Watch")
    today = datetime.now().date()
    
    upcoming = []
    for date_str, event in catalysts.items():
        event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        delta = (event_date - today).days
        
        if 0 <= delta <= 10:
            upcoming.append((delta, event))
            
    if upcoming:
        for delta, event in upcoming:
            color = "#FF5252" if delta <= 3 else "#FF9800"
            st.markdown(
                f"<div style='border-left: 4px solid {color}; padding-left: 10px; margin-bottom: 12px;'>"
                f"<strong>{event}</strong><br>"
                f"<span style='font-size: 0.8em; color: #aaa;'>In {delta} days</span>"
                "</div>", 
                unsafe_allow_html=True
            )
    else:
        st.markdown("<div style='color: #666; font-size: 0.9rem;'>No high-impact events in next 10 days.</div>", unsafe_allow_html=True)

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
    
    # SYSTEM HORIZON BANNER
    st.markdown("""
    <div style="background-color: #2b3a42; border-left: 5px solid #FFD700; padding: 15px; border-radius: 5px; margin-bottom: 25px;">
        <strong style="color: #FFD700;">⚠️ SYSTEM HORIZON & LEAD TIME</strong><br>
        <span style="font-size: 0.9em; color: #e0e0e0;">
        This dashboard detects <strong>structural fragility</strong>. Historically, high Turbulence scores precede price capitulation by <strong>7 to 14 days</strong>. 
        A "Critical" signal means the market floor is brittle; it does not guarantee a drop today. 
        <strong>Trade the Price, but Respect the Structure.</strong>
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize the system
    mis = MarketImmuneSystem()
    
    # Load data immediately
    with st.spinner("📡 Fetching market data..."):
        try:
            # Unpack the tuple
            returns, prices, volumes = load_market_data()
            spx, spx_ma, vix = load_context_data()
        except Exception as e:
            st.error(f"❌ Failed to load market data: {str(e)}")
            return

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
        
        # Determine available date range
        min_date = returns.index.min().to_pydatetime()
        max_date = returns.index.max().to_pydatetime()
        
        # 1. Chart Filter & Analysis Date Anchor
        # The right handle of this slider now drives the "Analysis Date"
        selected_range = st.slider(
            "Filter Range (Right handle = Analysis Date)",
            min_value=min_date,
            max_value=max_date,
            value=(max_date - timedelta(days=180), max_date),
            format="YYYY-MM-DD"
        )
        
        # Determine valid calculation date (Anchor to the end of the range)
        target_ts = pd.Timestamp(selected_range[1])
        if target_ts not in returns.index:
             # Fallback to nearest valid date if slider picked a weekend/holiday
             target_ts = returns.index[returns.index <= target_ts][-1]
        
        st.info(f"Targeting: {target_ts.strftime('%Y-%m-%d')}")
        st.info(f"Effective lookback: {mis.effective_lookback}-day window")
        
        st.markdown("---")
        
        # 3. Catalyst Watch
        render_catalyst_watch()
    
    # Calculate metrics
    with st.spinner("🧮 Calculating market metrics..."):
        try:
            # Pass prices and volumes for advanced metrics
            # Calculate metrics for the SELECTED date (Slider End)
            metrics = mis.get_current_metrics(returns, prices, volumes, target_date=target_ts)
            
            # Generate rolling series for charts
            turbulence_raw_series = mis.calculate_rolling_turbulence(returns)
            
            # Calibrate Turbulence Series to 0-1000 Scale (P99 = 370)
            # 1. Get P99 of the raw series (up to target date to avoid future bias)
            p99_raw = turbulence_raw_series.loc[:target_ts].quantile(0.99)
            if p99_raw == 0: p99_raw = 1.0 # Avoid div by zero
            
            # 2. Scale: (Raw / P99) * 370
            turbulence_series = (turbulence_raw_series / p99_raw) * 370
            # 3. Cap at 1000
            turbulence_series = turbulence_series.clip(upper=1000)
            
            # Update Current Metric to match this calibrated scale
            current_raw = metrics.turbulence_score 
            turbulence_norm = min((current_raw / p99_raw) * 370, 1000.0)
            
            # SPX Price Series
            if "^GSPC" in prices.columns:
                spx_full = prices["^GSPC"]
            elif "SPY" in prices.columns:
                spx_full = prices["SPY"] * 10
            else:
                spx_full = pd.Series(0, index=returns.index)
                
            # Calculate 50-MA
            spx_ma_full = spx_full.rolling(window=50).mean()
            
            corr_matrix = mis.calculate_rolling_correlation(returns.loc[:target_ts])
            
            # Amihud Z-Score Series (Vectorized)
            illiq_series = mis.calculate_rolling_liquidity(returns, prices, volumes, "SPY")

            # Align all series to the valid turbulence window
            valid_idx = turbulence_series.index
            
            spx_aligned = spx_full.reindex(valid_idx)
            spx_ma_aligned = spx_ma_full.reindex(valid_idx)
            illiq_series = illiq_series.reindex(valid_idx)

            # Filter data based on slider
            mask = (valid_idx >= pd.Timestamp(selected_range[0])) & (valid_idx <= pd.Timestamp(selected_range[1]))
            
            turb_filtered = turbulence_series.loc[mask]
            spx_filtered = spx_aligned.loc[mask]
            spx_ma_filtered = spx_ma_aligned.loc[mask]
            illiq_filtered = illiq_series.loc[mask]
            
            # Context Calculations
            # Fetch context data for the TARGET date
            # We already have spx_full, spx_ma_full. We just need to slice them.
            current_spx = spx_full.loc[target_ts]
            current_ma = spx_ma_full.loc[target_ts]
            
            # VIX fallback (vix fetched from load_context_data is current real-time)
            # For historical accuracy, we should use prices['^VIX'] if available
            current_vix = vix # default
            if '^VIX' in prices.columns:
                 current_vix = prices.loc[target_ts, '^VIX']
            
            days_elevated = mis.calculate_days_elevated(turbulence_series.loc[:target_ts], 180.0)
            ai_turbulence = mis.calculate_sector_turbulence(returns.loc[:target_ts], "AI & Growth")
            
            # Create Context Object
            market_context = MarketContext(
                spx_level=current_spx,
                spx_50d_ma=current_ma,
                vix_level=current_vix,
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
        - **Healthy Condition**: Turbulence < 180 (Warning Threshold)
        - **Thresholds**: Healthy: 0-180 | Warning: 180-370 | Critical: >370 (99th Percentile)
        
        **2. Current Reality**
        - **Status**: <span style='color: {signal_color}; font-weight: bold;'>{report['current_state']}</span>
        - **Math Verification**: {report['verification_math']}
        
        **3. Recommended Action**
        - {report['recommended_action']}
        """, unsafe_allow_html=True)

    # Render Executive Summary
    analysis_date_str = target_ts.strftime('%Y-%m-%d')
    render_executive_summary(metrics, market_context, signal_color, turbulence_norm, analysis_date_str)

    # Turbulence Attribution (Why is it high?)
    drivers = mis.get_turbulence_drivers(returns.loc[:target_ts])
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
    macro_signals = mis.get_macro_signals(returns, target_date=target_ts)
    vix_term = mis.get_vix_term_structure_signal(prices.loc[:target_ts])
    
    # Futures & Sentiment (Real-Time Spot Check)
    # Only show if looking at recent data, otherwise history basis is hard to reconstruct without storage
    # We will show it but note it's current
    futures_data = mis.get_futures_sentiment()
    
    st.markdown("### 🔮 Futures & Sentiment Pricing (Current)")
    
    f1, f2, f3 = st.columns(3)
    
    with f1:
        if futures_data:
             st.metric(
                "S&P 500 Basis (Future vs Spot)",
                f"{futures_data['spx_basis']:.2f}%",
                futures_data['spx_signal'],
                delta_color="normal" if futures_data['spx_basis'] > -0.02 else "inverse"
             )
        else:
             st.metric("S&P 500 Basis", "N/A", "Data Unavailable")

    with f2:
        if futures_data:
             st.metric(
                "Bitcoin Basis (CME vs Spot)",
                f"{futures_data['btc_basis']:.2f}%",
                futures_data['btc_signal'],
                delta_color="normal" if futures_data['btc_basis'] > -0.5 else "inverse"
             )
        else:
             st.metric("Bitcoin Basis", "N/A", "Data Unavailable")
             
    with f3:
        st.metric(
            "VIX Term Structure",
            vix_term,
            help="Spot VIX / 3-Month VIX (VXV). If Ratio > 1.0, short-term fear is higher than long-term fear (Backwardation/Panic)."
        )

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
        hurst_delta = None
        if metrics.hurst_exponent < 0.4 and market_context.spx_level > market_context.spx_50d_ma:
            hurst_delta = "⚠️ Trend Conflict (Mean Reversion)"
            
        st.metric(
            "Hurst Exponent (Fractal)",
            f"{metrics.hurst_exponent:.2f}",
            delta=hurst_delta,
            delta_color="inverse",
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

    st.markdown("### 🔄 Capital Rotation (The 'Offense' Engine)")
    
    # Calculate Cycle
    cycle_data = mis.get_market_cycle_status(returns.loc[:target_ts])
    
    if cycle_data:
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.info(f"Detected Regime: **{cycle_data['current_phase']}**")
            st.caption("Based on 3-month Sector Relative Strength vs SPY.")
            
            # Show the leaderboard
            sorted_cycles = sorted(cycle_data['details'].items(), key=lambda x: x[1], reverse=True)
            for phase, score in sorted_cycles:
                color = "#00C853" if score > 0 else "#FF5252"
                st.markdown(f"**{phase}**: <span style='color:{color}'>{score:+.2f}%</span> vs SPY", unsafe_allow_html=True)

        with c2:
            # Quick Style Check logic
            # Growth (QQQ) vs Value (VTV - need to add to universe or use proxies)
            # Proxy: XLK (Tech) vs XLE (Energy) + XLF (Finance)
            tech_rel = 0
            if "XLK" in returns.columns and "XLE" in returns.columns:
                 # Simple 20-day trend of Ratio
                 ratio = prices["XLK"] / prices["XLE"]
                 trend = "GROWTH" if ratio.iloc[-1] > ratio.iloc[-20] else "VALUE"
                 
                 st.metric("Style Rotation", trend, "Tech vs Energy/Value")
            
            st.markdown("""
            **Playbook:**
            - **Early:** Buy Banks (XLF), Small Caps (IWM).
            - **Mid:** Buy Tech (XLK), Industrials (XLI).
            - **Late:** Buy Energy (XLE), Commodities (DBC).
            - **Recession:** Cash, Gold (GLD), Utilities (XLU).
            """)

    # Macro Context: Yield Curve & Dollar
    st.markdown("### 🏦 Macro Context (Yields & Dollar)")
    m1, m2 = st.columns(2)
    
    with m1:
        # Yield Curve (10Y - 2Y) or (10Y - 13W)
        # ^TNX = 10 Year Yield (Index, so price is yield * 10) -> No wait, yahoo returns yield as price directly usually?
        # Actually ^TNX is CBOE 10 Year, usually price 40.00 = 4.00% yield.
        # Let's assume we have prices for them.
        
        curve_msg = "N/A"
        curve_val = 0.0
        
        if "^TNX" in prices.columns:
            ten_y = prices["^TNX"].iloc[-1] # e.g. 42.50 = 4.25%
            
            # 2Y is often missing in yahoo free data (^IRX is 13 weeks)
            # Let's use 13 Week (^IRX) as proxy for "Cash"
            if "^IRX" in prices.columns:
                thirteen_w = prices["^IRX"].iloc[-1]
                curve_val = (ten_y - thirteen_w) / 10.0 # Convert to percentage points
                
                if curve_val < 0:
                    curve_msg = "WARNING: INVERTED (Recession Indicator)"
                    curve_col = "inverse"
                else:
                    curve_msg = "Normal (Positive Slope)"
                    curve_col = "normal"
                    
                st.metric("Yield Curve (10Y - 3M)", f"{curve_val:+.2f}%", curve_msg, delta_color=curve_col)
            else:
                st.metric("10 Year Yield", f"{ten_y/10:.2f}%", "Signal Missing")
        else:
             st.info("Yield Curve Data Unavailable")

    with m2:
        # Dollar (UUP)
        if "UUP" in prices.columns:
            uup_series = prices["UUP"]
            current_uup = uup_series.iloc[-1]
            uup_trend_val = current_uup - uup_series.iloc[-20]
            
            uup_trend = "RISING" if uup_trend_val > 0 else "FALLING"
            uup_impact = "Headwind for Crypto/Assets" if uup_trend == "RISING" else "Tailwind for Assets"
            
            st.metric("US Dollar Trend (UUP)", uup_trend, uup_impact, delta_color="off" if uup_trend=="RISING" else "normal")


    if macro_signals:
        st.markdown("### 🏦 Institutional Macro Ratios")
        st.caption("Strategic recommendations based on relative asset flows (20-day trend). Click ratios for definitions.")
        
        macro_cols = st.columns(2)
        for i, sig in enumerate(macro_signals):
            col_idx = i % 2
            with macro_cols[col_idx]:
                # Style the card
                bg_color = "#1E1E2E"
                border_color = "#00C853" if sig['trend'] == "Rising" else "#FF9800"
                
                # HTML with Link
                st.markdown(f"""
                <div style="background-color: {bg_color}; border-left: 4px solid {border_color}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <a href="{sig['url']}" target="_blank" style="text-decoration: none; color: #AAA; font-weight: bold; font-size: 0.85rem;">
                        {sig['pair']} ({sig['trend']}) 🔗
                    </a>
                    <div style="font-size: 0.8rem; color: #666; margin-bottom: 4px;">{sig['desc']}</div>
                    <span style="font-size: 1rem; font-weight: 500; color: #FAFAFA;">{sig['signal']}</span>
                </div>
                """, unsafe_allow_html=True)
    
    # Methodology & Research
    with st.expander("📚 Methodology, Lead Times & Research Sources"):
        st.markdown(r"""
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
        - **Definitions**:
            - [Absorption Ratio (Investopedia)](https://www.investopedia.com/terms/a/absorption-ratio.asp)
            - [Mahalanobis Distance (Wikipedia)](https://en.wikipedia.org/wiki/Mahalanobis_distance)
            - [Hurst Exponent (Wikipedia)](https://en.wikipedia.org/wiki/Hurst_exponent)
            - [Amihud Illiquidity (QuantPedia)](https://quantpedia.com/strategies/amihud-illiquidity-premium/)
        """)

    # Main Panel - Metrics Row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        color = "#00C853"
        if turbulence_norm > 370:
            color = "#FF5252" # Red
        elif turbulence_norm > 180:
            color = "#FF9800" # Orange
            
        st.metric(
            "Turbulence Score", 
            f"{turbulence_norm:.0f}/1000", 
            f"Raw: {metrics.turbulence_score:.1f}",
            delta_color="off",
            help="**Turbulence (0-1000)**: Calibrated Risk Index.\n\n"
                 "- **0-180 (Healthy)**: Normal noise.\n"
                 "- **180-370 (Warning)**: Elevated stress (95th-99th %).\n"
                 "- **370+ (Critical)**: Extreme structural break (99th % Anchor).\n\n"
                 "*Logic*: Scaled such that history's 99th percentile hits exactly 370."
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

    # Charts
    st.markdown("### 🔍 Visual Analysis")
    
    # Chart 1: Market Health Monitor
    st.markdown("**1. Market Health Monitor**")
    st.caption(
        "**Left Axis (Red Area):** Turbulence. **Right Axis (Blue Lines):** S&P 500 Price & 50-MA.\n"
        "**Blue Shading:** Divergence Zones (High Turbulence + Rising Market)."
    )
    health_chart = create_health_monitor_chart(
        turb_filtered.index, turb_filtered, spx_filtered, spx_ma_filtered
    )
    st.plotly_chart(health_chart, use_container_width=True)
    
    # Chart 2: Liquidity Stress
    st.markdown("**2. Liquidity Stress Gauge**")
    st.caption("Amihud Ratio Z-Score. Spikes (> 2.0σ) indicate thin liquidity and 'Liquidity Holes'.")
    
    fig_liq = go.Figure()
    fig_liq.add_trace(go.Scatter(
        x=illiq_filtered.index,
        y=illiq_filtered.values,
        name="Amihud Z-Score",
        line=dict(color="#2196F3", width=2),
        fill='tozeroy',
        fillcolor='rgba(33, 150, 243, 0.1)'
    ))
    # Red Zone Shading
    fig_liq.add_hrect(y0=2.0, y1=max(5.0, illiq_filtered.max() + 1), fillcolor="#FF5252", opacity=0.1, line_width=0)
    fig_liq.add_hline(y=2.0, line_dash="dash", line_color="#FF5252", annotation_text="Stress Threshold")
    
    fig_liq.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=350,
        margin=dict(l=60, r=60, t=30, b=60),
        yaxis_title="Z-Score"
    )
    st.plotly_chart(fig_liq, use_container_width=True)

    # Chart 3: Correlation Heatmap
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
