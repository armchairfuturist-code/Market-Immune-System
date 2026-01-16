"""
Market Immune System Dashboard
Advanced risk monitoring and tactical asset allocation.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import textwrap
from datetime import datetime, timedelta
import yfinance as yf
from market_immune_system import MarketImmuneSystem, SignalStatus, MarketContext

# TRY IMPORT MACRO
try:
    from macro_connector import get_macro_connector
    MACRO_AVAILABLE = True
except ImportError:
    MACRO_AVAILABLE = False

# Page configuration
st.set_page_config(page_title="Market Immune System", layout="wide", page_icon="🛡️")

# Custom CSS for Premium Dark Theme
st.markdown(textwrap.dedent("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400&display=swap');
    
    :root {
        --bg-dark: #0E1117;
        --card-bg: #1E1E2E;
        --accent-green: #00C853;
        --accent-red: #FF5252;
        --accent-orange: #FF9800;
        --accent-blue: #2196F3;
        --text-main: #FAFAFA;
        --text-dim: #888;
    }

    .main { background-color: var(--bg-dark); }
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, var(--accent-green), #B2FF59);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    .metric-container {
        background: var(--card-bg);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #333;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }

    .stMetric {
        background: rgba(255,255,255,0.03);
        padding: 15px;
        border-radius: 8px;
    }

    /* Adjust sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161922;
        border-right: 1px solid #333;
    }
</style>
"""), unsafe_allow_html=True)

# --- CACHE DATA ---

@st.cache_data(ttl=3600)
def load_data():
    mis = MarketImmuneSystem()
    return mis.fetch_data() # Returns (log_returns, prices, volumes)

@st.cache_data(ttl=24*3600)
def get_earnings_watch():
    """Fetch upcoming earnings for high-beta leaders."""
    tickers = ["NVDA", "MSFT", "TSLA", "AAPL", "AMD", "COIN", "PLTR", "SMCI"]
    calendar = []
    today = datetime.now().date()
    
    for t in tickers:
        try:
            tick = yf.Ticker(t)
            cal = tick.calendar
            nxt = None
            
            if isinstance(cal, dict) and 'Earnings Date' in cal:
                nxt = cal['Earnings Date'][0]
            elif isinstance(cal, pd.DataFrame) and not cal.empty:
                # Transposed check
                if 'Earnings Date' in cal.index:
                    nxt = cal.loc['Earnings Date'].iloc[0]
                else:
                    nxt = cal.iloc[0, 0]
            
            if not nxt:
                # Fallback to info timestamp
                ts = tick.info.get('earningsTimestamp')
                if ts: nxt = datetime.fromtimestamp(ts)

            if nxt:
                dt = nxt.date() if hasattr(nxt, 'date') else nxt
                days = (dt - today).days
                if 0 <= days <= 45:
                    calendar.append({"ticker": t, "days": days, "date": dt})
        except:
            continue
    return sorted(calendar, key=lambda x: x['days'])

# --- UI COMPONENTS ---

def render_tactical_hud(metrics, context, cycle_data, analysis_date, crypto_zscore=0.0, spy_flat=False):
    """
    Renders the actionable Head-Up Display.
    """
    theme_color = "#00C853"
    if metrics.signal == SignalStatus.ORANGE: theme_color = "#FF9800"
    elif metrics.signal in [SignalStatus.RED, SignalStatus.BLACK]: theme_color = "#FF5252"
    elif metrics.signal == SignalStatus.BLUE: theme_color = "#2196F3"

    regime_title = "NORMAL MARKET"
    regime_desc = "Structure is stable. Normal investing applies."
    
    # Logic Overrides
    if metrics.absorption_ratio > 850:
        regime_title = "FRAGILE RALLY" if context.spx_level > context.spx_50d_ma else "SYSTEMIC SELL-OFF"
        regime_desc = "High market unification. Risk of sharp correction is elevated."
    elif metrics.signal == SignalStatus.RED:
        regime_title = "DIVERGENCE TRAP"
        regime_desc = "Price masking internal structural break. Do not chase."
    
    # Strategic Shortlist
    shortlist = []
    if metrics.turbulence_score > 370:
        shortlist.append("🔴 **SELL:** High System Stress. Raise Cash.")
    elif metrics.absorption_ratio > 850:
        shortlist.append("🟠 **CAUTION:** Market Locked. Tighten Stops.")
    else:
        shortlist.append("🟢 **HOLD:** System structure is healthy.")

    if cycle_data:
        top_sector = cycle_data['actionable_tickers'].split(',')[0]
        shortlist.append(f"🚀 **BUY:** {cycle_data['current_phase']} ({top_sector})")

    if crypto_zscore > 2.0 and spy_flat:
        shortlist.append("⚠️ **STRESS:** Crypto-led volatility detected.")

    shortlist_html = "".join([f"<div style='margin-bottom: 8px;'>{item}</div>" for item in shortlist])

    # HTML Render
    hud_html = f"""
<div style="background: #1E1E2E; border-left: 6px solid {theme_color}; padding: 25px; border-radius: 12px; margin-bottom: 30px; border-top: 1px solid #333; border-right: 1px solid #333; border-bottom: 1px solid #333;">
    <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 20px;">
        <div>
            <div style="font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: 2px;">MARKET REGIME • {analysis_date}</div>
            <h2 style="margin: 5px 0 0 0; color: {theme_color}; font-size: 2.2rem; font-weight: 800;">{regime_title}</h2>
        </div>
        <div style="background: rgba(255,255,255,0.05); padding: 8px 15px; border-radius: 6px; border: 1px solid #444; font-size: 0.9rem;">
            🛡️ <strong>Signal Confirmed</strong>
        </div>
    </div>
    <div style="display: grid; grid-template-columns: 1.2fr 1fr 1fr; gap: 30px;">
        <div>
            <div style="color: #DDD; font-size: 1rem; font-weight: 600; margin-bottom: 12px;">🔎 Performance Synthesis</div>
            <p style="color: #AAA; font-size: 0.9rem; line-height: 1.5; margin-bottom: 15px;">{regime_desc}</p>
            <div style="background: rgba(0,0,0,0.2); padding: 12px; border-radius: 6px;">{shortlist_html}</div>
        </div>
        <div style="border-left: 1px solid #333; padding-left: 25px;">
            <div style="color: #DDD; font-size: 1rem; font-weight: 600; margin-bottom: 12px;">🛡️ Tactical Stance</div>
            <ul style="color: #AAA; font-size: 0.9rem; line-height: 2; padding-left: 20px;">
                <li>Stops: <strong>{'Tight' if metrics.signal == SignalStatus.ORANGE else 'Standard'}</strong></li>
                <li>Leverage: <strong>{'None' if metrics.turbulence_score > 300 else 'OK'}</strong></li>
                <li>Hedging: <strong>{'Required' if metrics.absorption_ratio > 850 else 'None'}</strong></li>
            </ul>
        </div>
        <div style="border-left: 1px solid #333; padding-left: 25px;">
            <div style="color: #DDD; font-size: 1rem; font-weight: 600; margin-bottom: 12px;">🚀 Opportunity Engine</div>
            <div style="color: #AAA; font-size: 0.85rem; margin-bottom: 5px;">Primary Playbook:</div>
            <div style="font-size: 1.1rem; color: {theme_color}; background: rgba(0,200,83,0.05); padding: 12px; border-radius: 6px; border: 1px solid rgba(0,200,83,0.2);">
                {cycle_data.get('actionable_tickers', 'Diversified Index')}
            </div>
            <div style="font-size: 0.8rem; color: #666; margin-top: 10px;">{cycle_data.get('narrative', '')}</div>
        </div>
    </div>
</div>
"""
    st.markdown(hud_html, unsafe_allow_html=True)

def create_health_monitor_chart(dates, turbulence, prices, ma):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=dates, y=turbulence, name="Turbulence", line=dict(color="#FF5252", width=1.5), fill="tozeroy", fillcolor="rgba(255, 82, 82, 0.1)"), secondary_y=False)
    fig.add_trace(go.Scatter(x=dates, y=prices, name="Price", line=dict(color="#2196F3", width=2.5)), secondary_y=True)
    fig.add_trace(go.Scatter(x=dates, y=ma, name="50-MA", line=dict(color="#4FC3F7", width=1, dash="dash")), secondary_y=True)
    
    # Divergence Shading
    div_mask = (turbulence > 180) & (prices > ma)
    for i in range(1, len(div_mask)):
        if div_mask.iloc[i]:
            fig.add_vrect(x0=dates[i-1], x1=dates[i], fillcolor="#00C853", opacity=0.1, layer="below", line_width=0)

    fig.update_layout(template="plotly_dark", paper_bgcolor="#0E1117", plot_bgcolor="#0E1117", height=450, margin=dict(t=30, b=30), showlegend=False)
    fig.update_yaxes(title_text="Turbulence", secondary_y=False, range=[0, 1000])
    return fig

def create_correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """Create fragility correlation heatmap with Blue/Black/Red scale."""
    if corr_matrix.empty: return go.Figure()
    
    # Sample if too many assets for readability
    if len(corr_matrix.columns) > 30:
        # Select a diverse subset (Top 15 + Bottom 15 by volatility or just index)
        # Using simple slicing for performance
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
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title=dict(text="🔥 Systemic Fragility Map (Red = Lockstep)", font=dict(color="#FAFAFA")),
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis=dict(tickangle=45)
    )
    return fig

# --- MAIN APP ---

def main():
    st.markdown('<h1 class="main-header">🛡️ Market Immune System</h1>', unsafe_allow_html=True)
    
    mis = MarketImmuneSystem()
    
    with st.spinner("📡 Syncing Core Engine..."):
        returns, prices, volumes = load_data()
        
    if returns.empty or len(returns) < 10:
        st.error("❌ Failed to initialize market data. Please check connection and refreshing.")
        st.stop()
        
    # SLIDER FIX: Avoid Time Travel
    min_date = returns.index.min().to_pydatetime()
    max_date = returns.index.max().to_pydatetime()
    
    with st.sidebar:
        st.markdown("### ⚙️ SYSTEM CONTROLS")
        if st.button("🔄 REFRESH CACHE", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        rng = st.slider("ANALYSIS ANCHOR", min_date, max_date, (max_date-timedelta(365), max_date), format="YYYY-MM-DD")
        target_date = pd.Timestamp(rng[1])
        if target_date not in returns.index:
            target_date = returns.index[returns.index <= target_date][-1]

        # Earnings Watch Sidebar Widget
        st.markdown("---")
        st.markdown("### 📅 EARNINGS WATCH")
        calendar = get_earnings_watch()
        if calendar:
            for item in calendar:
                col = "#FF5252" if item['days'] < 3 else "#00C853"
                st.markdown(f"""
                <div style="border-left: 3px solid {col}; padding-left: 10px; margin-bottom: 10px;">
                    <div style="font-size: 0.9rem; font-weight: 700;">{item['ticker']}</div>
                    <div style="font-size: 0.75rem; color: #888;">{item['date'].strftime('%b %d')} ({item['days']}d)</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No major earnings (45d)")

    # --- CALCULATIONS ---
    sub_ret = returns.loc[:target_date]
    sub_prc = prices.loc[:target_date]
    sub_vol = volumes.loc[:target_date]
    
    metrics = mis.get_current_metrics(sub_ret, sub_prc, sub_vol, target_date=target_date)
    cycle_data = mis.get_market_cycle_status(sub_ret)
    narrative = mis.get_narrative_battle(sub_ret)
    crypto_z, _ = mis.calculate_crypto_zscore(sub_ret)
    
    # Macro Data
    fred_yield, fred_credit, sentiment = {}, {}, {}
    if MACRO_AVAILABLE:
        mc = get_macro_connector()
        fred_yield = mc.get_real_yield_curve()
        fred_credit = mc.get_credit_stress_index()
        sentiment = mc.get_sentiment_score("SPY")

    # --- UI RENDERING ---
    
    # 1. TACTICAL HUD
    spx_p = sub_prc["SPY"].iloc[-1] * 10 if "SPY" in sub_prc else sub_prc.iloc[-1, 0]
    spx_ma = (sub_prc["SPY"].rolling(50).mean().iloc[-1] * 10) if "SPY" in sub_prc else spx_p
    
    context = MarketContext(spx_level=spx_p, spx_50d_ma=spx_ma, vix_level=20.0, days_elevated=0, ai_turbulence=0.0, ai_market_ratio=0.0)
    render_tactical_hud(metrics, context, cycle_data, target_date.strftime('%Y-%m-%d'), crypto_z, abs(metrics.spy_return) < 0.5)

    # 2. METRICS ROW
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("System Turbulence", f"{metrics.turbulence_score:.0f}/1000", help="Calibrated Mahalanobis Distance.")
    with c2: st.metric("Absorption Ratio", f"{metrics.absorption_ratio:.0f}/1000", help="Market unification (fragility).")
    with c3:
        leader = narrative.get('leader', 'Neutral')
        delta = narrative.get('ai_perf', 0) - narrative.get('crypto_perf', 0)
        st.metric("Narrative Alpha", leader, f"{delta:+.1f}% Delta", help="5-day performance gap: AI vs Crypto.")

    # 3. MACRO ROW
    st.markdown("### 🏦 Institutional Truth")
    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Yield Curve", f"{fred_yield.get('value', 0):+.2f}%", fred_yield.get('signal', 'N/A'))
    with m2: st.metric("Credit Stress", f"{fred_credit.get('z_score', 0):.1f}σ", fred_credit.get('signal', 'NORMAL'))
    with m3: st.metric("News Sentiment", f"{sentiment.get('score', 50):.0f}", sentiment.get('label', 'Neutral'))

    # 4. CHARTS
    st.markdown("### 🔍 Historical Structure")
    
    # Prepare chart data
    hist_rets = returns.loc[rng[0]:rng[1]]
    hist_prcs = prices.loc[rng[0]:rng[1]]
    
    # Dynamic Turbulence Series for Chart
    turb_series = mis.calculate_rolling_turbulence(returns).loc[rng[0]:rng[1]]
    # Calibrate to 1000 scale
    p99 = turb_series.quantile(0.99) if not turb_series.empty else 1.0
    turb_series = (turb_series / p99 * 370).clip(upper=1000)
    
    chart_p = hist_prcs["SPY"].copy() * 10 if "SPY" in hist_prcs else hist_prcs.iloc[:, 0]
    chart_ma = chart_p.rolling(50).mean()
    
    st.plotly_chart(create_health_monitor_chart(turb_series.index, turb_series, chart_p, chart_ma), use_container_width=True)

    # 5. HEATMAP
    st.markdown("### 🔥 Asset Correlation Map")
    with st.expander("ℹ️ How to read this map"):
        st.markdown("**Red (+1.0)** = Assets moving in lockstep (Fragile). **Blue (-1.0)** = Hedging behavior. **Black (0)** = Uncorrelated.")
    
    # Calculate rolling correlation for the heatmap (last 30 days of the selected view)
    heatmap_data = hist_rets.tail(30)
    if not heatmap_data.empty:
        corr_matrix = heatmap_data.corr()
        st.plotly_chart(create_correlation_heatmap(corr_matrix), use_container_width=True)

    # Footer
    st.markdown("---")
    st.caption("🛡️ Market Immune System | Institutional Risk Engine | Data delayed 15m")

if __name__ == "__main__":
    main()
