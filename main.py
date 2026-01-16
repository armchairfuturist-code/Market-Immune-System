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
import yfinance as yf

# Import MacroConnector for FRED, calendars, sentiment
try:
    from macro_connector import get_macro_connector
    MACRO_AVAILABLE = True
except ImportError:
    MACRO_AVAILABLE = False


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
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        color: #00C853 !important;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
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

def render_tactical_hud(metrics: any, context: MarketContext, cycle_data: dict, analysis_date: str, crypto_zscore: float = 0.0, spy_flat: bool = False):
    """
    Renders the 'Head-Up Display' (HUD) - Action-Oriented Version.
    
    Args:
        metrics: MarketMetrics object
        context: MarketContext object
        cycle_data: Dict from get_market_cycle_status
        analysis_date: String date for display
        crypto_zscore: Average Z-Score of crypto assets (for early stress detection)
        spy_flat: True if SPY return is between -0.5% and +0.5%
    """
    # 1. Theme Logic
    theme_color = "#00C853" # Green
    if metrics.signal == SignalStatus.ORANGE: theme_color = "#FF9800"
    elif metrics.signal in [SignalStatus.RED, SignalStatus.BLACK]: theme_color = "#FF5252"
    elif metrics.signal == SignalStatus.BLUE: theme_color = "#2196F3"

    # 2. Novice-Friendly Narrative (The "What is happening?")
    regime_title = "NORMAL MARKET"
    regime_desc = "Conditions are safe. Standard investing applies."
    
    # Additional analysis lines for special conditions
    additional_analysis = ""
    
    # CRYPTO-LED STRESS DETECTION
    if crypto_zscore > 2.0 and spy_flat:
        additional_analysis = "<br><span style='color: #FF9800;'>⚠️ Crypto-Led Stress detected (Pre-cursor to broad volatility)</span>"
    
    if metrics.absorption_ratio > 850:
        if context.spx_level > context.spx_50d_ma:
            regime_title = "FRAGILE RALLY (MELT-UP)"
            regime_desc = "Prices are going up, but the market is moving in lockstep. <strong>A sudden drop is likely within 2 weeks.</strong>"
        else:
            regime_title = "SYSTEMIC SELL-OFF"
            regime_desc = "Everything is falling together. Cash is the only safe haven."
    elif metrics.signal == SignalStatus.RED:
        regime_title = "TRAP DETECTED (DIVERGENCE)"
        regime_desc = "The market looks good on the surface, but internal structure is breaking. <strong>Do not chase rallies.</strong>"
    elif metrics.signal == SignalStatus.BLACK:
        regime_title = "CRASH ALERT"
        regime_desc = "Extreme volatility. Protect capital immediately."
        
    # NEW LOGIC: The Synthesis Shortlist
    shortlist = []
    
    # 1. The Risk Trigger
    if metrics.turbulence_score > 370:
        shortlist.append("🔴 **SELL:** High System Stress. Raise 20% Cash.")
    elif metrics.absorption_ratio > 850:
        shortlist.append("🟠 **CAUTION:** Market is 'Locked'. Tighten Stops.")
    else:
        shortlist.append("🟢 **HOLD:** System structure is stable.")

    # 2. The Opportunity Trigger
    if cycle_data:
        # Split tickers string "Tech, Industrials" -> "Tech"
        top_sector = cycle_data['actionable_tickers'].split(',')[0] if cycle_data.get('actionable_tickers') else "Index"
        shortlist.append(f"🚀 **BUY:** {cycle_data['current_phase']} leaders ({top_sector})")
        
    # Format Shortlist for HTML
    shortlist_html = "".join([f"<div style='margin-bottom: 6px;'>{item}</div>" for item in shortlist])

    # 3. Actionable Playbook (The "What do I buy?")
    buy_recommendation = "Diversified Index (SPY)"
    narrative_reason = "No clear trend."
    
    if cycle_data:
        # Use the plain English list we added to market_immune_system.py
        buy_recommendation = f"<strong>{cycle_data['actionable_tickers']}</strong>"
        narrative_reason = cycle_data['narrative']

    # 4. Plain English Tactical Stance
    stop_loss = "Use Tight Stops (Protect Profits)" if metrics.signal in [SignalStatus.ORANGE, SignalStatus.RED] else "Standard Stops"
    leverage = "Reduce Leverage (High Risk)" if metrics.turbulence_score > 370 else "Leverage OK"
    hedging = "Consider Cash/Gold" if metrics.absorption_ratio > 850 else "Stay Invested"
    
    # --- RENDER THE HUD ---
    hud_html = f"""<div style="
background-color: #1E1E2E; 
border-left: 6px solid {theme_color}; 
padding: 20px; 
border-radius: 8px; 
margin-bottom: 25px;
box-shadow: 0 4px 6px rgba(0,0,0,0.2);
font-family: sans-serif;">
<div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #333; padding-bottom: 15px; margin-bottom: 15px;">
<div>
<span style="font-size: 0.85rem; color: #888; text-transform: uppercase; letter-spacing: 1px;">MARKET STATUS ({analysis_date})</span>
<h2 style="margin: 0; color: {theme_color}; font-size: 1.8rem; font-weight: 700;">{regime_title}</h2>
</div>
<div style="text-align: right;">
<span style="background-color: #333; color: #fff; padding: 6px 14px; border-radius: 4px; font-size: 0.85rem; border: 1px solid #444;">
⏱️ Next Signal: <strong>7-14 Days</strong>
</span>
</div>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
<!-- COLUMN 1: STRATEGIC SHORTLIST -->
<div>
<strong style="color: #DDD; font-size: 0.95rem;">📋 Strategic Shortlist</strong>
<div style="color: #AAA; font-size: 0.9rem; margin-top: 5px; line-height: 1.4; background: rgba(255,255,255,0.05); padding: 10px; border-radius: 4px;">
{shortlist_html}
</div>
<div style="font-size: 0.8rem; color: #666; margin-top: 8px;">
Risk Score: <span style="color: #DDD;">{metrics.turbulence_score:.0f}</span>/1000
</div>
</div>
<!-- COLUMN 2: TACTICAL STANCE -->
<div style="border-left: 1px solid #333; padding-left: 20px;">
<strong style="color: #DDD; font-size: 0.95rem;">🛡️ Safety Checks</strong>
<ul style="color: #AAA; font-size: 0.9rem; margin-top: 5px; padding-left: 20px; line-height: 1.6;">
<li>{stop_loss}</li>
<li>{leverage}</li>
<li>{hedging}</li>
</ul>
</div>
<!-- COLUMN 3: THE BUY SIGNAL -->
<div style="border-left: 1px solid #333; padding-left: 20px;">
<strong style="color: #DDD; font-size: 0.95rem;">🚀 What to Buy Now?</strong>
<p style="color: #AAA; font-size: 0.9rem; margin-top: 5px;">{narrative_reason}</p>
<div style="font-size: 0.9rem; color: {theme_color}; margin-top: 5px; background-color: rgba(255,255,255,0.05); padding: 10px; border-radius: 4px;">
{buy_recommendation}
</div>
</div>
</div>
</div>"""
    st.markdown(hud_html, unsafe_allow_html=True)


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

@st.cache_data(ttl=24*3600) # Cache for 24 hours
def get_earnings_calendar(tickers: list) -> dict:
    """Fetch next earnings dates for key tickers."""
    calendar = {}
    today = datetime.now().date()
    
    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            # Get calendar
            cal = ticker.calendar
            if cal is not None and not cal.empty:
                # Calendar is usually Key (0) -> Date
                # Or a DataFrame. yfinance structure varies by version.
                # Assuming 'Earnings Date' or similar implies the date.
                # Common yf structure: cal is Dict or DF.
                
                # Check for 'Earnings Date' lists
                dates = []
                if isinstance(cal, dict):
                    if 'Earnings Date' in cal:
                        dates = cal['Earnings Date']
                    elif 'Earnings High' in cal: # Sometimes structure is diff
                         pass
                elif isinstance(cal, pd.DataFrame):
                    # Transposed usually?
                    if 'Earnings Date' in cal.index:
                        dates = cal.loc['Earnings Date'].tolist()
                    else:
                        # Try to parse numeric dates if any
                        pass
                
                # Fallback: detection
                # If we fail to parse, skip
                # Let's try a safer 'next_event' approach if available or just catch errors
                pass

            # Alternative: Ticker.incomestmt usually has dates? No.
            # Let's try the simplest approach: Fast info?
            # Basic yfinance often returns next earnings in .info['earningsTimestamp']?
            # Creating a robust fallback.
            
            # NEW APPROACH: .info['earningsTimestamp']
            info = ticker.info
            if 'earningsTimestamp' in info:
                ts = info['earningsTimestamp']
                if ts:
                    dt = datetime.fromtimestamp(ts).date()
                    if dt >= today:
                        calendar[t] = dt
                        
        except Exception:
            continue
            
    return calendar

def render_catalyst_watch():
    """Display upcoming high-impact corporate/economic events."""
    
    # 1. Major Watchlist
    watchlist = ["NVDA", "SPY", "MSFT", "TSLA", "AAPL", "AMD", "COIN"]
    
    # 2. Fetch Earnings (Cached)
    # Note: SPY/Indices don't have standard "earnings", so we filter
    corp_tickers = [t for t in watchlist if t not in ["SPY", "QQQ"]]
    earnings_map = get_earnings_calendar(corp_tickers)
    
    # 3. Macro Events (Still partially manual due to lack of good free macro API)
    # We will keep a small manual list for CPI/FOMC as yfinance doesn't provide this clean macro data.
    # But user asked to "Delete the static dictionary immediately". 
    # USER REQUEST: "Replace this widget with a 'Next Earnings Watch'". 
    # So we focus PURELY on the Earnings Watch per instructions.
    
    st.markdown("### 📅 Next Earnings Watch")
    
    if not earnings_map:
        st.info("No immediate earnings detected for watchlist.")
        return

    # Sort by date
    sorted_events = sorted(earnings_map.items(), key=lambda x: x[1])
    today = datetime.now().date()
    
    shown_count = 0
    
    for ticker, event_date in sorted_events:
        delta = (event_date - today).days
        
        # Only show if within 30 days
        if 0 <= delta <= 45:
            shown_count += 1
            color = "#FF5252" if delta <= 3 else "#00C853"
            
            st.markdown(
                f"<div style='border-left: 4px solid {color}; padding-left: 10px; margin-bottom: 12px;'>"
                f"<strong>{ticker}</strong> Earnings<br>"
                f"<span style='font-size: 0.8em; color: #aaa;'>{event_date.strftime('%b %d')} (in {delta} days)</span>"
                "</div>", 
                unsafe_allow_html=True
            )
            
    if shown_count == 0:
         st.markdown("<div style='color: #666; font-size: 0.9rem;'>No major earnings in next 45 days.</div>", unsafe_allow_html=True)

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

@st.cache_data(ttl=86400)  # 24 hours - FRED data is slow-moving
def load_fred_data():
    """Load FRED macro data (yield curve, credit spreads)."""
    if not MACRO_AVAILABLE:
        return None, None
    try:
        macro = get_macro_connector()
        yield_curve = macro.get_real_yield_curve()
        credit_stress = macro.get_credit_stress_index()
        return yield_curve, credit_stress
    except Exception as e:
        print(f"FRED load error: {e}")
        return None, {}

@st.cache_data(ttl=3600)  # 1 hour for sentiment
def load_sentiment_data(ticker: str = "SPY"):
    """Load sentiment analysis for a ticker."""
    if not MACRO_AVAILABLE:
        return {"score": 50, "label": "N/A", "headlines_analyzed": 0}
    try:
        macro = get_macro_connector()
        return macro.get_sentiment_score(ticker)
    except Exception as e:
        print(f"Sentiment load error: {e}")
        return {"score": 50, "label": "N/A", "headlines_analyzed": 0}

def get_market_status():
    """Get current market open/closed status."""
    if not MACRO_AVAILABLE:
        return {"is_open": False, "message": "Calendar unavailable"}
    try:
        macro = get_macro_connector()
        return macro.get_market_calendar_status()
    except:
        return {"is_open": False, "message": "Calendar error"}

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ Market Immune System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-time detection of market fragility and crash signals</p>', unsafe_allow_html=True)
    


    
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
        
        # Market Status Badge
        market_status = get_market_status()
        status_color = "#00C853" if market_status.get("is_open") else "#FF9800"
        status_text = "🟢 OPEN" if market_status.get("is_open") else "🔴 CLOSED"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1E1E2E, #252535); padding: 10px 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid {status_color};">
            <span style="font-size: 0.8em; color: #888;">NYSE STATUS</span><br>
            <span style="font-size: 1.1em; font-weight: 600; color: {status_color};">{status_text}</span>
            <span style="font-size: 0.75em; color: #666; margin-left: 8px;">{market_status.get('message', '')}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Refresh button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Date range info
        st.markdown("### 📅 Analysis Period")
        
        # Determine available date range
        min_date = returns.index.min().to_pydatetime()
        # Ensure max_date is the actual last day of data, not a future date
        max_data_date = returns.index.max().to_pydatetime()
        
        selected_range = st.slider(
            "Filter Range (Right handle = Analysis Date)",
            min_value=min_date,
            max_value=max_data_date, # LOCK THIS TO DATA
            value=(max_data_date - timedelta(days=180), max_data_date),
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

    # Calculate Cycle Data EARLY for the HUD
    cycle_data = mis.get_market_cycle_status(returns.loc[:target_ts])
    
    # Calculate Crypto Z-Score for early stress detection
    crypto_zscore, crypto_high_z_tickers = mis.calculate_crypto_zscore(returns.loc[:target_ts])
    
    # Check if SPY is "flat" (between -0.5% and +0.5%)
    spy_flat = abs(metrics.spy_return) < 0.5
    
    # Render The New Tactical HUD with Crypto-Led Stress detection
    analysis_date_str = target_ts.strftime('%Y-%m-%d')
    render_tactical_hud(metrics, market_context, cycle_data, analysis_date_str, crypto_zscore, spy_flat)

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
    
    st.markdown("### 🔮 Market Sentiment (What Pros are betting on)")
    
    f1, f2, f3 = st.columns(3)
    
    with f1:
        if futures_data:
             # Translate Basis to Sentiment
             sent_label = "Bullish (Normal)"
             if "Backwardation" in futures_data['spx_signal']:
                 sent_label = "Bearish (Panic)"
             
             st.metric(
                "S&P 500 Sentiment",
                f"{futures_data['spx_basis']:.2f}% Premium",
                sent_label,
                delta_color="normal" if futures_data['spx_basis'] > -0.02 else "inverse",
                help="If Positive: Pros are paying extra to buy. If Negative: Pros are paying extra to sell (Panic)."
             )
        else:
             st.metric("S&P 500 Sentiment", "N/A", "Data Unavailable")

    with f2:
        if futures_data:
             st.metric(
                "Bitcoin Sentiment",
                f"{futures_data['btc_basis']:.2f}% Premium",
                futures_data['btc_signal'],
                delta_color="normal" if futures_data['btc_basis'] > -0.5 else "inverse"
             )
        else:
             st.metric("Bitcoin Sentiment", "N/A", "Data Unavailable")
             
    with f3:
        # Simplify VIX
        vix_clean = vix_term.split(" ")[0] # Just "Contango" or "Backwardation"
        vix_human = "Normal Fear" if "Contango" in vix_clean else "Extreme Panic"
        
        st.metric(
            "Volatility Outlook",
            vix_human,
            help="Normal Fear = Safe to invest. Extreme Panic = Crash imminent or happening."
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
            "Trend Quality (Hurst)",
            f"{metrics.hurst_exponent:.2f}",
            delta=hurst_delta,
            delta_color="inverse",
            help="**Trend Quality**: Is the market trending smoothly or becoming fragile?\n\n"
                 "- Score > 0.75: Crowded Trade (Risk of Reversal).\n"
                 "- Score ~ 0.50: Normal Random Movement."
        )
    with q3:
        st.metric(
            "Market Liquidity",
            f"{metrics.liquidity_z:.1f}σ",
            delta="Normal" if metrics.liquidity_z < 1.0 else "Thin",
            delta_color="inverse",
            help="**Liquidity**: How easily can you sell?\n\n"
                 "High Z-Score (>2.0) means liquidity is evaporating. Small sells cause big crashes."
        )

    st.markdown("### 🔄 Capital Rotation (The 'Offense' Engine)")
    
    # Cycle data is already calculated above for HUD
    
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
            # Growth (VUG) vs Value (VTV)
            # Proxy Fallback: XLK (Tech) vs XLE (Energy)
            
            style_trend = "NEUTRAL"
            style_label = "VUG vs VTV"
            
            if "VUG" in returns.columns and "VTV" in returns.columns:
                ratio = prices["VUG"] / prices["VTV"]
                style_trend = "GROWTH" if ratio.iloc[-1] > ratio.iloc[-20] else "VALUE"
                style_label = "VUG (Growth) vs VTV (Value)"
            elif "XLK" in returns.columns and "XLE" in returns.columns:
                 # Fallback Proxy
                 ratio = prices["XLK"] / prices["XLE"]
                 style_trend = "GROWTH" if ratio.iloc[-1] > ratio.iloc[-20] else "VALUE"
                 style_label = "Tech (XLK) vs Energy (XLE)"
                 
            st.metric("Style Rotation", style_trend, style_label)
            
            st.markdown("""
            **Playbook:**
            - **Early:** Buy Banks (XLF), Small Caps (IWM).
            - **Mid:** Buy Tech (XLK), Industrials (XLI).
            - **Late:** Buy Energy (XLE), Commodities (DBC).
            - **Recession:** Cash, Gold (GLD), Utilities (XLU).
            """)

    # Battle of the Narratives Widget
    st.markdown("### ⚔️ Battle of the Narratives")
    st.caption("Where is speculative liquidity flowing? (5-Day Performance)")
    
    narrative_battle = mis.get_narrative_battle(returns.loc[:target_ts])
    
    if narrative_battle and narrative_battle.get("narrative") != "Insufficient data":
        n1, n2, n3 = st.columns([1, 1, 1])
        
        with n1:
            ai_color = "#00C853" if narrative_battle["ai_perf"] > 0 else "#FF5252"
            st.markdown(f"""
            <div style="background: #1E1E2E; border-radius: 8px; padding: 15px; text-align: center; border: 1px solid #333;">
                <div style="font-size: 0.8rem; color: #888; text-transform: uppercase;">AI Sector</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: {ai_color};">{narrative_battle["ai_perf"]:+.2f}%</div>
                <div style="font-size: 0.75rem; color: #666;">NVDA, AMD, SMCI, PLTR</div>
            </div>
            """, unsafe_allow_html=True)
        
        with n2:
            # Leader Badge
            leader = narrative_battle["leader"]
            if leader == "AI":
                badge_color = "#2196F3"
                badge_icon = "🤖"
            elif leader == "Crypto":
                badge_color = "#FF9800"
                badge_icon = "₿"
            else:
                badge_color = "#666"
                badge_icon = "⚖️"
                
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1E1E2E, #252535); border-radius: 8px; padding: 15px; text-align: center; border: 2px solid {badge_color};">
                <div style="font-size: 2rem;">{badge_icon}</div>
                <div style="font-size: 1rem; font-weight: 700; color: {badge_color};">{leader} Leading</div>
                <div style="font-size: 0.75rem; color: #888; margin-top: 5px;">{narrative_battle["narrative"]}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with n3:
            crypto_color = "#00C853" if narrative_battle["crypto_perf"] > 0 else "#FF5252"
            st.markdown(f"""
            <div style="background: #1E1E2E; border-radius: 8px; padding: 15px; text-align: center; border: 1px solid #333;">
                <div style="font-size: 0.8rem; color: #888; text-transform: uppercase;">Crypto Sector</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: {crypto_color};">{narrative_battle["crypto_perf"]:+.2f}%</div>
                <div style="font-size: 0.75rem; color: #666;">BTC, ETH, SOL, AVAX</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Narrative battle data unavailable. Check if crypto tickers are loading.")

    # Macro Context: FRED Official Data + Dollar
    st.markdown("### 🏦 Official Macro Data (Fed Sources)")
    
    # Load FRED data
    fred_yield, fred_credit = load_fred_data()
    sentiment_data = load_sentiment_data("SPY")
    
    m1, m2, m3 = st.columns(3)
    
    with m1:
        # FRED Yield Curve (T10Y2Y) - Now returns a dict with de-inversion detection
        if fred_yield and fred_yield.get("value") is not None:
            curve_val = fred_yield["value"]
            curve_date = fred_yield["date"]
            is_deinverting = fred_yield.get("is_deinverting", False)
            is_inverted = fred_yield.get("is_inverted", False)
            
            # Determine display based on de-inversion (most critical) or inversion
            if is_deinverting:
                curve_msg = "🚨 DE-INVERTING"
                curve_col = "inverse"
            elif is_inverted:
                curve_msg = "⚠️ INVERTED (Recession)"
                curve_col = "inverse"
            else:
                curve_msg = "Normal (Positive)"
                curve_col = "normal"
                
            st.metric(
                "Yield Curve (10Y-2Y)", 
                f"{curve_val:+.2f}%", 
                curve_msg, 
                delta_color=curve_col,
                help="**Source**: FRED Series T10Y2Y\n\nThe official 10-Year minus 2-Year Treasury spread.\n\n"
                     "**⚠️ DE-INVERSION WARNING**: Paradoxically, the crash usually happens when the curve "
                     "UN-INVERTS (goes from negative to positive), not when it first tips negative. "
                     "This is because de-inversion signals the Fed is cutting rates in response to recession arriving."
            )
            
            # Show de-inversion alert prominently
            if is_deinverting:
                st.error("🚨 **De-Inversion Alert**: Curve un-inverting. Historical crash precursor (6-12 month warning).")
            
            st.caption(f"📅 As of {curve_date}")
        else:
            # Fallback to proxy
            if "^TNX" in prices.columns and "^IRX" in prices.columns:
                ten_y = prices["^TNX"].iloc[-1]
                thirteen_w = prices["^IRX"].iloc[-1]
                curve_val = (ten_y - thirteen_w) / 10.0
                curve_msg = "⚠️ INVERTED" if curve_val < 0 else "Normal"
                st.metric("Yield Curve (Proxy)", f"{curve_val:+.2f}%", curve_msg)
                st.caption("⚠️ Using Yahoo proxy")
            else:
                st.info("Yield Curve: FRED unavailable")

    with m2:
        # Credit Stress (BAMLH0A0HYM2)
        if fred_credit and fred_credit.get("value") is not None:
            spread_val = fred_credit["value"]
            z_score = fred_credit["z_score"]
            spread_date = fred_credit["date"]
            signal = fred_credit["signal"]
            
            # Color based on Z-score
            if z_score > 2.0:
                delta_color = "inverse"
            elif z_score > 1.0:
                delta_color = "off"
            else:
                delta_color = "normal"
                
            st.metric(
                "Credit Stress (HY Spread)", 
                f"{spread_val:.2f}%",
                f"Z: {z_score:+.1f}σ",
                delta_color=delta_color,
                help="**Source**: FRED Series BAMLH0A0HYM2\n\nICE BofA High Yield Spread. When this spikes (Z > 2.0), credit is freezing - an immediate crash warning."
            )
            if "CRITICAL" in signal:
                st.error(signal)
            elif "ELEVATED" in signal:
                st.warning(signal)
            st.caption(f"📅 As of {spread_date}")
        else:
            st.info("Credit Stress: FRED unavailable")
    
    with m3:
        # Sentiment Score (VADER)
        score = sentiment_data.get("score", 50)
        label = sentiment_data.get("label", "N/A")
        headlines = sentiment_data.get("headlines_analyzed", 0)
        
        # Color based on extremes
        if score > 70:
            sent_color = "#FF9800"  # Euphoria = Caution
        elif score < 30:
            sent_color = "#FF5252"  # Panic = Danger
        else:
            sent_color = "#00C853"  # Neutral = OK
            
        st.metric(
            "News Sentiment (SPY)",
            f"{score:.0f}/100",
            label,
            help="**Source**: FinViz headlines + VADER sentiment\n\nScale: 0=Extreme Fear, 50=Neutral, 100=Euphoria.\n\n**Top Signal**: Euphoria (>80) + High Turbulence = Potential Peak."
        )
        
        # Check for euphoria divergence
        if score > 80 and metrics.turbulence_score > 180:
            st.error("🚨 EUPHORIA PEAK: High sentiment + stress = potential top")
        elif score > 70 and metrics.turbulence_score > 250:
            st.warning("⚠️ Sentiment diverging from structure")
            
        st.caption(f"📰 {headlines} headlines analyzed")

    # Dollar Trend (Keep existing)
    st.markdown("---")
    d1, d2 = st.columns(2)
    
    with d1:
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
    
    # Toggle for secondary axis (S&P 500 vs Bitcoin)
    chart_col1, chart_col2 = st.columns([1, 4])
    with chart_col1:
        chart_asset = st.radio("Secondary Axis:", ["S&P 500", "Bitcoin"], horizontal=False, label_visibility="collapsed")
    
    with chart_col2:
        if chart_asset == "Bitcoin":
            st.caption(
                "**Left Axis (Red Area):** Turbulence. **Right Axis (Orange Line):** Bitcoin Price.\n"
                "**Green Shading:** Divergence Zones (High Turbulence + Rising Market)."
            )
        else:
            st.caption(
                "**Left Axis (Red Area):** Turbulence. **Right Axis (Blue Lines):** S&P 500 Price & 50-MA.\n"
                "**Green Shading:** Divergence Zones (High Turbulence + Rising Market)."
            )
    
    # Determine which price series to use
    if chart_asset == "Bitcoin" and "BTC-USD" in prices.columns:
        secondary_price = prices["BTC-USD"].reindex(turb_filtered.index).ffill()
        secondary_ma = secondary_price.rolling(window=50).mean()
        secondary_label = "Bitcoin (BTC)"
        secondary_color = "#FF9800"  # Orange for BTC
    else:
        secondary_price = spx_filtered
        secondary_ma = spx_ma_filtered
        secondary_label = "S&P 500 Level"
        secondary_color = "#1565C0"  # Blue for SPX
    
    health_chart = create_health_monitor_chart(
        turb_filtered.index, turb_filtered, secondary_price, secondary_ma
    )
    
    # Update chart labels if using Bitcoin
    if chart_asset == "Bitcoin" and "BTC-USD" in prices.columns:
        health_chart.update_traces(line=dict(color=secondary_color), selector=dict(name="S&P 500 Level"))
        health_chart.update_traces(name="Bitcoin (BTC)", selector=dict(name="S&P 500 Level"))
        health_chart.update_yaxes(title_text="Bitcoin Price (USD)", secondary_y=True)
    
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
