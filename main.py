import streamlit as st
import pandas as pd
import datetime
import config
from core import data_loader, math_engine, macro_connector, cycle_engine, report_generator, cycle_playbook
from ui import charts
import yfinance as yf

# 1. Page Config
st.set_page_config(
    page_title="Market Immune System v3.0",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CSS Injection
def load_css():
    with open("ui/theme.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# 3. Sidebar
st.sidebar.title("🛡️ Market Immune System")
st.sidebar.caption("System Horizon: 7-14 Days")
st.sidebar.markdown("---")

# Time Travel Cap
today = datetime.datetime.now().date()
analysis_date = st.sidebar.date_input("Analysis Date", today, max_value=today)

# Data Start Date (Replaces Lookback)
default_start = today - datetime.timedelta(days=730)
start_date = st.sidebar.date_input("Data Start Date", default_start, max_value=today)

# Catalyst Watch (Earnings & Macro)
st.sidebar.markdown("### 📅 Catalyst Watch")

if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()

st.sidebar.markdown("---")
st.sidebar.info("v3.0 | Zero-Trust Engine")

# 4. Data Loading
@st.cache_data(ttl=3600)
def get_market_data(start_date):
    # Fetch with buffer for calculations (Turbulence needs 365d history)
    if isinstance(start_date, datetime.datetime):
        start_date = start_date.date()
    buffer_date = start_date - datetime.timedelta(days=365)
    
    df_close, df_vol = data_loader.fetch_market_data(config.ASSET_UNIVERSE, start_date=buffer_date)
    # Earnings (Fast changing)
    earnings = data_loader.fetch_next_earnings(config.GROWTH_ASSETS, limit=10)
    # Futures Trend
    futures = data_loader.fetch_futures_data(period="3mo")
    
    return df_close, df_vol, earnings, futures

@st.cache_data(ttl=86400) # 24h Cache for Macro
def get_macro_data():
    macro = macro_connector.MacroConnector()
    yield_curve = macro.fetch_yield_curve()
    credit_spreads = macro.fetch_credit_spreads()
    sentiment = macro.fetch_sentiment()
    econ_calendar = macro.fetch_economic_calendar()
    return yield_curve, credit_spreads, sentiment, econ_calendar

with st.spinner("Initializing Zero-Trust Data Engine (The 99)..."):
    market_close, market_vol, earnings_df, futures_df = get_market_data(start_date)
    macro_yield, macro_credit, macro_sentiment, econ_df = get_macro_data()

if market_close.empty:
    st.error("Failed to fetch market data. Please check connection.")
    st.stop()

# Display Catalysts in Sidebar
st.sidebar.markdown("**Next Earnings:**")
if not earnings_df.empty:
    st.sidebar.dataframe(earnings_df, hide_index=True)
else:
    st.sidebar.text("No upcoming earnings found.")

st.sidebar.markdown("**Economic Calendar:**")
if not econ_df.empty:
    st.sidebar.dataframe(econ_df, hide_index=True)
else:
    st.sidebar.text("No releases found.")

# 5. Math Engine
@st.cache_data
def run_math(df_c, df_v):
    turbulence = math_engine.calculate_turbulence(df_c)
    absorption = math_engine.calculate_absorption_ratio(df_c)
    
    if "SPY" in df_c.columns:
        hurst_spy = math_engine.calculate_hurst(df_c["SPY"])
        amihud_spy = math_engine.calculate_amihud(df_c["SPY"], df_v["SPY"])
    else:
        hurst_spy = pd.Series()
        amihud_spy = pd.Series()
        
    rotation_df = math_engine.calculate_capital_rotation(df_c)
    crypto_z = math_engine.calculate_crypto_stress(df_c)
    
    # Sector Turbulence
    ai_turb = math_engine.calculate_sector_turbulence(df_c, config.GROWTH_ASSETS)
    crypto_turb = math_engine.calculate_sector_turbulence(df_c, config.CRYPTO_ASSETS)
    
    # Macro Ratios
    macro_ratios = math_engine.calculate_institutional_ratios(df_c)
    
    # Cycle Detection
    cycle_phase, cycle_details = cycle_engine.detect_market_cycle(df_c)
    
    return (turbulence, absorption, hurst_spy, amihud_spy, rotation_df, 
            crypto_z, cycle_phase, cycle_details, macro_ratios, ai_turb, crypto_turb)

(turb_series, abs_series, hurst_series, amihud_series, rotation_data, 
 last_crypto_z, current_cycle, cycle_data, macro_ratios_df, 
 ai_turb_series, crypto_turb_series) = run_math(market_close, market_vol)

# Slice Data (Display Period)
analysis_ts = pd.Timestamp(analysis_date)
start_ts = pd.Timestamp(start_date)

curr_close = market_close.loc[start_ts:analysis_ts]
curr_turb = turb_series.loc[start_ts:analysis_ts]
curr_abs = abs_series.loc[start_ts:analysis_ts]
curr_hurst = hurst_series.loc[start_ts:analysis_ts] if not hurst_series.empty else pd.Series()
curr_amihud = amihud_series.loc[start_ts:analysis_ts] if not amihud_series.empty else pd.Series()
curr_yield = macro_yield.loc[start_ts:analysis_ts] if not macro_yield.empty else pd.Series()
curr_ai_turb = ai_turb_series.loc[start_ts:analysis_ts] if not ai_turb_series.empty else pd.Series()
curr_crypto_turb = crypto_turb_series.loc[start_ts:analysis_ts] if not crypto_turb_series.empty else pd.Series()

# Latest Values
if curr_turb.empty:
    st.warning("No data for date.")
    st.stop()
    
last_turb = curr_turb.iloc[-1]
last_abs = curr_abs.iloc[-1]
last_hurst = curr_hurst.iloc[-1] if not curr_hurst.empty else 0.5
last_amihud = curr_amihud.iloc[-1] if not curr_amihud.empty else 0.0
last_yield = curr_yield.iloc[-1] if not curr_yield.empty else 0.0

# Sector Turbulence Ratios
last_ai_turb = curr_ai_turb.iloc[-1] if not curr_ai_turb.empty else last_turb
last_crypto_turb = curr_crypto_turb.iloc[-1] if not curr_crypto_turb.empty else last_turb
safe_turb = last_turb if last_turb > 1 else 1.0
ai_ratio = last_ai_turb / safe_turb
crypto_ratio = last_crypto_turb / safe_turb

# Regime & Signals
spy_col = "SPY"
spy_flat = False
if spy_col in curr_close.columns:
    spy_p = curr_close[spy_col]
    ma50 = spy_p.rolling(50).mean().iloc[-1]
    price = spy_p.iloc[-1]
    trend = "UP" if price > ma50 else "DOWN"
    if len(spy_p) > 1:
        spy_ret = (spy_p.iloc[-1] / spy_p.iloc[-2]) - 1
        spy_flat = abs(spy_ret) < 0.005
else:
    trend = "UNKNOWN"
    spy_p = pd.Series()
    price, ma50 = 0.0, 0.0

regime = math_engine.get_market_regime(last_turb, last_abs, trend)
crypto_stress_signal = (last_crypto_z > 2.0) and spy_flat
super_signal = math_engine.generate_super_signal(last_amihud, last_hurst, curr_close["SPY"])

# 6. UI Layout

# Time Context
last_date = curr_turb.index[-1]
day_name = last_date.strftime("%A")
is_weekend = day_name in ["Saturday", "Sunday"]

st.sidebar.markdown(f"**Data Horizon:** {last_date.date()} ({day_name})")

# Generate Report
vix_val = curr_close["^VIX"].iloc[-1] if "^VIX" in curr_close.columns else 0.0
is_divergence = (last_turb > 180) and (trend == "UP")

days_elevated = 0
if not curr_turb.empty:
    elevated_mask = curr_turb > 180
    if elevated_mask.iloc[-1]:
        for x in elevated_mask[::-1]:
            if x: days_elevated += 1
            else: break

status_report = report_generator.generate_immune_report(
    date=last_date.date(),
    turbulence_score=last_turb,
    spx_price=price,
    spx_ma50=ma50,
    vix_value=vix_val,
    is_divergence=is_divergence,
    regime=regime,
    sentiment_score=macro_sentiment,
    absorption_ratio=last_abs,
    days_elevated=days_elevated
)

# Display Report
with st.container(border=True):
    c_head1, c_head2 = st.columns([3, 1])
    c_head1.subheader(f"🛡️ IMMUNE SYSTEM STATUS: {status_report['warning_level']}")
    if status_report['badge_color'] == 'green':
        c_head2.success("System Healthy")
    elif status_report['badge_color'] == 'yellow':
        c_head2.warning("System Elevated")
    else:
        c_head2.error("System Critical")
        
    st.divider()
    
    # Executive Summary (Consolidated Narrative)
    st.markdown("#### 📝 Executive Summary")
    st.markdown(
        f"""<div style="font-size: 1.15rem; font-style: italic; color: #DDDDDD; line-height: 1.6; margin-bottom: 15px;">
        {status_report['summary_narrative']}
        </div>""", 
        unsafe_allow_html=True
    )

# Advanced Quant Signals
with st.expander("⚡ Advanced Quant Signals", expanded=False):
    # Calculate RSI locally for breakdown
    delta = curr_close["SPY"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    last_rsi = rsi.iloc[-1] if not rsi.empty else 50

    aq_cols = st.columns([1, 2])

    with aq_cols[0]:
        # Main Signal
        signal_color = "normal"
        if "BUY" in super_signal: signal_color = "off" # Greenish usually
        elif "SELL" in super_signal: signal_color = "inverse" # Red
        
        st.metric("Math Super-Signal", "ACTIVE", delta=super_signal, delta_color=signal_color)
        st.caption("Synthesis of Liquidity, Fractals, and Momentum.")

    with aq_cols[1]:
        with st.container(border=True):
            st.markdown("#### 🔬 Signal Analysis")
            
            # 1. Signal Breakdown
            st.markdown("**1. Signal Breakdown**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Liquidity (Driver)", f"{last_amihud:.1f}σ", "Stress" if last_amihud > 1 else "Flowing", delta_color="inverse")
            c2.metric("Structure (Hurst)", f"{last_hurst:.2f}", "Fragile" if last_hurst > 0.65 else "Robust", delta_color="inverse")
            c3.metric("Momentum (RSI)", f"{last_rsi:.0f}", "Overbought" if last_rsi > 70 else "Oversold" if last_rsi < 30 else "Neutral", delta_color="inverse")
            
            # 2. Weighting
            st.markdown("**2. Weighting Rationale:** Liquidity is the *prerequisite* for price movement. Trend Structure (Hurst) determines if a move is sustainable. Momentum (RSI) is the trigger.")
            
            # 3. Predictive Power
            st.markdown("**3. Predictive Power:** Historically effective at identifying *structural exhaustion* before price reversal. High success rate when Liquidity Stress aligns with Extremes in Hurst.")
            
            # 4. Actionable Insights
            st.markdown("**4. Actionable Insights:**")
            if "BUY" in super_signal:
                st.success("✅ **Aggressive Entry:** Liquidity returning to oversold market. Look for mean reversion.")
            elif "SELL" in super_signal:
                st.error("🛑 **Exit/Hedge:** Market is fragile, illiquid, and overextended. Crash risk high.")
            else:
                st.info("⏸️ **Wait:** Market structure is stable. Trade the trend, but verify setup.")

st.markdown("---")

# Institutional Macro Ratios
st.markdown("### 🏛️ Institutional Macro Ratios")
st.caption("Strategic recommendations based on relative asset flows (20-day trend).")

if not macro_ratios_df.empty:
    cols = st.columns(2)
    for i, (idx, row) in enumerate(macro_ratios_df.iterrows()):
        col_idx = i % 2
        with cols[col_idx]:
            with st.container(border=True):
                interpretations = {
                    "EEM/SPY": "Overweight Emerging Markets" if row['Trend'] == "Rising" else "Underweight Emerging Markets",
                    "SPY/TLT": "Risk-On: Stocks > Bonds" if row['Trend'] == "Rising" else "Risk-Off: Bonds > Stocks",
                    "GLD/SPY": "Defensive Rotation: Gold Leading" if row['Trend'] == "Rising" else "Growth Rotation: Stocks Leading",
                    "XLY/XLP": "Confident Consumer (Cyclical)" if row['Trend'] == "Rising" else "Defensive Consumer (Staples)",
                    "CPER/GLD": "Reflationary Growth (Dr. Copper)" if row['Trend'] == "Rising" else "Inflation/Fear Hedge (Gold)"
                }
                st.markdown(f"**{row['Pair']} ({row['Trend']})**")
st.markdown("---")

# Cycle Playbook
playbook = cycle_playbook.get_cycle_playbook(current_cycle)

st.markdown(f"### 📘 Playbook: {playbook['title']}")
with st.container(border=True):
    cp1, cp2, cp3 = st.columns([2, 1, 1])
    
    with cp1:
        st.markdown("#### 🧠 The Layman's Strategy")
        st.info(playbook['layman_strategy'])
        st.caption(f"**Context:** {playbook['context']}")
        
    with cp2:
        st.markdown("#### 🎨 Style Rotation")
        st.metric("Focus Style", playbook['style'])
        
    with cp3:
        st.markdown("#### 🔄 Capital Rotation")
        if playbook['capital_rotation']['Buy']:
            st.success(f"**BUY:** {', '.join(playbook['capital_rotation']['Buy'])}")
        if playbook['capital_rotation']['Sell/Avoid']:
            st.error(f"**AVOID:** {', '.join(playbook['capital_rotation']['Sell/Avoid'])}")

st.markdown("---")

# Metrics Row
turb_delta = "Weekend Mode" if is_weekend else "Low Vol" if last_turb < 50 else "Active"

# Dynamic Interpretations
# 1. Turbulence
turb_interp = "Calm/Normal"
if last_turb > 180: turb_interp = "Elevated Stress"
if last_turb > 370: turb_interp = "CRITICAL Instability"
turb_help = f"""**Definition:** Measures how 'weird' or unusual today's price moves are compared to the last year.\n\n**Significance:** High turbulence often precedes crashes. It detects hidden stress before price drops.\n\n**Current Status:** {last_turb:.0f} -> {turb_interp}."""

# 2. Fragility (Absorption)
abs_interp = "Resilient (Diverse)"
if last_abs > 0.80: abs_interp = "Highly Fragile (Unified)"
abs_help = f"""**Definition:** The % of assets moving in lockstep.\n\n**Significance:** When everything moves together (>80%), diversification fails. A crash in one asset drags down everything.\n\n**Current Status:** {last_abs:.0%}: -> {abs_interp}."""

# 3. Hurst
hurst_interp = "Random Walk (Healthy)"
if last_hurst > 0.65: hurst_interp = "Trending"
if last_hurst > 0.75: hurst_interp = "Crowded/Brittle Trend"
hurst_help = f"""**Definition:** Measures how 'persistent' a trend is.\n\n**Significance:** High scores (>0.75) mean everyone is on the same side of the trade. If they rush for the exit, price collapses.\n\n**Current Status:** {last_hurst:.2f} -> {hurst_interp}."""

# 4. Liquidity (Amihud)
liq_interp = "Normal Liquidity"
if last_amihud > 1.0: liq_interp = "Thin Liquidity"
if last_amihud > 2.0: liq_interp = "Liquidity Hole (Danger)"
liq_help = f"""**Definition:** How much price moves per dollar traded.\n\n**Significance:** 'Liquidity Holes' mean small sell orders cause huge price drops. Essential for crash detection.\n\n**Current Status:** {last_amihud:.1f}σ -> {liq_interp}."""

# 5. Sentiment
sent_interp = "Neutral"
if macro_sentiment > 60: sent_interp = "Greed (Contrarian Sell?)"
if macro_sentiment < 40: sent_interp = "Fear (Contrarian Buy?)"
sent_help = f"""**Definition:** Aggregated mood from top financial news headlines.\n\n**Significance:** Extreme Greed (>80) often marks tops; Extreme Fear (<20) often marks bottoms.\n\n**Current Status:** {macro_sentiment:.0f} -> {sent_interp}."""

# 6. AI Ratio
ai_interp = "Normal"
if ai_ratio > 1.5: ai_interp = "Overheated (Bubble Risk)"
if ai_ratio < 0.8: ai_interp = "Lagging Market"
ai_help = f"""**Definition:** Volatility of AI stocks relative to the broad market.\n\n**Significance:** >1.5x means AI is decoupling (Bubble behavior). High risk of mean reversion.\n\n**Current Status:** {ai_ratio:.1f}x -> {ai_interp}."""

# 7. Crypto Ratio
crypto_interp = "Normal"
if crypto_ratio > 1.5: crypto_interp = "Speculative Excess"
crypto_help = f"""**Definition:** Volatility of Crypto relative to the broad market.\n\n**Significance:** Often a leading indicator for risk appetite. If Crypto cracks, stocks often follow.\n\n**Current Status:** {crypto_ratio:.1f}x -> {crypto_interp}."""

m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
m1.metric("Turbulence", f"{last_turb:.0f}", delta=turb_delta, delta_color="off", help=turb_help)
m2.metric("Fragility", f"{last_abs*100:.0f}%", help=abs_help)
m3.metric("Hurst", f"{last_hurst:.2f}", help=hurst_help)
m4.metric("Liquidity", f"{last_amihud:.1f}", help=liq_help)
m5.metric("Sentiment", f"{macro_sentiment:.0f}", help=sent_help)
m6.metric("AI/Mkt Ratio", f"{ai_ratio:.1f}x", help=ai_help)
m7.metric("Crypto/Mkt", f"{crypto_ratio:.1f}x", help=crypto_help)

# Macro Row
st.markdown("#### 🌍 Macro Truth")
mac1, mac2 = st.columns(2)

# Yield Curve Help
yield_interp = "Normal (Growth)"
if last_yield < 0: yield_interp = "Inverted (Recession Warning)"
elif last_yield < 0.2: yield_interp = "Flat (Caution)"
yield_help = f"""**Definition:** The difference between 10-Year and 2-Year Treasury yields.\n\n**Significance:** The most reliable recession predictor in history. Inversion (<0) signals trouble ahead.\n\n**Current Status:** {last_yield:.2f}% -> {yield_interp}."""

# Credit Stress Help
credit_val = macro_credit if isinstance(macro_credit, float) else 0.0
credit_interp = "Stable"
if credit_val > 1.0: credit_interp = "Stress Rising"
if credit_val > 2.0: credit_interp = "Credit Freeze"
credit_help = f"""**Definition:** High Yield Bond Spreads (Risk Premium).\n\n**Significance:** If lenders demand high interest to lend to risky companies, the credit cycle is breaking.\n\n**Current Status:** {credit_val:.1f}σ -> {credit_interp}."""

mac1.metric(f"Yield Curve (10Y-2Y)", f"{last_yield:.2f}%", delta="Inverted" if last_yield < 0 else "Normal", help=yield_help)
mac2.metric("Credit Stress (HY)", f"{credit_val:.1f}σ" if isinstance(macro_credit, float) else "N/A", help=credit_help)

# Charts
st.markdown("### 📉 Market Health Monitor")
if not spy_p.empty:
    fig_main = charts.plot_divergence_chart(spy_p, curr_turb, futures_data=futures_df)
    st.plotly_chart(fig_main, use_container_width=True)

# Narrative Battle
st.markdown("### ⚔️ Narrative Battle")
crypto_cols = [c for c in config.CRYPTO_ASSETS if c in curr_close.columns]
growth_cols = [c for c in config.GROWTH_ASSETS if c in curr_close.columns]
if crypto_cols and growth_cols:
    c_df = curr_close[crypto_cols].sum(axis=1)
    a_df = curr_close[growth_cols].sum(axis=1)
    fig_battle = charts.plot_narrative_battle(c_df, a_df)
    st.plotly_chart(fig_battle, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Zero-Trust Architecture**")
st.sidebar.markdown(f"Assets Tracked: {len(market_close.columns)}")