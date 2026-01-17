import streamlit as st
import pandas as pd
import datetime
import config
from core import data_loader, math_engine, macro_connector, cycle_engine
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
    df_close, df_vol = data_loader.fetch_market_data(config.ASSET_UNIVERSE, start_date=start_date)
    # Earnings (Fast changing)
    earnings = data_loader.fetch_next_earnings(config.GROWTH_ASSETS, limit=10)
    return df_close, df_vol, earnings

@st.cache_data(ttl=86400) # 24h Cache for Macro
def get_macro_data():
    macro = macro_connector.MacroConnector()
    yield_curve = macro.fetch_yield_curve()
    credit_spreads = macro.fetch_credit_spreads()
    sentiment = macro.fetch_sentiment()
    econ_calendar = macro.fetch_economic_calendar()
    return yield_curve, credit_spreads, sentiment, econ_calendar

with st.spinner("Initializing Zero-Trust Data Engine (The 99)..."):
    market_close, market_vol, earnings_df = get_market_data(start_date)
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
    
    # Cycle Detection
    cycle_phase, cycle_details = cycle_engine.detect_market_cycle(df_c)
    
    return turbulence, absorption, hurst_spy, amihud_spy, rotation_df, crypto_z, cycle_phase, cycle_details

turb_series, abs_series, hurst_series, amihud_series, rotation_data, last_crypto_z, current_cycle, cycle_data = run_math(market_close, market_vol)

# Slice Data
analysis_ts = pd.Timestamp(analysis_date)
curr_close = market_close.loc[:analysis_ts]
curr_turb = turb_series.loc[:analysis_ts]
curr_abs = abs_series.loc[:analysis_ts]
curr_hurst = hurst_series.loc[:analysis_ts] if not hurst_series.empty else pd.Series()
curr_amihud = amihud_series.loc[:analysis_ts] if not amihud_series.empty else pd.Series()
curr_yield = macro_yield.loc[:analysis_ts] if not macro_yield.empty else pd.Series()

# Latest Values
if curr_turb.empty:
    st.warning("No data for date.")
    st.stop()
    
last_turb = curr_turb.iloc[-1]
last_abs = curr_abs.iloc[-1]
last_hurst = curr_hurst.iloc[-1] if not curr_hurst.empty else 0.5
last_amihud = curr_amihud.iloc[-1] if not curr_amihud.empty else 0.0
last_yield = curr_yield.iloc[-1] if not curr_yield.empty else 0.0

# Regime & Signals
spy_col = "SPY"
spy_flat = False
if spy_col in curr_close.columns:
    spy_p = curr_close[spy_col]
    ma50 = spy_p.rolling(50).mean().iloc[-1]
    price = spy_p.iloc[-1]
    trend = "UP" if price > ma50 else "DOWN"
    
    # Check if SPY is flat (1-day return)
    if len(spy_p) > 1:
        spy_ret = (spy_p.iloc[-1] / spy_p.iloc[-2]) - 1
        spy_flat = abs(spy_ret) < 0.005 # < 0.5%
else:
    trend = "UNKNOWN"
    spy_p = pd.Series()

regime = math_engine.get_market_regime(last_turb, last_abs, trend)
crypto_stress_signal = (last_crypto_z > 2.0) and spy_flat

# 6. UI Layout

# Header Banner
if regime in ["CRASH ALERT", "SYSTEMIC SELL-OFF"]:
    st.error(f"⚠️ SYSTEM WARNING: {regime} DETECTED")
elif regime == "FRAGILE RALLY":
    st.warning(f"⚠️ SYSTEM WARNING: {regime} DETECTED")
elif crypto_stress_signal:
    st.warning("⚠️ ALERT: Crypto-Led Stress Detected")
else:
    st.success(f"System Status: {regime}")

# The Tactical HUD
st.markdown("### 🛸 Tactical HUD")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"**Regime Status**")
    st.metric("Regime", regime, delta="Structure Check")
    english_map = {
        "FRAGILE RALLY": "Prices rising, but market locked. Drop likely.",
        "STRUCTURAL DIVERGENCE": "Price masking internal break. Do not chase.",
        "SYSTEMIC SELL-OFF": "Everything falling together. Cash is safe.",
        "CRASH ALERT": "Extreme volatility. Protect capital.",
        "NORMAL": "Structure stable.",
        "WARNING": "Caution advised."
    }
    st.caption(english_map.get(regime, ""))

with c2:
    st.markdown("**Safety Checks**")
    if regime in ["NORMAL", "FRAGILE RALLY"]:
        st.write("🛑 Stops: **Standard**")
        st.write("⚖️ Leverage: **Allowed**")
    else:
        st.write("🛑 Stops: **Tight**")
        st.write("⚖️ Leverage: **Restricted**")
        
    if last_turb > 180:
        st.write("🛡️ Hedging: **Recommended**")
    else:
        st.write("🛡️ Hedging: **Optional**")

with c3:
    st.markdown("**Market Cycle**")
    st.metric("Phase", current_cycle.split(":")[0]) # Show "Phase I", "Phase II" etc
    
    # Cycle Explainer Tooltip logic
    cycle_desc = {
        "Phase I: Early Cycle (Recovery)": "Economy recovering. Rates low. Best: Financials, Real Estate, Discretionary.",
        "Phase II: Mid-Cycle (Expansion)": "Growth steady. Profits peak. Best: Tech, Comm Services.",
        "Phase III: Late Cycle (Slowdown)": "Inflation high. Rates rising. Best: Energy, Materials.",
        "Phase IV: Recession (Contraction)": "Economy shrinking. Fear high. Best: Staples, Healthcare, Utilities."
    }
    
    st.caption(f"*{cycle_desc.get(current_cycle, '')}*")
    st.caption("Forecast Window: 7-14 Days")

with c4:
    st.markdown("**Analysis**")
    if crypto_stress_signal:
        st.error("⚠️ Crypto-Led Stress")
        st.caption("Pre-cursor to broad volatility.")
    elif regime == "SYSTEMIC SELL-OFF":
        st.error("Systemic Failure")
    else:
        st.write("No anomalies detected.")

with st.expander("📘 Regime Definitions (Layman's Guide)"):
    st.markdown("""
    **What is a 'Fragile Rally'?**
    Imagine a car speeding up (Prices Rising) while the engine is shaking violently (High Fragility/Absorption).
    - **Context:** Investors are buying, but they are all buying the *same* few stocks. If one falls, they all fall.
    - **Risk:** High chance of a sudden "air pocket" drop.
    
    **Other Regimes:**
    - **Normal:** Smooth sailing. Low turbulence, independent asset movement.
    - **Structural Divergence:** Price is going up, but the "smart money" (Turbulence) detects cracks under the surface. A trap.
    - **Systemic Sell-Off:** The crash. Everything falls together. Cash is King.
    - **Crash Alert:** Extreme volatility readings usually seen *during* or immediately *before* a collapse.
    """)
    
    st.markdown("---")
    st.markdown("""
    **The 4 Market Cycles:**
    1. **Early Cycle (Recovery):** Post-recession. Low rates. *Buy Banks, Real Estate.*
    2. **Mid-Cycle (Expansion):** Steady growth. *Buy Tech.*
    3. **Late Cycle (Slowdown):** Overheating/Inflation. *Buy Energy, Materials.*
    4. **Recession (Contraction):** Fear. *Buy Toothpaste (Staples), Utilities.*
    """)

st.markdown("---")

# Metrics Row
last_date = curr_turb.index[-1]
day_name = last_date.strftime("%A")
is_weekend = day_name in ["Saturday", "Sunday"]

st.sidebar.markdown(f"**Data Horizon:** {last_date.date()} ({day_name})")

turb_delta = "Weekend Mode" if is_weekend else "Low Vol" if last_turb < 50 else "Active"

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(
    "Turbulence (0-1000)", 
    f"{last_turb:.0f}", 
    delta=turb_delta,
    delta_color="off",
    help="Measures how 'unusual' today's market behavior is compared to history. High scores signal structural breaks or hidden stress."
)
m2.metric(
    "Fragility (Abs %)", 
    f"{last_abs*100:.1f}%", 
    delta_color="inverse",
    help="Measures how much assets are moving in lockstep. High % means the market is rigid and prone to a systemic crash."
)
m3.metric(
    "Hurst Exp (>0.75)", 
    f"{last_hurst:.2f}", 
    delta="Brittle" if last_hurst>0.75 else "Healthy", 
    delta_color="inverse",
    help="Measures trend persistence. Above 0.75 means a trend is 'crowded' and more likely to reverse violently."
)
m4.metric(
    "Liquidity (Amihud Z)", 
    f"{last_amihud:.2f}", 
    delta="Stress" if last_amihud>2.0 else "Stable", 
    delta_color="inverse",
    help="Measures how easily prices move per dollar traded. High Z-score indicates a 'Liquidity Hole' where small trades cause big drops."
)
m5.metric(
    "Sentiment (0-100)", 
    f"{macro_sentiment:.0f}",
    help="Aggregated mood from the top 20 news headlines. 100 = Extremely Bullish, 0 = Extremely Bearish."
)

# Macro Row
st.markdown("#### 🌍 Macro Truth")
mac1, mac2 = st.columns(2)
# Update label with date if available
yield_date = curr_yield.name.date() if hasattr(curr_yield, 'name') and curr_yield.name else "Latest"
mac1.metric(f"Yield Curve (10Y-2Y)", f"{last_yield:.2f}%", delta="Inverted" if last_yield < 0 else "Normal")
mac2.metric("Credit Stress (HY Spread Z)", f"{macro_credit if isinstance(macro_credit, float) else 'N/A'}")

# Charts
st.markdown("### 📉 Market Health Monitor")
if not spy_p.empty:
    fig_main = charts.plot_divergence_chart(spy_p, curr_turb)
    st.plotly_chart(fig_main, use_container_width=True)

# Narrative Battle
st.markdown("### ⚔️ Narrative Battle")
# Crypto vs AI
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