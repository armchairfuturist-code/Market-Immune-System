import streamlit as st
import pandas as pd
import datetime
import re
import config
from core import data_loader, math_engine, macro_connector, cycle_engine, report_generator, cycle_playbook, smc_engine
from core.vector_engine import VectorEngine
from ui import charts

def determine_business_cycle(yield_curve, trend, credit_stress, turbulence, sentiment, dix):
    """Maps data to the 4-stage Business Cycle"""
    if yield_curve > 0.2 and trend == "UP" and credit_stress < 1.0:
        return {"phase": "Expansion", "next": "Peak", "assets": "Growth Stocks, Real Estate, Commodities", "progress": 25, "warning": "Horizon looks clear. Monitor for Yield Curve flattening (<0.2)."}
    elif yield_curve <= 0.2 and (turbulence > 180 or sentiment > 60):
        return {"phase": "Peak", "next": "Contraction", "assets": "Defensives, Gold, Cash", "progress": 50, "warning": "Structural fever detected. Transition to risk-off strategy."}
    elif yield_curve < 0 or (trend == "DOWN" and credit_stress > 1.5):
        return {"phase": "Contraction", "next": "Trough", "assets": "Bonds, Gold, Staples", "progress": 75, "warning": "Economic winter. Watch for Institutional Accumulation (DIX > 45%)."}
    elif sentiment < 30 and dix > 45:
        return {"phase": "Trough", "next": "Expansion", "assets": "Value Stocks, Small Caps", "progress": 100, "warning": "Market is washing out. Institutions are buying. Plan for recovery."}
    else:
        return {"phase": "Transition", "next": "Expansion", "assets": "Cash/Neutral", "progress": 10, "warning": "Market in structural shift. Stay patient."}

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    st.warning("yfinance not available. Please install with: pip install yfinance")

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

# Market Status Badge
def is_bank_holiday(check_date):
    """Check for US Market Holidays"""
    # Fix: January 20th is NOT a holiday in 2026 (MLK is Jan 19)
    # Simple fix for current context:
    holidays = [(1, 1), (1, 19), (7, 4), (12, 25)]
    return (check_date.month, check_date.day) in holidays

def get_market_status():
    """Determine market status and return status info"""
    current_time = datetime.datetime.now()
    day_name = current_time.strftime("%A")

    # Check if it's weekend
    if day_name in ["Saturday", "Sunday"]:
        return {
            "status": "CLOSED",
            "reason": "Weekend",
            "is_trading": False,
            "impacts": ["real_time_data", "institutional_footprints", "volume_analysis"]
        }

    # Check if it's a bank holiday
    if is_bank_holiday(current_time.date()):
        return {
            "status": "CLOSED",
            "reason": "Bank Holiday",
            "is_trading": False,
            "impacts": ["real_time_data", "institutional_footprints", "volume_analysis"]
        }

    # Check if market hours (9:30 AM - 4:00 PM ET)
    # Convert to ET for market hours check
    et_offset = datetime.timedelta(hours=5)  # UTC to ET
    et_time = current_time - et_offset
    
    # Market hours: 9:30 AM to 4:00 PM ET
    market_open = datetime.time(9, 30)
    market_close = datetime.time(16, 0)
    
    if et_time.time() < market_open or et_time.time() > market_close:
        return {
            "status": "AFTER HOURS",
            "reason": "Market Closed (Outside Trading Hours)",
            "is_trading": False,
            "impacts": ["real_time_data", "institutional_footprints"]
        }
    
    return {
        "status": "OPEN",
        "reason": "Trading Hours",
        "is_trading": True,
        "impacts": []
    }

market_status = get_market_status()
is_weekend = market_status["status"] == "CLOSED" and market_status["reason"] in ["Weekend", "Bank Holiday"]

status_color_map = {
    "OPEN": "green",
    "AFTER HOURS": "yellow",
    "CLOSED": "red"
}

status_icon_map = {
    "OPEN": "✅",
    "AFTER HOURS": "🌙",
    "CLOSED": "⚠️"
}

status_badge = f"{status_icon_map[market_status['status']]} **MARKET {market_status['status']}**"
status_badge += f" - {market_status['reason']}"

if market_status['impacts']:
    impacted_vars = ", ".join([f"`{var}`" for var in market_status['impacts']])
    status_badge += f" | **Impacted:** {impacted_vars}"

if market_status['status'] == "OPEN":
    st.sidebar.success(status_badge)
elif market_status['status'] == "AFTER HOURS":
    st.sidebar.warning(status_badge)
else:
    st.sidebar.error(status_badge)

if not market_status['is_trading']:
    st.sidebar.caption(f"**Note:** {market_status['reason']} - Analysis based on last available data. Real-time updates paused.")

st.sidebar.markdown("---")

# Data Source Selection
st.sidebar.markdown("### 📊 Data Source")
data_source = st.sidebar.radio(
    "Select Data Source",
    options=["yfinance (Default)", "OpenBB (Faster)"],
    index=0,
    help="OpenBB provides more reliable data but requires installation"
)
use_openbb = data_source == "OpenBB (Faster)"

# Time Travel Cap
today = datetime.datetime.now().date()
analysis_date = st.sidebar.date_input("Analysis Date", today, max_value=today)

# Data Start Date (Replaces Lookback)
default_start = today - datetime.timedelta(days=730)
start_date = st.sidebar.date_input("Data Start Date", default_start, max_value=today)

# Catalyst Watch (Earnings & Macro)
st.sidebar.markdown("### 📅 Catalyst Watch")

# 4. Data Loading
@st.cache_data(ttl=3600)
def get_market_data(start_date, use_openbb=False):
    # Fetch with buffer for calculations (Turbulence needs 365d history)
    if isinstance(start_date, datetime.datetime):
        start_date = start_date.date()
    buffer_date = start_date - datetime.timedelta(days=365)
    
    df_close, hourly_df = data_loader.fetch_market_data(config.ASSET_UNIVERSE, start_date=buffer_date, use_openbb=use_openbb)
    # Earnings (Fast changing)
    earnings = data_loader.fetch_next_earnings(config.GROWTH_ASSETS, limit=10, use_openbb=use_openbb)
    # Futures Trend
    futures = data_loader.fetch_futures_data(period="3mo", use_openbb=use_openbb)
    
    return df_close, earnings, futures, hourly_df

@st.cache_data(ttl=86400) # 24h Cache for Macro
def get_macro_data():
    macro = macro_connector.MacroConnector()
    yield_curve = macro.fetch_yield_curve()
    credit_spreads = macro.fetch_credit_spreads()
    sentiment = macro.fetch_sentiment()
    m2_supply = macro.fetch_m2_money_supply()
    cpi = macro.fetch_cpi()
    nvidia_sentiment = macro.fetch_sentiment_nvidia()
    bitcoin_sentiment = macro.fetch_sentiment_bitcoin()
    econ_calendar = macro.fetch_economic_calendar()
    whale_tracker = macro.get_whale_tracker()
    return yield_curve, credit_spreads, sentiment, m2_supply, cpi, nvidia_sentiment, bitcoin_sentiment, econ_calendar, whale_tracker

st.sidebar.markdown("---")
st.sidebar.info("v3.0 | Zero-Trust Engine")

# 4. Data Loading & State Management
if 'data' not in st.session_state or st.sidebar.button("Forced Reload"):
    st.cache_data.clear()
    with st.spinner("Initializing Zero-Trust Data Engine (The 99)..."):
        market_close, earnings_df, futures_df, hourly_df = get_market_data(start_date, use_openbb=use_openbb)
        macro_yield, macro_credit, macro_sentiment, macro_m2, macro_cpi, nvidia_sent, bitcoin_sent, econ_df, whale_tracker = get_macro_data()
        
        # Vectorize volume profile
        vec_engine = VectorEngine()
        vec_engine.vectorize_volume_profile(hourly_df, ticker="SPY")

        st.session_state.data = {
            'market_close': market_close,
            'earnings_df': earnings_df,
            'futures_df': futures_df,
            'hourly_df': hourly_df,
            'macro_yield': macro_yield,
            'macro_credit': macro_credit,
            'macro_sentiment': macro_sentiment,
            'macro_m2': macro_m2,
            'macro_cpi': macro_cpi,
            'nvidia_sentiment': nvidia_sent,
            'bitcoin_sentiment': bitcoin_sent,
            'econ_df': econ_df,
            'whale_tracker': whale_tracker,
            'vector_engine': vec_engine
        }
else:
    market_close = st.session_state.data['market_close']
    earnings_df = st.session_state.data['earnings_df']
    futures_df = st.session_state.data['futures_df']
    hourly_df = st.session_state.data.get('hourly_df', pd.DataFrame()) # Handle legacy state
    macro_yield = st.session_state.data['macro_yield']
    macro_credit = st.session_state.data['macro_credit']
    macro_sentiment = st.session_state.data['macro_sentiment']
    macro_m2 = st.session_state.data.get('macro_m2', pd.Series([21.7e12] * 100))
    macro_cpi = st.session_state.data.get('macro_cpi', pd.Series([3.2] * 100))
    nvidia_sent = st.session_state.data.get('nvidia_sentiment', 0.0)
    bitcoin_sent = st.session_state.data.get('bitcoin_sentiment', 0.0)
    econ_df = st.session_state.data['econ_df']
    whale_tracker = st.session_state.data.get('whale_tracker', {"dix": 50.0, "distribution": False, "volume_anomaly": False})
    vec_engine = st.session_state.data.get('vector_engine')

if market_close.empty:
    st.error("Failed to fetch market data. Please check connection.")
    st.stop()

# Macro Liquidity
with st.sidebar.expander("💧 Macro Liquidity"):
    latest_cpi = macro_cpi.iloc[-1] if not macro_cpi.empty else 3.2
    latest_m2 = macro_m2.iloc[-1] if not macro_m2.empty else 21.7e12
    st.metric("Inflation (CPI)", f"{latest_cpi:.1f}%", help="Consumer Price Index - current inflation rate.")
    st.metric("M2 Money Supply", f"${latest_m2 / 1e12:.1f}T", help="M2 Money Supply - total money in circulation.")
    st.metric("Unemployment Rate", "4.1%", help="Current unemployment rate.")

# Placeholder for last_yield (will be updated after data loading)
last_yield = 0.0

# Calculate Market-Implied Odds
inversion_depth = -last_yield if last_yield < 0 else 0
if inversion_depth == 0:
    recession_odds = 10
elif inversion_depth < 0.5:
    recession_odds = 20
elif inversion_depth < 1.5:
    recession_odds = 40
else:
    recession_odds = 70

# Assume current inflation (in production, fetch real-time)
inflation = 3.2  # Placeholder from sidebar expander
diff = inflation - 2
if diff > 0.5:
    fed_pivot_odds = 60  # Accelerating inflation suggests higher pivot odds
else:
    fed_pivot_odds = 20  # Stable or decelerating

# Market-Implied Odds
st.sidebar.subheader("📊 Market-Implied Odds")
rec_delta = "High" if recession_odds > 75 else "Low" if recession_odds < 25 else None
fed_delta = "High" if fed_pivot_odds > 75 else "Low" if fed_pivot_odds < 25 else None

st.sidebar.metric("Recession Odds", f"{recession_odds}%", delta=rec_delta, help="Yield curve inversion depth predicts recession likelihood. Minimal <0.5%: 10-20%, moderate 0.5-1.5%: 30-50%, deep >1.5%: 60-80%. Leading indicator with ~6-12 month horizon.")
st.sidebar.metric("Fed Pivot Odds", f"{fed_pivot_odds}%", delta=fed_delta, help="Inflation differential from 2% target. >0.5% above target: 50-90% pivot odds (accelerating inflation), <0.5%: 10-40% (decelerating). Indicates Fed funds rate direction.")
# EUR/USD Strategy Logic
eurusd_col = "EURUSD=X"
if eurusd_col in market_close.columns:
    eurusd_p = market_close[eurusd_col]
    eur_ma50 = eurusd_p.rolling(50).mean()
    eur_ma20 = eurusd_p.rolling(20).mean()
    
    last_eur = eurusd_p.iloc[-1]
    last_eur_ma50 = eur_ma50.iloc[-1]
    last_eur_ma20 = eur_ma20.iloc[-1]
    
    if last_eur > last_eur_ma50:
        eur_trend = "UP (EUR Strengthening)"
        eur_rec = "HOLD EUR"
        eur_color = "green"
    elif last_eur < last_eur_ma50:
        eur_trend = "DOWN (USD Strengthening)"
        eur_rec = "HOLD USD"
        eur_color = "blue"
    else:
        eur_trend = "FLAT"
        eur_rec = "NEUTRAL"
        eur_color = "gray"
else:
    eur_trend = "N/A"
    eur_rec = "N/A"
    eur_color = "gray"
    last_eur = 0.0

# 4b. Currency Strategy (Portugal Context)
with st.sidebar.expander("💱 Currency Strategy (USD/EUR)"):
    st.markdown(f"**Rate:** {last_eur:.4f}")
    st.markdown(f"**Trend:** {eur_trend}")

    if eur_rec == "HOLD EUR":
        st.success(f"**STRATEGY: {eur_rec}**")
        st.caption("Euro is gaining momentum against the Dollar.")
    elif eur_rec == "HOLD USD":
        st.info(f"**STRATEGY: {eur_rec}**")
        st.caption("Dollar is strengthening. Protect purchasing power in USD.")
    else:
        st.markdown(f"**STRATEGY: {eur_rec}**")



# 5. Math Engine
@st.cache_data
def run_math(df_c, df_h):
    turbulence = math_engine.calculate_turbulence(df_c)
    absorption = math_engine.calculate_absorption_ratio(df_c)
    
    if "SPY" in df_c.columns:
        hurst_spy = math_engine.calculate_hurst(df_c["SPY"])
        # Amihud calculation requires volume data, which we no longer have
        # Using a placeholder or skipping for now
        amihud_spy = pd.Series()
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
    
    # Institutional Context (SMC)
    smc_context = smc_engine.get_institutional_context(df_c, ticker="SPY", hourly_df=df_h)
    
    return (turbulence, absorption, hurst_spy, amihud_spy, rotation_df,
            crypto_z, cycle_phase, cycle_details, macro_ratios, ai_turb, crypto_turb, smc_context)

(turb_series, abs_series, hurst_series, amihud_series, rotation_data,
 last_crypto_z, current_cycle, cycle_data, macro_ratios_df,
 ai_turb_series, crypto_turb_series, smc_context) = run_math(market_close, hourly_df)



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

# Latest Values
if curr_turb.empty:
    st.warning("No data for date.")
    st.stop()

last_turb = curr_turb.iloc[-1]
last_abs = curr_abs.iloc[-1]
last_hurst = curr_hurst.iloc[-1] if not curr_hurst.empty else 0.5
last_amihud = curr_amihud.iloc[-1] if not curr_amihud.empty else 0.0
last_yield = curr_yield.iloc[-1] if not curr_yield.empty else 0.0
credit_val = macro_credit.iloc[-1] if isinstance(macro_credit, pd.Series) and not macro_credit.empty else 0.0
cycle_info = determine_business_cycle(last_yield, trend, credit_val, last_turb, macro_sentiment, whale_tracker['dix'])


# Sector Turbulence Ratios
last_ai_turb = curr_ai_turb.iloc[-1] if not curr_ai_turb.empty else last_turb
last_crypto_turb = curr_crypto_turb.iloc[-1] if not curr_crypto_turb.empty else last_turb
safe_turb = last_turb if last_turb > 1 else 1.0
ai_ratio = last_ai_turb / safe_turb
crypto_ratio = last_crypto_turb / safe_turb



# 6. SMC Synthesis & HUD Alert Upgrade
regime = math_engine.get_market_regime(last_turb, last_abs, trend)

# Upgrade logic: If Turbulence > 180 AND CHoCH == Bearish (-1)
hud_alert_upgrade = False
if last_turb > 180 and smc_context["choch_val"] == -1:
    hud_alert_upgrade = True

crypto_stress_signal = (last_crypto_z > 2.0) and spy_flat
super_signal = math_engine.generate_super_signal(last_amihud, last_hurst, spy_p)

# Black Swan Verification
black_swan_status = math_engine.verify_black_swan_alignment(last_turb, smc_context["choch_val"], last_yield, price, ma50)
# 6. Report & Context Generation
vix_val = curr_close["^VIX"].iloc[-1] if "^VIX" in curr_close.columns else 0.0
divergence_alert = (last_turb > 180) and (vix_val < 20)
is_divergence = (last_turb > 180) and (trend == "UP")

# Divergence Alert Notice in Sidebar
if divergence_alert and (not st.session_state.get('divergence_dismissed') or (datetime.datetime.now() - st.session_state['divergence_dismissed']).total_seconds() > 86400):
    with st.sidebar.expander("⚠️ Divergence Alert Notice", expanded=True):
        st.write("**Hidden Stress Detected:** Turbulence is rising while VIX remains low, indicating potential early warning for risk-off moves.")
        st.write("**Risk Mitigation Suggestions:**")
        st.markdown("- Consider portfolio rebalancing towards defensive assets.")
        st.markdown("- Monitor for increased volatility.")
        dismiss = st.checkbox("Dismiss this notice for 24 hours")
        if dismiss:
            st.session_state['divergence_dismissed'] = datetime.datetime.now()

days_elevated = 0
if not curr_turb.empty:
    elevated_mask = curr_turb > 180
    if elevated_mask.iloc[-1]:
        for x in elevated_mask[::-1]:
            if x: days_elevated += 1
            else: break

# Time Context & Market Status
last_date = curr_turb.index[-1]
day_name = last_date.strftime("%A")
current_time = datetime.datetime.now()

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

playbook = cycle_playbook.get_cycle_playbook(current_cycle)

# System State Dictionary
ss = {
    "level": "CRITICAL" if last_turb > 370 else "ELEVATED" if last_turb > 180 else "NORMAL",
    "color": "red" if last_turb > 370 else "orange" if last_turb > 180 else "green",
    "trend": trend,
    "is_distribution": smc_context["choch_val"] == -1,
    "vix_alert": vix_val > 20
}

# Determine Business Cycle

# Logic for the Diagnostic Hover
checks = [
    ("High Turbulence", last_turb > 180),
    ("Institutional Distribution", smc_context["choch_val"] == -1),
    ("Inverted Yield", last_yield < 0),
    ("Below 50-day MA", price < ma50)
]
diag_text = "\n".join([f"{'🚨' if state else '✅'} {label}" for label, state in checks])

# 7. UI Layout

st.sidebar.markdown(f"**Data Horizon:** {last_date.date()} ({day_name})")
st.sidebar.caption("**System Horizon vs Data Horizon:**")
st.sidebar.caption("• **System Horizon (7-14 Days):** The forward-looking timeframe the system uses to detect structural breaks and regime changes. It's the 'lookahead' window for identifying emerging risks.")
st.sidebar.caption("• **Data Horizon (Today):** The historical data point being analyzed. This is the 'lookback' anchor - the most recent market close used for current calculations.")


# Dynamic Interpretations
# 1. System Fever (Turbulence)
turb_interp = "NORMAL" if last_turb < 180 else "ELEVATED" if last_turb < 370 else "CRITICAL"
turb_help = f"""**Measures how 'weird' or unusual today's price moves are compared to the last year.**\n\n**Current Status:** {last_turb:.0f} -> {turb_interp}."""

# 2. Contagion Risk
abs_interp = "Resilient (Diverse)" if last_abs < 0.80 else "Highly Fragile (Unified)"
abs_help = f"""**The % of assets moving in lockstep.**\n\n**Current Status:** {last_abs:.0%} -> {abs_interp}."""

# 3. Herd Behavior (Hurst)
hurst_interp = "Random Walk (Healthy)" if last_hurst < 0.70 else "Panic/FOMO - High reversal risk"
hurst_help = f"""**Measures how 'persistent' a trend is.**\n\n**Current Status:** {last_hurst:.2f} -> {hurst_interp}."""

# 4. Market Pulse (Amihud)
liq_interp = "Normal Liquidity" if last_amihud < 2.0 else "Holes Found - Big players struggling"
liq_help = f"""**How much price moves per dollar traded.**\n\n**Current Status:** {last_amihud:.1f}σ -> {liq_interp}."""

# 5. Sentiment
sent_interp = "Fear (Contrarian Buy?)" if macro_sentiment < 40 else "Greed (Contrarian Sell?)" if macro_sentiment > 60 else "Neutral"
sent_help = f"""**Aggregated mood from top financial news headlines.**\n\n**Extreme Greed (>80) often marks tops; Extreme Fear (<20) often marks bottoms.**\n\n**Current Status:** {macro_sentiment:.0f} -> {sent_interp}."""

# 6. AI Ratio
ai_interp = "Lagging Market" if ai_ratio < 0.8 else "Overheated (Bubble Risk)" if ai_ratio > 1.5 else "Normal"
ai_help = f"""**Volatility of AI stocks relative to the broad market.**\n\n**Current Status:** {ai_ratio:.1f}x -> {ai_interp}."""

# 7. Crypto Ratio
crypto_interp = "Normal" if crypto_ratio <= 1.5 else "Speculative Excess"
crypto_help = f"""**Volatility of Crypto relative to the broad market.**\n\n**Often a leading indicator for risk appetite. If Crypto cracks, stocks often follow.**\n\n**Current Status:** {crypto_ratio:.1f}x -> {crypto_interp}."""

# Yield Curve Help
yield_interp = "Inverted (Recession Warning)" if last_yield < 0 else "Flat (Caution)" if last_yield < 0.2 else "Normal (Growth)"
yield_help = f"""**The difference between 10-Year and 2-Year Treasury yields.**\n\n**Current Status:** {last_yield:.2f}% -> {yield_interp}."""

# Credit Stress Help
credit_interp = "Stable" if credit_val < 1.0 else "Stress Rising" if credit_val < 2.0 else "Credit Freeze"
credit_help = f"""**High Yield Bond Spreads (Risk Premium).**\n\n**Current Status:** {credit_val:.1f}σ -> {credit_interp}."""

# System HUD
with st.container(border=True):
    # Use ss['color'] to determine the border/text color of the header
    st.markdown(f"### <span style='color:{ss['color']}'>🛡️ System Status: {ss['level']}</span>", unsafe_allow_html=True)

    # Parse Black Swan probability
    prob_str = black_swan_status['probability']
    if prob_str.startswith("High"):
        prob_val = 90
    elif prob_str == "Moderate":
        prob_val = 20
    elif prob_str == "Low":
        prob_val = 2
    elif prob_str == "None":
        prob_val = 0
    else:
        try:
            prob_val = float(prob_str.rstrip('%')) if prob_str.endswith('%') else float(prob_str)
        except ValueError:
            prob_val = 0

    # Two rows of two columns for mobile-friendliness
    row1 = st.columns(2)
    with row1[0]:
        stress_color = "green" if last_turb < 180 else "orange" if last_turb < 370 else "red"
        st.markdown(f"<div style='text-align: center;'><strong>System Stress</strong><br><span style='font-size: 2em; color: {stress_color}; font-weight: bold;'>{int(last_turb)}</span></div>", unsafe_allow_html=True)
        st.caption("**Scale:** Turbulence score ranges from 0-1000, where higher values indicate more unusual market behavior.")
    with row1[1]:
        fear_color = "red" if vix_val > 20 else "green"
        delta_symbol = "▲" if vix_val > 20 else "▼"
        st.markdown(f"<div style='display: flex; justify-content: space-between; align-items: center;'><span><strong>Fear Gauge (VIX)</strong></span><span style='font-size: 1.2em;'>{vix_val:.1f}</span><span style='font-size: 1.2em; color: {fear_color};'>{delta_symbol}</span></div>", unsafe_allow_html=True)
        st.caption("VIX North Star: Fear index. >20 indicates significant fear, <15 suggests complacency.")

    row2 = st.columns(2)
    with row2[0]:
        dix_val = whale_tracker['dix']
        smart_color = "green" if dix_val > 45 else "red" if dix_val < 40 else "gray"
        delta_symbol = "▲" if dix_val > 45 else "▼" if dix_val < 40 else ""
        st.markdown(f"<div style='display: flex; justify-content: space-between; align-items: center;'><span><strong>Smart Money (DIX)</strong></span><span style='font-size: 1.2em;'>{dix_val:.1f}%</span><span style='font-size: 1.2em; color: {smart_color};'>{delta_symbol}</span></div>", unsafe_allow_html=True)
        st.caption("DIX North Star: Institutional positioning. >45% signals accumulation, <40% distribution.")
    with row2[1]:
        if prob_val > 15:
            bs_value = "High"
            bs_color = "red"
        elif prob_val > 5:
            bs_value = "Moderate"
            bs_color = "orange"
        else:
            bs_value = "Low"
            bs_color = "green"
        st.markdown(f"<div style='display: flex; justify-content: space-between; align-items: center;'><span><strong>Black Swan Risk</strong></span><span style='font-size: 1.2em; color: {bs_color}; font-weight: bold;'>{bs_value}</span><span style='font-size: 0.8em;'>{prob_str}</span></div>", unsafe_allow_html=True)
        st.caption("Probability of a major market disruption based on alignment of key signals.")

    # Divergence Alert
    divergence_alert = (last_turb > 180) and (vix_val < 20)
    if divergence_alert:
        st.warning("⚠️ **Divergence Alert**: Turbulence > 180 & VIX < 20. Hidden stress detected - early warning active.")
    else:
        st.info("✅ No Divergence Alert")

tab1, tab2, tab3, tab4 = st.tabs(["🔬 Mechanics", "🏛️ Macro", "🏦 Institutional", "🎯 Strategy"])

with tab1:
    m_cols = st.columns(4)
    m_cols[0].metric("Stress Index", int(last_turb), help=turb_help)
    m_cols[1].metric("Herd Behavior", f"{last_hurst:.2f}", help=hurst_help)
    m_cols[2].metric("Market Liquidity", f"{last_amihud:.1f}σ", help=liq_help)
    m_cols[3].metric("Contagion Risk", f"{last_abs:.0%}", help=abs_help)
    if not spy_p.empty:
        vpoc = vec_engine.get_vpoc_level(price) if vec_engine else None
        fig_main = charts.plot_divergence_chart(spy_p, curr_turb, futures_data=futures_df, vpoc_level=vpoc)
        st.plotly_chart(fig_main, use_container_width=True)

    # Sector Stress Gauges
    st.markdown("### Sector Stress Gauges")
    gauge_cols = st.columns(3)
    with gauge_cols[0]:
        broad_val = int(last_turb)
        color = "green" if broad_val < 100 else "orange" if broad_val <= 150 else "red"
        st.markdown(f"<div style='text-align: center;'><strong>Broad Market Stress</strong><br><span style='font-size: 2em; color: {color};'>{broad_val}</span></div>", unsafe_allow_html=True)
    with gauge_cols[1]:
        ai_val = int(last_ai_turb)
        color = "green" if ai_val < 100 else "orange" if ai_val <= 150 else "red"
        st.markdown(f"<div style='text-align: center;'><strong>AI Sector Stress</strong><br><span style='font-size: 2em; color: {color};'>{ai_val}</span></div>", unsafe_allow_html=True)
    with gauge_cols[2]:
        crypto_val = int(last_crypto_turb)
        color = "green" if crypto_val < 100 else "orange" if crypto_val <= 150 else "red"
        st.markdown(f"<div style='text-align: center;'><strong>Crypto Sector Stress</strong><br><span style='font-size: 2em; color: {color};'>{crypto_val}</span></div>", unsafe_allow_html=True)

    # Narratives
    st.markdown("#### Narratives")
    # AI Narrative
    ai_tier = "LOW" if ai_ratio <= 1.2 else "MEDIUM" if ai_ratio <= 1.5 else "HIGH"
    icon = "👍" if ai_tier == "LOW" else "😐" if ai_tier == "MEDIUM" else "⚠️"

    # Check for Hidden Structural Threat for AI (NVDA)
    hidden_threat_ai = False
    if "NVDA" in curr_close.columns and len(curr_close["NVDA"]) >= 2:
        nvda_price_change = (curr_close["NVDA"].iloc[-1] / curr_close["NVDA"].iloc[-2] - 1) * 100 > 0
        nvda_sent_change = nvidia_sent < 0  # Assuming current sentiment, but to check change, need previous. Placeholder: if negative, assume dropping.
        if nvda_price_change and nvda_sent_change:
            hidden_threat_ai = True

    ai_narrative = f"{icon} AI Bubble Risk: {ai_tier}"
    if hidden_threat_ai:
        ai_narrative += " | ⚠️ Hidden Structural Threat: Price rising on falling sentiment."
    st.markdown(f"**AI Narrative:** {ai_narrative}")

    # Crypto Narrative
    if len(curr_crypto_turb) >= 2:
        crypto_pct_change = (curr_crypto_turb.iloc[-1] / curr_crypto_turb.iloc[-2] - 1) * 100
        broad_pct_change = (curr_turb.iloc[-1] / curr_turb.iloc[-2] - 1) * 100
        if crypto_pct_change > 20 and broad_pct_change <= 0:
            crypto_status = "BEARISH"
        else:
            crypto_status = "NEUTRAL"
    else:
        crypto_status = "N/A"

    # Check for Hidden Structural Threat for Crypto (BTC)
    hidden_threat_crypto = False
    if "BTC-USD" in curr_close.columns and len(curr_close["BTC-USD"]) >= 2:
        btc_price_change = (curr_close["BTC-USD"].iloc[-1] / curr_close["BTC-USD"].iloc[-2] - 1) * 100 > 0
        btc_sent_change = bitcoin_sent < 0
        if btc_price_change and btc_sent_change:
            hidden_threat_crypto = True

    crypto_narrative = f"Crypto Leading Indicator: {crypto_status}"
    if hidden_threat_crypto:
        crypto_narrative += " | ⚠️ Hidden Structural Threat: Price rising on falling sentiment."
    st.markdown(f"**Crypto Narrative:** {crypto_narrative}")

    # Sector Fever Heatmap
    st.markdown("### Sector Fever Heatmap")
    heat_cols = st.columns(3)

    def get_trend_and_percentile(series, current_val):
        if len(series) < 8:
            return "N/A", "N/A"
        recent_avg = series.iloc[-8:-1].mean()
        trend = "Rising" if current_val > recent_avg else "Falling"
        percentile = (series < current_val).sum() / len(series) * 100
        top_pct = 100 - percentile
        rank = f"Top {top_pct:.0f}% Stressed"
        return trend, rank

    with heat_cols[0]:
        trend, rank = get_trend_and_percentile(curr_turb, last_turb)
        emoji = "🔥" if trend == "Rising" else "❄️" if trend == "Falling" else "❓"
        st.markdown(f"**Broad Market Fever Card**<br>Stress: {last_turb:.0f}<br>7-Day Trend: {trend} {emoji}<br>Historical Rank: {rank}", unsafe_allow_html=True)
    with heat_cols[1]:
        trend, rank = get_trend_and_percentile(curr_ai_turb, last_ai_turb)
        emoji = "🔥" if trend == "Rising" else "❄️" if trend == "Falling" else "❓"
        st.markdown(f"**AI Chips Fever Card**<br>Stress: {last_ai_turb:.0f}<br>7-Day Trend: {trend} {emoji}<br>Historical Rank: {rank}", unsafe_allow_html=True)
    with heat_cols[2]:
        trend, rank = get_trend_and_percentile(curr_crypto_turb, last_crypto_turb)
        emoji = "🔥" if trend == "Rising" else "❄️" if trend == "Falling" else "❓"
        st.markdown(f"**Crypto Assets Fever Card**<br>Stress: {last_crypto_turb:.0f}<br>7-Day Trend: {trend} {emoji}<br>Historical Rank: {rank}", unsafe_allow_html=True)

with tab2:
    st.caption("Strategic recommendations based on relative asset flows (20-day trend).")

    if not macro_ratios_df.empty:
        # Pre-map labels and logic
        macro_map = {
            "SPY/TLT": {
                "label": "Stocks vs Bonds",
                "left": "Bonds (Safe)",
                "right": "Stocks (Growth)",
                "flip": False,
                "desc": "When this rises, investors prefer profits over the safety of government debt.",
                "safety_if": "Falling"
            },
            "XLY/XLP": {
                "label": "Wants vs Needs",
                "left": "Needs (XLP)",
                "right": "Wants (XLY)",
                "flip": False,
                "desc": "Compares luxury spending (Discretionary) to essential spending (Staples). If falling, it suggests the average person is feeling the pinch.",
                "safety_if": "Falling"
            },
            "GLD/SPY": {
                "label": "Gold vs Stocks",
                "left": "Gold (Fear)",
                "right": "Stocks (Growth)",
                "flip": True,
                "desc": "When this rises, the 'Smart Money' is buying insurance (Gold) because they don't trust the stock market rally.",
                "safety_if": "Rising"
            },
            "EEM/SPY": {
                "label": "World vs USA",
                "left": "USA (SPY)",
                "right": "World (EEM)",
                "flip": False,
                "desc": "Rising means money is flowing into high-risk global markets. Falling means money is 'Hiding in the US Dollar.'",
                "safety_if": "Falling"
            },
            "CPER/GLD": {
                "label": "Industrial vs Safety",
                "left": "Safety (Gold)",
                "right": "Industrial (CPER)",
                "flip": False,
                "desc": "Copper is used to build things; Gold is used to hide wealth. Rising means the actual global economy is expanding.",
                "safety_if": "Falling"
            }
        }

        safety_count = 0
        for i, row in macro_ratios_df.iterrows():
            pair = row['Pair']
            if pair in macro_map and row['Trend'] == macro_map[pair]['safety_if']:
                safety_count += 1

        if safety_count >= 3:
            st.warning("⚠️ **UNDERLYING CAUTION:** Macro indicators are moving toward defensive positions.")

        cols = st.columns(2)
        for i, (idx, row) in enumerate(macro_ratios_df.iterrows()):
            pair = row['Pair']
            if pair not in macro_map: continue

            m_info = macro_map[pair]
            col_idx = i % 2
            with cols[col_idx]:
                with st.container(border=True):
                    st.markdown(f"**{m_info['label']}**", help=m_info['desc'])

                    # Tug of War Logic
                    z = row.get('Z-Score', 0.0)
                    # Map -2 to +2 Z-score to 0-100 range
                    prog_val = 50 + (z * 25)
                    if m_info['flip']:
                        prog_val = 100 - prog_val
                    prog_val = max(0, min(100, int(prog_val)))

                    # Custom Tug-of-War Bar using Progress
                    t_cols = st.columns([1, 4, 1])
                    t_cols[0].markdown(f"<div style='text-align: right; color: #888; font-size: 0.8rem;'>{m_info['left']}</div>", unsafe_allow_html=True)
                    t_cols[1].progress(prog_val)
                    t_cols[2].markdown(f"<div style='text-align: left; color: #888; font-size: 0.8rem;'>{m_info['right']}</div>", unsafe_allow_html=True)

                    # Winner Logic
                    trend = row['Trend']
                    pair = row['Pair']
                    if trend == "Rising":
                        winner = macro_map[pair]['right'].split(' (')[0]  # e.g., "Stocks"
                        color = "#00C853"
                    else:
                        winner = macro_map[pair]['left'].split(' (')[0]  # e.g., "Bonds"
                        color = "#D32F2F"
                    st.markdown(f"<div style='text-align: center; color: {color}; font-weight: bold;'>Winner: {winner}</div>", unsafe_allow_html=True)

    # The Economic Gravity
    st.markdown("### 🌍 The Economic Gravity")

    # M2 Liquidity Wave
    st.markdown("#### 💧 M2 Liquidity Wave")
    st.info("M2 Money Supply growth vs. SPY price analysis. (Historical M2 data integration pending - currently using static values)")

    # Yield Curve History
    st.markdown("#### 📈 Yield Curve History")
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curr_yield.index, y=curr_yield.values, mode='lines', name='10Y-2Y Spread', line=dict(color='blue')))
    fig.add_hrect(y0=-10, y1=0, fillcolor="red", opacity=0.2, line_width=0, annotation_text="Inversion Zone", annotation_position="top left")
    fig.update_layout(title="Yield Curve History (10Y-2Y Spread)", xaxis_title="Date", yaxis_title="Spread (%)")
    st.plotly_chart(fig)

    # Yield Curve and Credit Stress
    st.markdown("#### 🌍 Macro Truth")
    mac1, mac2 = st.columns(2)
    mac1.metric(f"Yield Curve (10Y-2Y)", f"{last_yield:.2f}%", delta="Inverted" if last_yield < 0 else "Normal", help=yield_help)
    mac2.metric("Credit Stress (HY)", f"{credit_val:.1f}σ", help=credit_help)

    # Dual-Axis Chart: M2 Liquidity vs SPX Price
    st.markdown("#### 💧 M2 Liquidity Wave Dual-Axis")
    if not macro_m2.empty and spy_col in curr_close.columns:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Left axis: M2 Money Supply
        fig.add_trace(
            go.Scatter(x=macro_m2.index, y=macro_m2.values, name="M2 Money Supply", line=dict(color="blue")),
            secondary_y=False
        )

        # Right axis: SPX Price
        fig.add_trace(
            go.Scatter(x=curr_close[spy_col].index, y=curr_close[spy_col].values, name="SPX Price", line=dict(color="green")),
            secondary_y=True
        )

        fig.update_layout(title="M2 Liquidity vs SPX Price: Is the Market Rising on Fuel or Empty Air?", template="plotly_dark")
        fig.update_yaxes(title_text="M2 Money Supply ($)", secondary_y=False)
        fig.update_yaxes(title_text="SPX Price", secondary_y=True)

        st.plotly_chart(fig)
    else:
        st.info("M2 data or SPX data not available for chart.")

with tab3:
    st.subheader("📊 Institutional Sentiment")
    dix_val = whale_tracker['dix']
    if dix_val > 50:
        st.success("🟢 Institutions are BUYING the dip")
    elif dix_val < 40:
        st.error("🔴 Institutions are SELLING into strength")
    else:
        st.info("🟡 Institutions are NEUTRAL")

    if whale_tracker['volume_anomaly']:
        st.error("**🚨 WHALE ACTIVITY DETECTED:** Large institutional block trades (Dark Pool) identified at current levels. Proceed with caution!")

    if not market_status['is_trading']:
        # Show static status for non-trading periods
        status_indicator = "📊 Weekend Mode" if market_status['reason'] == "Weekend" else "📅 Holiday Mode" if market_status['reason'] == "Bank Holiday" else "🌙 After Hours"
        st.caption(f"{status_indicator}: Static institutional structure (Market {market_status['status']})")

        # Detailed impact explanation
        impact_explanation = {
            "real_time_data": "Real-time order flow and volume analysis",
            "institutional_footprints": "Smart Money Concepts (FVG, OB, CHoCH)",
            "volume_analysis": "Volume profile and liquidity metrics"
        }

        impacted_vars_text = ", ".join([impact_explanation.get(var, var) for var in market_status['impacts']])
        st.info(f"**Market {market_status['status']} Analysis:** Institutional footprints remain fixed until next market open. No new institutional activity expected. **Impacted:** {impacted_vars_text}")

        smc1, smc2, smc3 = st.columns(3)

        with smc1:
            st.markdown("**Price Imbalance (FVG)**")
            st.info("Static - Market Closed")
            st.caption("Unfilled gaps where institutions moved too fast for 'fair value' to be established. No new FVGs forming.")

        with smc2:
            st.markdown("**Institutional Floor/Ceiling (OB)**")
            st.info("Static - Market Closed")
            st.caption("Active levels where massive institutional orders were last filled. Levels remain unchanged.")

        with smc3:
            st.markdown("**Change of Character (CHoCH)**")
            st.info("Static - Market Closed")
            st.caption("First sign of trend structural breakdown or reversal. No new CHoCH signals expected.")
    else:
        st.caption("Cross-verifying Quant signals with institutional price patterns (Smart Money Concepts).")
        smc1, smc2, smc3 = st.columns(3)

        with smc1:
            st.markdown("**Price Imbalance (FVG)**")
            st.info(smc_context["fvg"])
            st.caption("Unfilled gaps where institutions moved too fast for 'fair value' to be established.")

        with smc2:
            st.markdown("**Institutional Floor/Ceiling (OB)**")
            st.info(smc_context["ob"])
            st.caption("Active levels where massive institutional orders were last filled.")

        with smc3:
            st.markdown("**Change of Character (CHoCH)**")
            if smc_context["choch_val"] == 1:
                st.success(f"✅ {smc_context['choch']}")
            elif smc_context["choch_val"] == -1:
                st.error(f"🚨 {smc_context['choch']}")
            else:
                st.info(smc_context["choch"])
            st.caption("First sign of trend structural breakdown or reversal.")

    st.markdown("### Institutional Insights")
    if dix_val > 45:
        status = "Accumulating"
        color = "green"
    elif dix_val < 40:
        status = "Distributing"
        color = "red"
    else:
        status = "Neutral"
        color = "gray"
    st.markdown(f"DIX {dix_val:.1f}% - <span style='color:{color}; font-weight:bold;'>{status}</span><br>DIX >50% often signals confidence; <40% may indicate caution. Monitor for volume anomalies to spot big moves.", unsafe_allow_html=True)



with tab4:
    st.header(f"Business Cycle: {cycle_info['phase']}")

    # 2. Horizon View (The Planning Section)
    h1, h2 = st.columns(2)
    with h1:
        st.markdown("**Current Status**")
        st.info(f"**{cycle_info['phase']}**: Active now")
        st.caption(cycle_info['warning'])
    with h2:
        st.markdown("**Early Planning (Next Stage)**")
        st.warning(f"**{cycle_info['next']}**: Prepare for this transition")
        st.success(f"**Rotation Target:** {cycle_info['assets']}")

    # 3. Strategy Playbook (Merged here for clarity)
    st.markdown("---")
    st.subheader("🗺️ Economic Cycle Navigator")
    cols = st.columns(4)
    stages = [
        {"name": "Expansion", "assets": "Growth Stocks, Real Estate, Commodities", "color": "green"},
        {"name": "Peak", "assets": "Defensives, Gold, Cash", "color": "orange"},
        {"name": "Contraction", "assets": "Bonds, Gold, Staples", "color": "red"},
        {"name": "Trough", "assets": "Value Stocks, Small Caps, Emerging Markets", "color": "blue"}
    ]
    
    for i, stage in enumerate(stages):
        is_active = cycle_info['phase'] == stage['name']
        with cols[i]:
            with st.container(border=True):
                if is_active:
                    color_rgba = {"green": "rgba(0,255,0,0.1)", "orange": "rgba(255,165,0,0.1)", "red": "rgba(255,0,0,0.1)", "blue": "rgba(0,0,255,0.1)"}[stage['color']]
                    shadow_color = stage['color']
                    assets_list = stage['assets'].split(', ')
                    bullets = '</li><li>'.join(assets_list)
                    st.markdown(f"<div style='background-color: {color_rgba}; padding: 10px; border-radius: 5px; box-shadow: 0 0 10px {shadow_color}; border: 2px solid {shadow_color};'><h4>🎯 {stage['name']}</h4><p><strong>ACTIVE PHASE</strong></p><ul><li>{bullets}</li></ul></div>", unsafe_allow_html=True)
                else:
                    assets_list = stage['assets'].split(', ')
                    bullets = '</li><li>'.join(assets_list)
                    st.markdown(f"<div style='opacity: 0.4;'><h4>{stage['name']}</h4><p>Target Assets:</p><ul><li>{bullets}</li></ul></div>", unsafe_allow_html=True)

    p2 = st.columns([1])[0]  # Since p1 is replaced, create p2 as full width or adjust
    with p2:
        st.markdown("#### 🎨 Focus Style")
        st.button(f"🎯 {playbook['style']}", use_container_width=True)


