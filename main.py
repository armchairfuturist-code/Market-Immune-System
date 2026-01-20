import streamlit as st
import pandas as pd
import datetime
import config
from core import data_loader, math_engine, macro_connector, cycle_engine, report_generator, cycle_playbook, smc_engine
from ui import charts
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
    """Check if date is a US bank holiday"""
    # Common US bank holidays - simplified for demo
    # In production, you'd use a proper holiday calendar
    month_day = (check_date.month, check_date.day)
    holidays = [
        (1, 1),   # New Year's Day
        (1, 20),  # Martin Luther King Day (3rd Monday in January)
        (7, 4),   # Independence Day
        (12, 25), # Christmas
        (12, 24), # Christmas Eve (half-day, but often closed)
        (11, 27), # Thanksgiving (4th Thursday)
        (9, 2),   # Labor Day (1st Monday)
        (2, 17),  # Presidents' Day (3rd Monday)
        (5, 26),  # Memorial Day (last Monday)
    ]
    return month_day in holidays

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
    econ_calendar = macro.fetch_economic_calendar()
    return yield_curve, credit_spreads, sentiment, econ_calendar

st.sidebar.markdown("---")
st.sidebar.info("v3.0 | Zero-Trust Engine")

# 4. Data Loading & State Management
if 'data' not in st.session_state or st.sidebar.button("Forced Reload"):
    st.cache_data.clear()
    with st.spinner("Initializing Zero-Trust Data Engine (The 99)..."):
        market_close, earnings_df, futures_df, hourly_df = get_market_data(start_date, use_openbb=use_openbb)
        macro_yield, macro_credit, macro_sentiment, econ_df = get_macro_data()
        
        st.session_state.data = {
            'market_close': market_close,
            'earnings_df': earnings_df,
            'futures_df': futures_df,
            'hourly_df': hourly_df,
            'macro_yield': macro_yield,
            'macro_credit': macro_credit,
            'macro_sentiment': macro_sentiment,
            'econ_df': econ_df
        }
else:
    market_close = st.session_state.data['market_close']
    earnings_df = st.session_state.data['earnings_df']
    futures_df = st.session_state.data['futures_df']
    hourly_df = st.session_state.data.get('hourly_df', pd.DataFrame()) # Handle legacy state
    macro_yield = st.session_state.data['macro_yield']
    macro_credit = st.session_state.data['macro_credit']
    macro_sentiment = st.session_state.data['macro_sentiment']
    econ_df = st.session_state.data['econ_df']

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
st.sidebar.markdown("---")
st.sidebar.markdown("### 💱 Currency Strategy (USD/EUR)")
with st.sidebar.container(border=True):
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

# 6. SMC Synthesis & HUD Alert Upgrade
regime = math_engine.get_market_regime(last_turb, last_abs, trend)

# Upgrade logic: If Turbulence > 180 AND CHoCH == Bearish (-1)
hud_alert_upgrade = False
if last_turb > 180 and smc_context["choch_val"] == -1:
    hud_alert_upgrade = True

crypto_stress_signal = (last_crypto_z > 2.0) and spy_flat
super_signal = math_engine.generate_super_signal(last_amihud, last_hurst, spy_p)
# 6. Report & Context Generation
vix_val = curr_close["^VIX"].iloc[-1] if "^VIX" in curr_close.columns else 0.0
is_divergence = (last_turb > 180) and (trend == "UP")

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

# 7. UI Layout

# Tactical Stance Banner (The "Action" Banner)
st.markdown("### 🎯 Tactical Stance")
stance_msg = f"TACTICAL STANCE: {status_report['warning_level']}. {playbook['layman_strategy']}"
if hud_alert_upgrade:
    st.error(f"**⚠️ INSTITUTIONAL DISTRIBUTION DETECTED. Multi-factor structural breakdown in progress.**")
elif status_report['badge_color'] == 'green':
    st.success(f"**{stance_msg}**")
elif status_report['badge_color'] == 'yellow':
    st.warning(f"**{stance_msg}**")
else:
    st.error(f"**{stance_msg}**")

st.sidebar.markdown(f"**Data Horizon:** {last_date.date()} ({day_name})")
st.sidebar.caption("**System Horizon vs Data Horizon:**")
st.sidebar.caption("• **System Horizon (7-14 Days):** The forward-looking timeframe the system uses to detect structural breaks and regime changes. It's the 'lookahead' window for identifying emerging risks.")
st.sidebar.caption("• **Data Horizon (Today):** The historical data point being analyzed. This is the 'lookback' anchor - the most recent market close used for current calculations.")

# Display Report
with st.container(border=True):
    st.subheader(f"🛡️ Market Stress Index: {status_report['warning_level']}")
    
    # Executive Summary (Consolidated Narrative)
    st.markdown("#### 📝 Executive Summary")
    st.markdown(
        f"""<div style="font-size: 1.15rem; font-style: italic; color: #DDDDDD; line-height: 1.6; margin-bottom: 15px;">
        {status_report['summary_narrative']}
        </div>""", 
        unsafe_allow_html=True
    )

# Institutional Footprints (SMC)
with st.expander("🏦 Institutional Footprints", expanded=False):
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

with st.expander("⚡ Market Mechanics (The 'Engine' Room)", expanded=False):
    st.markdown("### 🔬 System Health Diagnostics")
    c1, c2, c3 = st.columns(3)
    
    # Hurst -> Herd Behavior
    herd_status = "Random/Healthy"
    if last_hurst > 0.75: herd_status = "⚠️ CROWDED (Bubble Risk)"
    elif last_hurst < 0.45: herd_status = "Choppy/Uncertain"
    
    c1.metric("Herd Behavior (Hurst)", f"{last_hurst:.2f}", herd_status)
    c1.caption("Measures if everyone is chasing the same trade. High = Brittle.")
    
    # Amihud -> Market Fluidity
    fluidity = "Liquid/Healthy"
    if last_amihud > 2.0: fluidity = "⚠️ DRYING UP (Crash Risk)"
    
    c2.metric("Market Fluidity", f"{last_amihud:.1f}σ", fluidity)
    c2.caption("Size of the exit door. If 'Drying Up', small sales cause big drops.")
    
    # Turbulence -> System Fever
    fever = "Normal"
    if last_turb > 370: fever = "CRITICAL"
    elif last_turb > 180: fever = "ELEVATED"
    
    c3.metric("System Fever (Stress)", f"{last_turb:.0f}/1000", fever)
    c3.caption("Measures 'weirdness'. High fever precedes structural breaks.")

# Institutional Macro Ratios
with st.expander("🏛️ Institutional Macro Ratios", expanded=False):
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
                    if trend == "Rising":
                        st.markdown(f"<div style='text-align: center; color: #00C853; font-weight: bold;'>Winner: GROWTH</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='text-align: center; color: #FFAB00; font-weight: bold;'>Winner: SAFETY</div>", unsafe_allow_html=True)

st.markdown(f"### 📘 Playbook: {playbook['title']}")
with st.container(border=True):
    cp1, cp2, cp3 = st.columns([2, 1, 1])
    
    with cp1:
        st.markdown("#### 🧠 The Layman's Strategy")
        st.info(playbook['layman_strategy'])
        st.caption(f"**Context:** {playbook['context']}")
        
    with cp2:
        st.markdown("#### 🎨 Style Rotation")
        st.markdown(f"**Focus Style:** {playbook['style']}")
        
    with cp3:
        st.markdown("#### 🔄 Capital Rotation")
        if playbook['capital_rotation']['Buy']:
            st.success(f"**BUY:** {', '.join(playbook['capital_rotation']['Buy'])}")
        if playbook['capital_rotation']['Sell/Avoid']:
            st.error(f"**AVOID:** {', '.join(playbook['capital_rotation']['Sell/Avoid'])}")

# Metrics Row
turb_delta = f"{market_status['status']} Mode" if not market_status['is_trading'] else "Low Vol" if last_turb < 50 else "Active"

# Dynamic Interpretations
# 1. System Fever (Turbulence)
turb_interp = "Calm/Normal"
if last_turb > 300: turb_interp = "SICK - Market acting abnormal"
turb_help = f"""**Definition:** Measures how 'weird' or unusual today's price moves are compared to the last year.\n\n**Significance:** >300: The market is acting 'sick'. Normal patterns are breaking down.\n\n**Current Status:** {last_turb:.0f} -> {turb_interp}."""

# 2. Contagion Risk
abs_interp = "Resilient (Diverse)"
if last_abs > 0.80: abs_interp = "Highly Fragile (Unified)"
abs_help = f"""**Definition:** The % of assets moving in lockstep.\n\n**Significance:** 94% of stocks are moving in lockstep; if one trips, they all fall.\n\n**Current Status:** {last_abs:.0%}: -> {abs_interp}."""

# 3. Herd Behavior (Hurst)
hurst_interp = "Random Walk (Healthy)"
if last_hurst > 0.70: hurst_interp = "Panic/FOMO - High reversal risk"
hurst_help = f"""**Definition:** Measures how 'persistent' a trend is.\n\n**Significance:** >0.70: Market is panicking or FOMO-ing in one direction. High risk of a reversal.\n\n**Current Status:** {last_hurst:.2f} -> {hurst_interp}."""

# 4. Market Pulse (Amihud)
liq_interp = "Normal Liquidity"
if last_amihud > 2.0: liq_interp = "Holes Found - Big players struggling"
liq_help = f"""**Definition:** How much price moves per dollar traded.\n\n**Significance:** >2.0: Holes found in the market. Big players are struggling to find buyers/sellers.\n\n**Current Status:** {last_amihud:.1f}σ -> {liq_interp}."""

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

# Gauges
st.write("")
gauge_cols = st.columns(2)
with gauge_cols[0]:
    st.plotly_chart(charts.create_gauge_chart(last_turb, "Market Stress Index", 0, 1000, {180: "orange", 370: "red"}), use_container_width=True)
with gauge_cols[1]:
    st.plotly_chart(charts.create_gauge_chart(last_abs * 100, "Contagion Risk (%)", 0, 100, {60: "orange", 80: "red"}), use_container_width=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Trend Quality", f"{last_hurst:.2f}", help=hurst_help, delta_color="inverse")
m2.metric("Exit Door Size", f"{last_amihud:.1f}", help=liq_help, delta_color="inverse")
m3.metric("Sentiment", f"{macro_sentiment:.0f}", help=sent_help)
m4.metric("AI/Mkt Ratio", f"{ai_ratio:.1f}x", help=ai_help)
m5.metric("Crypto/Mkt", f"{crypto_ratio:.1f}x", help=crypto_help)

# Macro Row
st.markdown("#### 🌍 Macro Truth")
mac1, mac2 = st.columns(2)

# Yield Curve Help
yield_interp = "Normal (Growth)"
if last_yield < 0: yield_interp = "Inverted (Recession Warning)"
elif last_yield < 0.2: yield_interp = "Flat (Caution)"
yield_help = f"""**Definition:** The difference between 10-Year and 2-Year Treasury yields.\n\n**Significance:** The most reliable recession predictor in history. Inversion (<0) signals trouble ahead.\n\n**Current Status:** {last_yield:.2f}% -> {yield_interp}."""

# Credit Stress Help
credit_val = macro_credit.iloc[-1] if isinstance(macro_credit, pd.Series) and not macro_credit.empty else 0.0
credit_interp = "Stable"
if credit_val > 1.0: credit_interp = "Stress Rising"
if credit_val > 2.0: credit_interp = "Credit Freeze"
credit_help = f"""**Definition:** High Yield Bond Spreads (Risk Premium).\n\n**Significance:** If lenders demand high interest to lend to risky companies, the credit cycle is breaking.\n\n**Current Status:** {credit_val:.1f}σ -> {credit_interp}."""

mac1.metric(f"Yield Curve (10Y-2Y)", f"{last_yield:.2f}%", delta="Inverted" if last_yield < 0 else "Normal", help=yield_help)
mac2.metric("Credit Stress (HY)", f"{credit_val:.1f}σ", help=credit_help)

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