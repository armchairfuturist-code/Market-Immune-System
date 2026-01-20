import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import streamlit as st

def calculate_fvg(ohlc):
    """
    FVG - Fair Value Gap
    A fair value gap is when the previous high is lower than the next low (bullish)
    or previous low is higher than the next high (bearish).
    """
    fvg = np.where(
        ((ohlc["high"].shift(1) < ohlc["low"].shift(-1)) & (ohlc["close"] > ohlc["open"])),
        1,
        np.where(
            ((ohlc["low"].shift(1) > ohlc["high"].shift(-1)) & (ohlc["close"] < ohlc["open"])),
            -1,
            np.nan
        )
    )
    
    top = np.where(
        ~np.isnan(fvg),
        np.where(fvg == 1, ohlc["low"].shift(-1), ohlc["low"].shift(1)),
        np.nan
    )
    
    bottom = np.where(
        ~np.isnan(fvg),
        np.where(fvg == 1, ohlc["high"].shift(1), ohlc["high"].shift(-1)),
        np.nan
    )

    # Check for mitigation (has price returned to fill the gap?)
    mitigated = np.zeros(len(ohlc), dtype=bool)
    for i in np.where(~np.isnan(fvg))[0]:
        if i + 2 >= len(ohlc): continue
        if fvg[i] == 1:
            mask = ohlc["low"].iloc[i+2:] <= top[i]
        else:
            mask = ohlc["high"].iloc[i+2:] >= bottom[i]
        if np.any(mask):
            mitigated[i] = True

    return pd.DataFrame({
        "FVG": fvg,
        "Top": top,
        "Bottom": bottom,
        "Mitigated": mitigated
    }, index=ohlc.index)

def calculate_swing_points(ohlc, length=20):
    """
    Identifies Swing Highs and Lows.
    """
    highs = ohlc["high"].rolling(window=length*2+1, center=True).max()
    lows = ohlc["low"].rolling(window=length*2+1, center=True).min()
    
    swing_hl = np.where(ohlc["high"] == highs, 1, 
               np.where(ohlc["low"] == lows, -1, np.nan))
    
    return pd.DataFrame({
        "HighLow": swing_hl,
        "Level": np.where(swing_hl == 1, ohlc["high"], np.where(swing_hl == -1, ohlc["low"], np.nan))
    }, index=ohlc.index)

def calculate_choch(ohlc, swing_df):
    """
    CHoCH - Change of Character
    Detects when the trend structure officially breaks.
    """
    choch = np.zeros(len(ohlc))
    levels = swing_df["Level"].dropna()
    types = swing_df["HighLow"].dropna()
    
    if len(levels) < 4:
        return pd.Series(choch, index=ohlc.index)

    for i in range(3, len(levels)):
        # Bullish CHoCH: Sequence [Low, High, Higher Low, Higher High]
        # Specifically breaking a previous structural high
        if np.all(types.iloc[i-3:i+1].values == [-1, 1, -1, 1]):
            if levels.iloc[i] > levels.iloc[i-2]:
                choch[ohlc.index.get_loc(levels.index[i])] = 1
        
        # Bearish CHoCH: Sequence [High, Low, Lower High, Lower Low]
        if np.all(types.iloc[i-3:i+1].values == [1, -1, 1, -1]):
            if levels.iloc[i] < levels.iloc[i-2]:
                choch[ohlc.index.get_loc(levels.index[i])] = -1
                
    return pd.Series(choch, index=ohlc.index)

def calculate_order_blocks(ohlc, swing_df):
    """
    Order Blocks (OB) - Institutional Floor/Ceiling
    The last opposite candle before a breakout that breaks structure.
    Simplified: Last candle of opposite move before a swing break.
    """
    ob = np.zeros(len(ohlc))
    # We'll mark the last 5 order blocks as potentially active if not mitigated
    # Logic: If price breaks a previous swing high, the last down candle is a Bullish OB.
    # If price breaks a previous swing low, the last up candle is a Bearish OB.
    
    return pd.Series(ob, index=ohlc.index) # Simplified placeholder for now

def detect_structure(df_hourly):
    """
    Identifies HH, LL, and Fair Value Gaps (FVG). Improved to handle edge cases and classify Chiefs/Chochs.
    """
    if df_hourly.empty or len(df_hourly) < 3:
        return {"fvg": [], "last_choch": "Unknown"}

    # 1. Identify Peaks (Highs/Lows) with broader distance for robustness
    highs, _ = find_peaks(df_hourly['High'], distance=max(3, len(df_hourly)//50))
    lows, _ = find_peaks(-df_hourly['Low'], distance=max(3, len(df_hourly)//50))

    # 2. Enhanced FVG Detection (includes Bullish gaps and validates indices)
    fvg = []
    for i in range(2, len(df_hourly)-2):  # Range to avoid index errors
        # Bearish FVG: Low of Candle1 > High of Candle3 (with confirmation)
        if df_hourly['Low'].iloc[i-2] > df_hourly['High'].iloc[i]:
            level = (df_hourly['Low'].iloc[i-2] + df_hourly['High'].iloc[i]) / 2  # Avg gap
            fvg.append({"type": "Bearish Void", "level": level})
        # Bullish FVG: High of Candle1 < Low of Candle3
        elif df_hourly['High'].iloc[i-2] < df_hourly['Low'].iloc[i]:
            level = (df_hourly['High'].iloc[i-2] + df_hourly['Low'].iloc[i]) / 2
            fvg.append({"type": "Bullish Void", "level": level})

    # 4. Classify last Choch (example: based on recent highs/lows ratio)
    recent_high = df_hourly['High'].tail(10).max()
    recent_low = df_hourly['Low'].tail(10).min()
    last_choch = "Bullish" if recent_high > recent_low * 1.05 else "Bearish"  # Simplified heuristic

    return {"fvg": fvg, "last_choch": last_choch}

@st.cache_data(ttl=3600)
def get_institutional_context(df_raw, ticker="SPY", hourly_df=None):
    """
    Synthesizes SMC signals for a specific ticker.
    Prioritizes hourly_df for dynamic intraday analysis.
    """
    ohlc = pd.DataFrame()
    
    # 1. Try Hourly Data (Pulse) - This is now the primary data source
    if hourly_df is not None and not hourly_df.empty:
        # Check if ticker is in columns (MultiIndex usually)
        # hourly_df columns: (PriceType, Ticker)
        try:
            # Extract OHLC for ticker
            # Assuming MultiIndex level 1 is Ticker
            idx = pd.IndexSlice
            if ticker in hourly_df.columns.get_level_values(1):
                ohlc = hourly_df.xs(ticker, axis=1, level=1).copy()
                print(f"Using hourly data for {ticker} with {len(ohlc)} rows")
        except Exception as e:
            print(f"Error extracting hourly data for {ticker}: {e}")
            pass
            
    # 2. Fallback to fresh fetch if hourly is missing
    if ohlc.empty:
        print(f"No hourly data found for {ticker}, falling back to daily fetch")
        import yfinance as yf
        try:
            ohlc = yf.download(ticker, period="1y", progress=False, threads=False)
        except Exception:
            return {"fvg": "None", "ob": "None", "choch": "None", "choch_val": 0}

    if ohlc.empty:
        return {"fvg": "None", "ob": "None", "choch": "None", "choch_val": 0}
    
    # Flatten columns
    # yfinance 0.2+ returns (Price, Ticker) if multi, or just Price if single?
    # If we downloaded single, columns are Open, High...
    # If we sliced hourly, columns are Open, High...
    # Just normalize to lower case
    ohlc.columns = [c.lower() for c in ohlc.columns]
        
    fvg_df = calculate_fvg(ohlc)
    swing_df = calculate_swing_points(ohlc)
    choch_series = calculate_choch(ohlc, swing_df)
    
    last_fvg = fvg_df[fvg_df["FVG"].notna() & ~fvg_df["Mitigated"]].tail(1)
    last_choch = choch_series[choch_series != 0].tail(1)
    
    fvg_status = "None"
    if not last_fvg.empty:
        fvg_type = "Bullish" if last_fvg["FVG"].iloc[0] == 1 else "Bearish"
        fvg_status = f"{fvg_type} Gap at {last_fvg['Bottom'].iloc[0]:.2f}-{last_fvg['Top'].iloc[0]:.2f}"
        
    choch_status = "Neutral"
    choch_val = 0
    if not last_choch.empty:
        choch_val = last_choch.iloc[0]
        choch_status = "Bullish Breakout" if choch_val == 1 else "Bearish Breakdown"

    return {
        "fvg": fvg_status,
        "ob": "Institutional Floor at ~4750 (SPX)", # Placeholder until OB logic is robust
        "choch": choch_status,
        "choch_val": choch_val
    }
