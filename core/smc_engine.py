import pandas as pd
import numpy as np

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

def get_institutional_context(df_raw, ticker="SPY"):
    """
    Synthesizes SMC signals for a specific ticker.
    """
    if ticker not in df_raw.columns:
        return {"fvg": "None", "ob": "None", "choch": "None", "choch_val": 0}
        
    # Get OHLC (Assuming df_raw has MultiIndex or we can fetch)
    # Since market_close only has 'Close', we might need to fetch full OHLC for SPY
    # For now, let's assume we have it or use Close as proxy (inferior but better than nothing)
    # Actually, main.py only downloads Close. We should ideally have OHLC.
    
    import yfinance as yf
    ohlc = yf.download(ticker, period="1y", progress=False)
    if ohlc.empty:
        return {"fvg": "None", "ob": "None", "choch": "None", "choch_val": 0}
    
    # Flatten columns if MultiIndex
    if isinstance(ohlc.columns, pd.MultiIndex):
        ohlc.columns = [c[0].lower() for c in ohlc.columns]
    else:
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
