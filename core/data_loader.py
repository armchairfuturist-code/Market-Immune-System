import yfinance as yf
import pandas as pd
import datetime
import streamlit as st
from fredapi import Fred
from config import FRED_API_KEY

@st.cache_data(ttl=3600)
def fetch_market_data(assets, period="2y", start_date=None):
    """
    Fetches market data with 'Zero-Trust' fixes.
    1. Monday Fix: Handles 24/7 Crypto vs M-F Stocks.
    2. Today Fix: Appends live intraday data if missing.
    
    Returns:
        df_close: Daily close prices for all assets (730 days baseline)
        df_hourly: Hourly OHLC data for leaders (SPY, QQQ, NVDA, BTC-USD) - last 5 days
    """
    
    # Split assets into Crypto and Stocks for separate handling if needed, 
    # but yfinance can download all at once. The issue is alignment.
    # However, downloading separately gives more control.
    
    crypto_assets = [a for a in assets if "-USD" in a]
    stock_assets = [a for a in assets if "-USD" not in a]
    
    data_frames = []
    
    # Fetch Stocks
    if stock_assets:
        print(f"Fetching {len(stock_assets)} stocks...")
        # auto_adjust=True helps with splits/divs, group_by='ticker' or 'column'
        # threads=False to prevent yfinance internal race condition/KeyError
        if start_date:
            stocks_data = yf.download(stock_assets, start=start_date, progress=False, group_by='column', threads=False)
        else:
            stocks_data = yf.download(stock_assets, period=period, progress=False, group_by='column', threads=False)
        
        # yfinance > 0.2 returns MultiIndex columns [('Adj Close', 'AAPL'), ...] or [('Close', 'AAPL')...]
        # We prefer 'Close' if auto_adjust=True, or 'Adj Close'. 
        # Let's assume standard behavior.
        
        # Check if MultiIndex
        if isinstance(stocks_data.columns, pd.MultiIndex):
            # Extract Close and Volume
            try:
                s_close = stocks_data['Close']
                s_volume = stocks_data['Volume']
            except KeyError:
                # Fallback if 'Close' not found (maybe 'Adj Close')
                s_close = stocks_data['Adj Close'] if 'Adj Close' in stocks_data else stocks_data['Close']
                s_volume = stocks_data['Volume']
        else:
            # Single ticker case or flat index?
            # If single ticker, columns are 'Open', 'High', ...
            s_close = stocks_data['Close'].to_frame(name=stock_assets[0])
            s_volume = stocks_data['Volume'].to_frame(name=stock_assets[0])
            
        data_frames.append((s_close, s_volume))
        
    # Fetch Crypto
    if crypto_assets:
        print(f"Fetching {len(crypto_assets)} crypto assets...")
        if start_date:
            crypto_data = yf.download(crypto_assets, start=start_date, progress=False, group_by='column', threads=False)
        else:
            crypto_data = yf.download(crypto_assets, period=period, progress=False, group_by='column', threads=False)
        
        if isinstance(crypto_data.columns, pd.MultiIndex):
            c_close = crypto_data['Close']
            c_volume = crypto_data['Volume']
        else:
            c_close = crypto_data['Close'].to_frame(name=crypto_assets[0])
            c_volume = crypto_data['Volume'].to_frame(name=crypto_assets[0])
            
        data_frames.append((c_close, c_volume))
    
    if not data_frames:
        return pd.DataFrame(), pd.DataFrame()
        
    # Merge - Monday Fix (Outer Join)
    # We need to merge Closes and Volumes separately
    
    if len(data_frames) > 1:
        full_close = pd.concat([df[0] for df in data_frames], axis=1)
        full_volume = pd.concat([df[1] for df in data_frames], axis=1)
    else:
        full_close = data_frames[0][0]
        full_volume = data_frames[0][1]
        
    # Forward fill missing stock data (weekends/holidays)
    full_close = full_close.ffill()
    full_volume = full_volume.fillna(0) # Volume on weekends for stocks is 0
    
    # Today Fix
    today = datetime.datetime.now().date()
    # Safely get last index date
    if not full_close.empty:
        last_date = full_close.index[-1].date()
        
        if last_date < today:
            print("Applying 'Today' fix...")
            try:
                intraday = yf.download(assets, period="1d", interval="1h", progress=False)
                if not intraday.empty:
                    # Handle MultiIndex for intraday
                    if isinstance(intraday.columns, pd.MultiIndex):
                        i_close = intraday['Close'].iloc[-1]
                        i_volume = intraday['Volume'].iloc[-1]
                    else:
                        i_close = intraday['Close'].iloc[-1]
                        i_volume = intraday['Volume'].iloc[-1]
                        # If scalar (single asset), make series
                        if not isinstance(i_close, pd.Series):
                             i_close = pd.Series({assets[0]: i_close})
                             i_volume = pd.Series({assets[0]: i_volume})

                    # Create new rows
                    new_idx = [pd.Timestamp(today).tz_localize(full_close.index.dtype.tz if hasattr(full_close.index.dtype, 'tz') else None)]
                    
                    new_close_row = pd.DataFrame([i_close.values], columns=i_close.index, index=new_idx)
                    new_volume_row = pd.DataFrame([i_volume.values], columns=i_volume.index, index=new_idx)
                    
                    # Align columns
                    new_close_row = new_close_row.reindex(columns=full_close.columns)
                    new_volume_row = new_volume_row.reindex(columns=full_volume.columns)
                    
                    if not new_close_row.dropna(how='all').empty:
                        full_close = pd.concat([full_close, new_close_row]).ffill()
                        full_volume = pd.concat([full_volume, new_volume_row]).fillna(0)

            except Exception as e:
                print(f"Today fix failed: {e}")
                
    # Fetch Hourly Data for Leaders (SPY, QQQ, NVDA, BTC-USD)
    # Last 5 days of 1-hour data for SMC engine
    hourly_data = pd.DataFrame()
    try:
        leader_assets = ["SPY", "QQQ", "NVDA", "BTC-USD"]
        print("Fetching 1-hour data for leaders (last 5 days)...")
        hourly_data = yf.download(leader_assets, period="5d", interval="1h", progress=False, group_by='column', threads=False)
        
        # Handle MultiIndex if needed (yf > 0.2)
        if isinstance(hourly_data.columns, pd.MultiIndex):
            # Keep the full OHLC structure for SMC engine
            pass
    except Exception as e:
        print(f"Hourly fetch failed: {e}")
            
    return full_close, hourly_data

def fetch_next_earnings(tickers, limit=10):
    """
    Fetches next earnings dates for a list of tickers.
    """
    earnings_data = []
    today = datetime.datetime.now().date()
    
    # Process only top N to avoid timeout
    for t in tickers[:limit]:
        try:
            tick = yf.Ticker(t)
            # Robust fetch
            df = tick.get_earnings_dates(limit=8)
            
            if df is not None and not df.empty:
                # Filter for future dates
                future = df[df.index.date >= today]
                if not future.empty:
                    # Closest future date is the smallest date >= today
                    next_date = future.index.min().date()
                    earnings_data.append({"Ticker": t, "Date": next_date})
        except Exception:
            continue
            
    if earnings_data:
        return pd.DataFrame(earnings_data).sort_values("Date")
    return pd.DataFrame(columns=["Ticker", "Date"])

@st.cache_data(ttl=3600)
def fetch_futures_data(period="1mo"):
    """
    Fetches S&P 500 Futures (ES=F) for trend projection.
    """
    try:
        # Fetch generic future
        f = yf.download("ES=F", period=period, progress=False, threads=False)
        if isinstance(f.columns, pd.MultiIndex):
            return f['Close']['ES=F']
        return f['Close']
    except Exception as e:
        print(f"Futures fetch failed: {e}")
        return pd.Series()

@st.cache_data(ttl=3600)
def fetch_macro_data():
    """
    Fetches macro data from FRED.
    """
    fred = Fred(api_key=FRED_API_KEY)
    
    # Example: 10Y-2Y Yield Spread (T10Y2Y), VIX (VIXCLS - might be on FRED or YF)
    # PRD mentions "Recession Signal: (From FRED Yield Curve)"
    
    try:
        # T10Y2Y is the series ID for 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
        yield_curve = fred.get_series('T10Y2Y')
        return yield_curve
    except Exception as e:
        print(f"Error fetching FRED data: {e}")
        return pd.Series()

if __name__ == "__main__":
    # Quick test
    from config import ASSET_UNIVERSE
    df = fetch_market_data(ASSET_UNIVERSE[:5]) # Test with small subset
    print(df.tail())
    print(df.info())
