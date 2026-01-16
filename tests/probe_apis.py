import yfinance as yf
from fredapi import Fred
import pandas as pd
import datetime
import config

def check_yfinance_earnings():
    print("--- Checking yfinance Earnings ---")
    tickers = ["NVDA", "MSFT"]
    for t in tickers:
        try:
            tick = yf.Ticker(t)
            # Try different methods
            cal = tick.calendar
            print(f"{t} Calendar:\n{cal}")
            
            # get_earnings_dates returns historical and future
            dates = tick.get_earnings_dates(limit=5)
            print(f"{t} Earnings Dates:\n{dates}")
        except Exception as e:
            print(f"{t} Error: {e}")

def check_fred_calendar():
    print("\n--- Checking FRED Calendar ---")
    try:
        fred = Fred(api_key=config.FRED_API_KEY)
        
        # Release IDs: 
        # 10: CPI
        # 50: Employment Situation
        # 53: GDP
        # 323: PCE
        
        release_ids = {
            "CPI": 10,
            "Jobs (NFP)": 50,
            "GDP": 53,
            "PCE": 323
        }
        
        for name, rid in release_ids.items():
            # Fetch dates for this year
            dates = fred.get_release_dates(release_id=rid, limit=5, sort_order='desc')
            print(f"{name} (ID {rid}):\n{dates.head()}")
            
    except Exception as e:
        print(f"FRED Error: {e}")

if __name__ == "__main__":
    check_yfinance_earnings()
    check_fred_calendar()
