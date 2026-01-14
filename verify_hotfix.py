import pandas as pd
import yfinance as yf
from datetime import datetime
from market_immune_system import MarketImmuneSystem

def test_fetch_logic():
    print("\n>>> 1. TESTING DATA LATENCY FIX (MarketImmuneSystem)")
    mis = MarketImmuneSystem(fetch_days=60) # Short fetch for speed
    
    print("Fetching data...")
    try:
        # returns, prices, volumes
        r, p, v = mis.fetch_data()
        
        if p.empty:
            print("❌ ERROR: No data returned.")
            return
            
        last_dt = p.index[-1]
        today = datetime.now().date()
        
        print(f"  Last Index: {last_dt}")
        print(f"  Today:      {today}")
        
        # Check alignment
        if last_dt.date() == today:
            print("  ✅ SUCCESS: DataFrame contains today's date.")
        else:
            if today.weekday() >= 5:
                print("  ℹ️ INFO: Today is weekend. Last date mismatch is expected.")
            else:
                # Iterate to check if maybe time zones caused issues
                print("  ⚠️ WARNING: Last date is NOT today.")
                
        # Check if we have multiple rows for today (duplicate check)
        # p.index should be unique
        if p.index.duplicated().any():
             print("  ❌ ERROR: Duplicate indices found!")
        else:
             print("  ✅ Data Index is unique.")
             
    except Exception as e:
        print(f"  ❌ CRASH: {e}")

def test_earnings_logic():
    print("\n>>> 2. TESTING EARNINGS API (Simulation)")
    watchlist = ["NVDA", "MSFT"]
    print(f"  Fetching for: {watchlist}")
    
    found = 0
    for t in watchlist:
        try:
            ticker = yf.Ticker(t)
            # Try the logic we used
            info = ticker.info
            ts = info.get('earningsTimestamp')
            if ts:
                dt = datetime.fromtimestamp(ts).date()
                print(f"  ✅ {t}: Found earnings timestamp -> {dt}")
                found += 1
            else:
                # Try calendar fallback
                cal = ticker.calendar
                if cal is not None and not cal.empty:
                     print(f"  ✅ {t}: Found calendar DF/Dict")
                     found += 1
                else:
                    print(f"  ⚠️ {t}: No earnings data found (might be valid).")
        except Exception as e:
            print(f"  ❌ {t}: Error {e}")
            
    if found > 0:
        print("  ✅ API Connectivity Confirmed.")

if __name__ == "__main__":
    test_fetch_logic()
    test_earnings_logic()
