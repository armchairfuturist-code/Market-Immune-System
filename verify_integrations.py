"""
Verification script for advanced data integrations.
Tests FRED connectivity, market calendar, and sentiment API.
"""

from datetime import datetime, date

def test_fred():
    print("\n>>> 1. TESTING FRED API")
    try:
        from macro_connector import get_macro_connector
        macro = get_macro_connector()
        
        # Yield Curve
        yield_data = macro.get_real_yield_curve()
        if yield_data[0] is not None:
            print(f"  ✅ Yield Curve (T10Y2Y): {yield_data[0]:+.2f}% (as of {yield_data[1]})")
        else:
            print("  ⚠️ Yield Curve: No data returned")
            
        # Credit Stress
        credit = macro.get_credit_stress_index()
        if credit.get("value") is not None:
            print(f"  ✅ Credit Spread: {credit['value']:.2f}%, Z-Score: {credit['z_score']:+.1f}σ")
            print(f"     Signal: {credit['signal']}")
        else:
            print("  ⚠️ Credit Stress: No data returned")
            
    except Exception as e:
        print(f"  ❌ FRED Error: {e}")

def test_calendar():
    print("\n>>> 2. TESTING MARKET CALENDAR")
    try:
        from macro_connector import get_macro_connector
        macro = get_macro_connector()
        
        # Market Status
        status = macro.get_market_calendar_status()
        print(f"  ✅ Market Status: {status.get('message')}")
        print(f"     Is Trading Day: {status.get('is_trading_day')}")
        print(f"     Next Open: {status.get('next_open')}")
        
        # Trading Day Check
        today = date.today()
        is_trading = macro.is_trading_day(today)
        print(f"  ✅ Is {today} a trading day? {is_trading}")
        
        # Next Trading Day
        next_day = macro.get_next_trading_day(today)
        print(f"  ✅ Next trading day after {today}: {next_day}")
        
    except Exception as e:
        print(f"  ❌ Calendar Error: {e}")

def test_sentiment():
    print("\n>>> 3. TESTING SENTIMENT (VADER)")
    try:
        from macro_connector import get_macro_connector
        macro = get_macro_connector()
        
        sentiment = macro.get_sentiment_score("SPY")
        print(f"  ✅ SPY Sentiment: {sentiment['score']:.1f}/100 ({sentiment['label']})")
        print(f"     Headlines analyzed: {sentiment['headlines_analyzed']}")
        
    except Exception as e:
        print(f"  ❌ Sentiment Error: {e}")

def test_dashboard_load():
    print("\n>>> 4. TESTING DASHBOARD DATA LOAD")
    try:
        from market_immune_system import MarketImmuneSystem
        
        mis = MarketImmuneSystem(fetch_days=30)
        r, p, v = mis.fetch_data()
        
        if not p.empty:
            print(f"  ✅ Data loaded: {len(p)} rows, {len(p.columns)} assets")
            print(f"     Date range: {p.index[0].date()} to {p.index[-1].date()}")
            print(f"     Today: {date.today()}")
        else:
            print("  ❌ No data returned")
            
    except Exception as e:
        print(f"  ❌ Dashboard Error: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("ADVANCED INTEGRATIONS VERIFICATION")
    print("=" * 50)
    
    test_fred()
    test_calendar()
    test_sentiment()
    test_dashboard_load()
    
    print("\n" + "=" * 50)
    print("VERIFICATION COMPLETE")
    print("=" * 50)
