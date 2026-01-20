import pandas as pd
import requests
import datetime
import streamlit as st
from fredapi import Fred
import config
from finvizfinance.news import News

class MacroConnector:
    def __init__(self):
        self.fred = Fred(api_key=config.FRED_API_KEY)
        self.api_key = config.FRED_API_KEY
        
    @st.cache_data(ttl=3600)
    def fetch_economic_calendar(_self):
        """
        Fetches next release dates for key economic indicators using raw FRED API.
        """
        # Key Release IDs
        releases = {
            "CPI": 10,
            "Jobs (NFP)": 50,
            "GDP": 53,
            "PCE": 323,
            "FOMC Projections": 355
        }
        
        calendar_data = []
        today = datetime.date.today().strftime("%Y-%m-%d")
        
        for event, rid in releases.items():
            try:
                url = f"https://api.stlouisfed.org/fred/release/dates?release_id={rid}&realtime_start={today}&include_release_dates_with_no_data=true&sort_order=asc&limit=10&api_key={_self.api_key}&file_type=json"
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    
                    if "release_dates" in data:
                        dates = data["release_dates"]
                        # Find next date >= today
                        for d in dates:
                            r_date_str = d["date"]
                            r_date = datetime.datetime.strptime(r_date_str, "%Y-%m-%d").date()
                            if r_date >= datetime.date.today():
                                calendar_data.append({"Event": event, "Date": r_date})
                                break
            except Exception as e:
                print(f"Error fetching {event}: {e}")
                
        if calendar_data:
            return pd.DataFrame(calendar_data).sort_values("Date")
        return pd.DataFrame(columns=["Event", "Date"])

    @st.cache_data(ttl=3600)
    def fetch_yield_curve(_self):
        """
        Fetches 10Y-2Y Treasury Yield Spread.
        Series: T10Y2Y
        """
        try:
            return _self.fred.get_series('T10Y2Y')
        except Exception as e:
            print(f"FRED Yield Curve Error: {e}")
            return pd.Series()
            
    @st.cache_data(ttl=3600)
    def fetch_credit_spreads(_self):
        """
        Fetches ICE BofA US High Yield Index Option-Adjusted Spread.
        Fallback: Uses HYG/LQD ratio if FRED is down.
        """
        try:
            spreads = _self.fred.get_series('BAMLH0A0HYM2')
            print(f"FRED spreads fetched: {len(spreads)} entries")
            if spreads is None or spreads.empty:
                raise ValueError("FRED Data Empty")

            # Calculate Z-Score
            roll_mean = spreads.rolling(window=252, min_periods=126).mean()
            roll_std = spreads.rolling(window=252, min_periods=126).std()
            z_score = (spreads - roll_mean) / roll_std
            print("FRED Z-score calculated successfully")
            return z_score

        except Exception as e:
            print(f"FRED Credit Spread Error: {e}. Using Proxy (HYG/LQD).")
            try:
                # Fallback: HYG vs LQD
                import yfinance as yf
                proxy_data = yf.download(["HYG", "LQD"], period="2y", progress=False, threads=False)['Close']
                print(f"Proxy data fetched: {len(proxy_data)} entries")
                if not proxy_data.empty:
                    # Ratio: HYG (Junk) / LQD (Quality)
                    # Higher Ratio = Risk On (Low Stress). Lower Ratio = Risk Off (High Stress).
                    ratio = proxy_data["HYG"] / proxy_data["LQD"]

                    # We want Stress metric. So Invert.
                    # Z-Score of Ratio.
                    roll_mean = ratio.rolling(window=60).mean()
                    roll_std = ratio.rolling(window=60).std()
                    z_proxy = (ratio - roll_mean) / roll_std

                    # Invert so Positive Z = High Stress (Low HYG/LQD)
                    print("Proxy Z-score calculated successfully")
                    return -z_proxy
            except Exception as proxy_e:
                print(f"Proxy failed: {proxy_e}")

            print("Returning empty Series")
            return pd.Series()
            
    @st.cache_data(ttl=3600)
    def fetch_sentiment(_self):
        """
        Fetches general market sentiment from FinViz News.
        Uses NLTK Vader.
        """
        try:
            fnews = News()
            news_df = fnews.get_news()
            
            # news_df columns: ['Date', 'Title', 'Source', 'Link']
            # We need to analyze 'Title'.
            
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            import nltk
            
            try:
                nltk.data.find('vader_lexicon')
            except LookupError:
                nltk.download('vader_lexicon', quiet=True)
                
            sia = SentimentIntensityAnalyzer()
            
            # Simple average sentiment of recent headlines
            scores = []
            for title in news_df['Title'].head(20): # Analyze top 20 headlines
                scores.append(sia.polarity_scores(title)['compound'])
                
            if scores:
                avg_score = sum(scores) / len(scores)
                # Scale -1 to 1 -> 0 to 100
                scaled_score = (avg_score + 1) * 50
                return scaled_score
            return 50.0
            
        except Exception as e:
            print(f"FinViz Sentiment Error: {e}")
            return 50.0 # Neutral fallback

    @st.cache_data(ttl=3600)
    def get_polymarket_odds(_self):
        """
        Fetches live odds for 'Will the Fed cut rates?'
        Polymarket API is free and public. Handles market ID search dynamically.
        """
        try:
            # Dynamic search for Fed-related markets (avoids hardcoded IDs)
            search_url = "https://clob.polymarket.com/markets?tags=Fed&active=true"
            response = requests.get(search_url)
            response.raise_for_status()
            markets = response.json().get('data', [])
            if markets:
                # Assume first relevant market; in production, filter by exact title
                market = markets[0]
                market_id = market['id']
                prob = float(market.get('probability', 0)) * 100  # Convert to %
                event = market.get('question', 'Fed Rate Cut')
                return {"event": f"{event} (Estimated)", "probability": f"{prob:.1f}%"}
            else:
                return {"event": "Fed Rate Cut", "probability": "N/A"}  # Fallback
        except requests.RequestException as e:
            print(f"Polymarket API error: {e}")
            return {"event": "Fed Rate Cut", "probability": "Error"}
