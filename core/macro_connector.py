import pandas as pd
import requests
import datetime
from fredapi import Fred
import config
from finvizfinance.news import News

class MacroConnector:
    def __init__(self):
        self.fred = Fred(api_key=config.FRED_API_KEY)
        self.api_key = config.FRED_API_KEY
        
    def fetch_economic_calendar(self):
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
                url = f"https://api.stlouisfed.org/fred/release/dates?release_id={rid}&realtime_start={today}&include_release_dates_with_no_data=true&sort_order=asc&limit=10&api_key={self.api_key}&file_type=json"
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

    def fetch_yield_curve(self):
        """
        Fetches 10Y-2Y Treasury Yield Spread.
        Series: T10Y2Y
        """
        try:
            return self.fred.get_series('T10Y2Y')
        except Exception as e:
            print(f"FRED Yield Curve Error: {e}")
            return pd.Series()
            
    def fetch_credit_spreads(self):
        """
        Fetches ICE BofA US High Yield Index Option-Adjusted Spread.
        Series: BAMLH0A0HYM2
        Returns Z-Score.
        """
        try:
            spreads = self.fred.get_series('BAMLH0A0HYM2')
            # Calculate Z-Score: (Current - Mean) / StdDev
            # Using 1-year window for Z-Score context or full history?
            # PRD: "High Yield Spreads Z-Score"
            # Let's use a rolling 1-year window or 3-year window for context.
            # Standard: 252 days
            
            roll_mean = spreads.rolling(window=252).mean()
            roll_std = spreads.rolling(window=252).std()
            z_score = (spreads - roll_mean) / roll_std
            
            return z_score
        except Exception as e:
            print(f"FRED Credit Spread Error: {e}")
            return pd.Series()
            
    def fetch_sentiment(self):
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
