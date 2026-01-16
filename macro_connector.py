"""
MacroConnector - External Data Source Integrations
Connects to FRED, market calendars, and sentiment APIs.
"""

import os
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class MacroConnector:
    """
    Connects to official economic data sources to replace proxy calculations.
    
    Integrations:
    - FRED (Federal Reserve Economic Data): Yield curve, credit spreads
    - pandas_market_calendars: NYSE market status
    - finvizfinance + VADER: News sentiment
    
    SECURITY NOTE:
    The FRED API key should be set via environment variable FRED_API_KEY
    or Streamlit secrets, NOT hardcoded in this file.
    """
    
    # FRED Series IDs
    FRED_YIELD_CURVE = "T10Y2Y"           # 10Y-2Y Treasury Spread
    FRED_CREDIT_SPREAD = "BAMLH0A0HYM2"   # ICE BofA High Yield Spread
    
    def __init__(self, fred_api_key: Optional[str] = None):
        """
        Initialize the MacroConnector.
        
        Args:
            fred_api_key: API key for FRED (get free at https://fred.stlouisfed.org/docs/api/api_key.html)
                         If not provided, checks:
                         1. Environment variable FRED_API_KEY
                         2. Streamlit secrets (st.secrets["FRED_API_KEY"])
                         3. Falls back to None (FRED features disabled)
        """
        # Priority: argument > env var > streamlit secrets
        self.fred_key = fred_api_key
        
        if self.fred_key is None:
            self.fred_key = os.environ.get("FRED_API_KEY")
        
        if self.fred_key is None:
            try:
                import streamlit as st
                self.fred_key = st.secrets.get("FRED_API_KEY", None)
            except:
                pass
        
        if self.fred_key is None:
            print("⚠️ FRED API key not configured. Set FRED_API_KEY environment variable.")
            print("   Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        
        self._fred = None
        self._calendar = None
        
    def _get_fred(self):
        """Lazy-load FRED connection."""
        if self._fred is None:
            try:
                from fredapi import Fred
                self._fred = Fred(api_key=self.fred_key)
            except ImportError:
                print("fredapi not installed. Run: pip install fredapi")
                return None
            except Exception as e:
                print(f"FRED init error: {e}")
                return None
        return self._fred
    
    def _get_calendar(self):
        """Lazy-load NYSE calendar."""
        if self._calendar is None:
            try:
                import pandas_market_calendars as mcal
                self._calendar = mcal.get_calendar('NYSE')
            except ImportError:
                print("pandas_market_calendars not installed.")
                return None
        return self._calendar
    
    # =========================================================================
    # FRED DATA METHODS
    # =========================================================================
    
    def get_real_yield_curve(self) -> Dict:
        """
        Fetch the OFFICIAL 10Y-2Y Treasury spread (Series: T10Y2Y).
        
        Includes De-Inversion Detection:
        Paradoxically, the crash usually happens when the curve UN-INVERTS 
        (goes from negative to positive), not when it first tips negative.
        
        Returns:
            Dict with 'value', 'date', 'is_inverted', 'is_deinverting', 'signal'
        """
        result = {
            "value": None,
            "date": None,
            "is_inverted": False,
            "is_deinverting": False,
            "signal": "N/A"
        }
        
        fred = self._get_fred()
        if fred is None:
            return result
            
        try:
            series = fred.get_series(self.FRED_YIELD_CURVE)
            if series is not None and len(series) > 30:
                series = series.dropna()
                
                # Get the last valid observation
                current_value = float(series.iloc[-1])
                obs_date = series.index[-1].date()
                
                result["value"] = current_value
                result["date"] = obs_date
                
                # Check inversion status
                result["is_inverted"] = current_value < 0
                
                # DE-INVERSION DETECTION
                # Check if curve was inverted 30 days ago but is now positive
                thirty_days_ago = obs_date - timedelta(days=30)
                history = series[series.index >= pd.Timestamp(thirty_days_ago)]
                
                if len(history) > 10:
                    # Was the minimum in the recent history negative?
                    min_recent = history.min()
                    
                    # De-inverting: Was negative recently, now positive (or close to it)
                    if min_recent < -0.05 and current_value > -0.05:
                        result["is_deinverting"] = True
                
                # Determine signal
                if result["is_deinverting"]:
                    result["signal"] = "🚨 DE-INVERSION: Curve un-inverting (Historical crash precursor)"
                elif result["is_inverted"]:
                    result["signal"] = "⚠️ INVERTED: Recession signal active"
                else:
                    result["signal"] = "Normal (Positive Slope)"
                    
        except Exception as e:
            print(f"FRED Yield Curve Error: {e}")
            
        return result
    
    def get_credit_stress_index(self) -> Dict:
        """
        Fetch High Yield Spreads (BAMLH0A0HYM2).
        The canary in the coal mine for liquidity crises.
        
        Returns:
            Dict with 'value', 'z_score', 'date', 'signal'
        """
        result = {
            "value": None,
            "z_score": None,
            "date": None,
            "signal": "N/A"
        }
        
        fred = self._get_fred()
        if fred is None:
            return result
            
        try:
            # Fetch 2 years of history for Z-score calculation
            series = fred.get_series(self.FRED_CREDIT_SPREAD)
            if series is None or len(series) < 50:
                return result
                
            series = series.dropna()
            current = series.iloc[-1]
            obs_date = series.index[-1].date()
            
            # Calculate Z-Score against 1-year history
            one_year_ago = obs_date - timedelta(days=365)
            history = series[series.index >= pd.Timestamp(one_year_ago)]
            
            if len(history) > 20:
                mu = history.mean()
                sigma = history.std()
                z_score = (current - mu) / sigma if sigma > 0 else 0.0
            else:
                z_score = 0.0
                
            # Determine signal
            if z_score > 2.0:
                signal = "CRITICAL: Credit Freeze Warning"
            elif z_score > 1.0:
                signal = "ELEVATED: Stress Rising"
            else:
                signal = "NORMAL"
                
            result = {
                "value": float(current),
                "z_score": float(z_score),
                "date": obs_date,
                "signal": signal
            }
            
        except Exception as e:
            print(f"FRED Credit Spread Error: {e}")
            
        return result
    
    # =========================================================================
    # QUIVER QUANTITATIVE (SMART MONEY)
    # =========================================================================
    
    def get_smart_money_senate(self) -> list:
        """
        Fetch Senate trading data to find 'Insider' accumulation.
        Returns top 5 net-bought tickers by member of Congress.
        """
        top_picks = []
        try:
            import quiverquant
            
            # Get Key
            token = os.environ.get("QUIVER_API_KEY")
            if not token:
                try:
                    import streamlit as st
                    token = st.secrets.get("QUIVER_API_KEY")
                except:
                    pass
            
            if not token:
                return [{"ticker": "No API Key", "desc": "Set QUIVER_API_KEY"}]
                
            qv = quiverquant.quiver(token)
            
            # Fetch last 30 days
            # Note: quiver.senate_trading() returns a DataFrame of trades
            df = qv.senate_trading()
            
            if df.empty:
                return []
                
            # Filter for recent trades (last 60 days)
            df['Date'] = pd.to_datetime(df['Date'])
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=60)
            recent = df[df['Date'] > cutoff]
            
            if recent.empty:
                return [{"ticker": "None", "desc": "No recent trades"}]
            
            # Parse 'Amount' ranges to estimate value
            # Ranges are like "$1,001 - $15,000"
            def parse_amount(amt_str):
                try:
                    clean = amt_str.replace("$", "").replace(",", "")
                    if "-" in clean:
                        low, high = clean.split("-")
                        return (float(low) + float(high)) / 2
                    else:
                        return float(clean)
                except:
                    return 0.0
            
            recent['Value'] = recent['Amount'].apply(parse_amount)
            
            # Purchases positive, Sales negative
            recent['NetValue'] = recent.apply(
                lambda x: x['Value'] if "Purchase" in x['Transaction'] else -x['Value'], axis=1
            )
            
            # Group by Ticker
            net_flows = recent.groupby('Ticker')['NetValue'].sum().sort_values(ascending=False)
            
            # Top 5
            top_5 = net_flows.head(5)
            
            for ticker, flow in top_5.items():
                if flow > 0:
                    top_picks.append({
                        "ticker": ticker,
                        "desc": f"Net Bought: ${flow/1000:.0f}k (Senate)"
                    })
                    
        except Exception as e:
            print(f"Quiver Error: {e}")
            top_picks.append({"ticker": "Error", "desc": str(e)})
            
        return top_picks

    
    def get_market_calendar_status(self) -> Dict:
        """
        Check if the NYSE is currently open.
        
        Returns:
            Dict with 'is_open', 'current_session', 'next_open'
        """
        result = {
            "is_open": False,
            "is_trading_day": False,
            "next_open": None,
            "message": "Unknown"
        }
        
        cal = self._get_calendar()
        if cal is None:
            return result
            
        try:
            now = datetime.now()
            today = now.date()
            
            # Get schedule for current year
            start = today - timedelta(days=7)
            end = today + timedelta(days=30)
            schedule = cal.schedule(start_date=start, end_date=end)
            
            # Check if today is a trading day
            today_ts = pd.Timestamp(today)
            if today_ts in schedule.index:
                result["is_trading_day"] = True
                
                # Check if market is currently open
                market_open = schedule.loc[today_ts, 'market_open']
                market_close = schedule.loc[today_ts, 'market_close']
                
                # Use timezone-aware now for comparison
                now_utc = pd.Timestamp.now('UTC')
                
                if market_open <= now_utc <= market_close:
                    result["is_open"] = True
                    result["message"] = "Market OPEN"
                elif now_utc < market_open:
                    result["message"] = f"Pre-Market (Opens {market_open.strftime('%H:%M')} UTC)"
                else:
                    result["message"] = "After-Hours"
            else:
                result["message"] = "Market Closed (Holiday/Weekend)"
                
            # Find next trading day
            future = schedule[schedule.index > today_ts]
            if len(future) > 0:
                next_date = future.index[0].date()
                result["next_open"] = next_date
                
        except Exception as e:
            print(f"Calendar Error: {e}")
            result["message"] = f"Error: {e}"
            
        return result
    
    def is_trading_day(self, check_date: date = None) -> bool:
        """
        Check if a specific date is a trading day.
        
        Args:
            check_date: Date to check (default: today)
            
        Returns:
            True if trading day, False otherwise
        """
        cal = self._get_calendar()
        if cal is None:
            # Fallback: Simple weekday check
            d = check_date or date.today()
            return d.weekday() < 5
            
        try:
            d = check_date or date.today()
            schedule = cal.schedule(start_date=d, end_date=d)
            return len(schedule) > 0
        except:
            d = check_date or date.today()
            return d.weekday() < 5
    
    def get_next_trading_day(self, after_date: date = None) -> Optional[date]:
        """
        Get the next trading day after the given date.
        
        Args:
            after_date: Date to start from (default: today)
            
        Returns:
            Next trading day as date object
        """
        cal = self._get_calendar()
        d = after_date or date.today()
        
        if cal is None:
            # Fallback: Skip weekends only
            next_d = d + timedelta(days=1)
            while next_d.weekday() >= 5:
                next_d += timedelta(days=1)
            return next_d
            
        try:
            start = d + timedelta(days=1)
            end = d + timedelta(days=14)
            schedule = cal.schedule(start_date=start, end_date=end)
            
            if len(schedule) > 0:
                return schedule.index[0].date()
        except:
            pass
            
        # Fallback
        next_d = d + timedelta(days=1)
        while next_d.weekday() >= 5:
            next_d += timedelta(days=1)
        return next_d
    
    # =========================================================================
    # SENTIMENT METHODS
    # =========================================================================
    
    def get_sentiment_score(self, ticker: str = "SPY") -> Dict:
        """
        Fetch news headlines and calculate VADER sentiment score.
        
        Args:
            ticker: Stock ticker to analyze
            
        Returns:
            Dict with 'score' (0-100), 'label', 'headlines_analyzed'
        """
        result = {
            "score": 50,  # Neutral default
            "label": "Neutral",
            "headlines_analyzed": 0,
            "signal": None
        }
        
        try:
            from finvizfinance.quote import finvizfinance
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            
            # Fetch news
            stock = finvizfinance(ticker)
            news_df = stock.ticker_news()
            
            if news_df is None or len(news_df) == 0:
                return result
                
            # Initialize VADER
            sia = SentimentIntensityAnalyzer()
            
            # Analyze headlines
            scores = []
            for title in news_df['Title'].head(20):  # Last 20 headlines
                sentiment = sia.polarity_scores(str(title))
                # VADER compound score is -1 to 1
                scores.append(sentiment['compound'])
                
            if scores:
                # Convert to 0-100 scale (50 = neutral)
                avg_score = sum(scores) / len(scores)
                normalized = (avg_score + 1) * 50  # -1->0, 0->50, 1->100
                
                # Determine label
                if normalized > 70:
                    label = "Euphoria"
                elif normalized > 55:
                    label = "Bullish"
                elif normalized < 30:
                    label = "Panic"
                elif normalized < 45:
                    label = "Bearish"
                else:
                    label = "Neutral"
                    
                result = {
                    "score": round(normalized, 1),
                    "label": label,
                    "headlines_analyzed": len(scores),
                    "signal": None
                }
                
        except ImportError as e:
            # Missing dependencies
            print(f"Sentiment dependencies missing: {e}")
            result["label"] = "Dependencies Missing"
        except AttributeError as e:
            # finvizfinance HTML structure changed (common issue with scrapers)
            print(f"finvizfinance scraping error (HTML structure may have changed): {e}")
            result["label"] = "Scraping Error"
        except KeyError as e:
            # Expected column missing from scraped data
            print(f"finvizfinance data format changed: {e}")
            result["label"] = "Data Format Error"
        except ConnectionError as e:
            # Network issues
            print(f"Network error fetching sentiment: {e}")
            result["label"] = "Network Error"
        except Exception as e:
            # Generic fallback - don't crash the dashboard
            print(f"Sentiment Error for {ticker}: {e}")
            result["label"] = "Error"
            
        return result
    
    def get_euphoria_signal(self, ticker: str = "SPY", turbulence: float = 0) -> Optional[str]:
        """
        Check for the "Top Signal" - High Sentiment + High Turbulence.
        
        Args:
            ticker: Stock to check sentiment for
            turbulence: Current turbulence score (0-100 scale)
            
        Returns:
            Warning string if euphoria detected, None otherwise
        """
        sentiment = self.get_sentiment_score(ticker)
        
        # Top Signal: Euphoria (>80) AND Elevated Turbulence (>18)
        if sentiment["score"] > 80 and turbulence > 18:
            return "🚨 EUPHORIA PEAK: High sentiment + rising stress = potential top"
        elif sentiment["score"] > 70 and turbulence > 25:
            return "⚠️ CAUTION: Bullish sentiment diverging from structure"
            
        return None


# Singleton for caching
_macro_connector_instance = None

def get_macro_connector(fred_api_key: Optional[str] = None) -> MacroConnector:
    """Get or create the MacroConnector singleton."""
    global _macro_connector_instance
    if _macro_connector_instance is None:
        _macro_connector_instance = MacroConnector(fred_api_key)
    return _macro_connector_instance
