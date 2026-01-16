"""
MacroConnector - External Data Source Integrations
"""
import os
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, Dict
import warnings
warnings.filterwarnings('ignore')

class MacroConnector:
    # FRED Series IDs
    FRED_YIELD_CURVE = "T10Y2Y"
    FRED_CREDIT_SPREAD = "BAMLH0A0HYM2"
    
    def __init__(self, fred_api_key: Optional[str] = None):
        self.fred_key = fred_api_key or os.environ.get("FRED_API_KEY")
        # Fallback public demo key (may be rate limited)
        if not self.fred_key: self.fred_key = "fc2e25e796936565703717197b34efa8"
        self._fred = None
        self._calendar = None

    def _get_fred(self):
        if self._fred is None:
            try:
                from fredapi import Fred
                self._fred = Fred(api_key=self.fred_key)
            except: return None
        return self._fred

    def get_real_yield_curve(self) -> Dict:
        res = {"value": None, "signal": "N/A"}
        fred = self._get_fred()
        if not fred: return res
        try:
            s = fred.get_series(self.FRED_YIELD_CURVE).dropna()
            if len(s) > 30:
                cur = float(s.iloc[-1])
                res["value"] = cur
                res["date"] = s.index[-1].date()
                
                # De-inversion logic
                hist = s.iloc[-30:]
                deinv = hist.min() < -0.05 and cur > -0.05
                
                if deinv: res["signal"] = "🚨 DE-INVERSION"
                elif cur < 0: res["signal"] = "⚠️ INVERTED"
                else: res["signal"] = "Normal"
        except: pass
        return res

    def get_credit_stress_index(self) -> Dict:
        res = {"z_score": 0.0, "signal": "N/A"}
        fred = self._get_fred()
        if not fred: return res
        try:
            s = fred.get_series(self.FRED_CREDIT_SPREAD).dropna()
            if len(s) > 50:
                cur = s.iloc[-1]
                hist = s.iloc[-365:]
                z = (cur - hist.mean()) / hist.std()
                res.update({"value": cur, "z_score": z, "date": s.index[-1].date()})
                if z > 2.0: res["signal"] = "CRITICAL: Freeze"
                elif z > 1.0: res["signal"] = "ELEVATED"
                else: res["signal"] = "NORMAL"
        except: pass
        return res

    def get_sentiment_score(self, ticker: str = "SPY") -> Dict:
        res = {"score": 50, "label": "Neutral", "headlines_analyzed": 0}
        try:
            from finvizfinance.quote import finvizfinance
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            import nltk
            try: nltk.data.find('sentiment/vader_lexicon.zip')
            except: nltk.download('vader_lexicon', quiet=True)
            
            news = finvizfinance(ticker).ticker_news()
            if not news.empty:
                sia = SentimentIntensityAnalyzer()
                scores = [sia.polarity_scores(t)['compound'] for t in news['Title'].head(20)]
                norm = (sum(scores)/len(scores) + 1) * 50
                res.update({"score": round(norm), "headlines_analyzed": len(scores)})
                res["label"] = "Euphoria" if norm > 70 else "Panic" if norm < 30 else "Neutral"
        except: pass
        return res

_macro = None
def get_macro_connector(k=None):
    global _macro
    if not _macro: _macro = MacroConnector(k)
    return _macro