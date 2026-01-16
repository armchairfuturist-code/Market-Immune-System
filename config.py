import os

# API Keys
try:
    import streamlit as st
    # Try secrets, then env, then fallback (for now, to keep app running)
    if "FRED_API_KEY" in st.secrets:
        FRED_API_KEY = st.secrets["FRED_API_KEY"]
    else:
        FRED_API_KEY = os.environ.get("FRED_API_KEY", "fc2e25e796936565703717197b34efa8")
except ImportError:
    FRED_API_KEY = os.environ.get("FRED_API_KEY", "fc2e25e796936565703717197b34efa8")

# Asset Universe ("The 99")
# 1. Broad Markets (40%)
BROAD_ASSETS = [
    "SPY", "QQQ", "DIA", "IWM", "VXX", "TLT", "IEF", "SHY", "LQD", "HYG", 
    "BND", "AGG", "GLD", "SLV", "CPER", "USO", "UNG", "DBC", "PALL", "UUP", 
    "FXE", "FXY", "FXB", "CYB", "XLF", "XLE", "XLK", "XLY", "XLI", "XLB", 
    "XLRE", "^VIX", "^TNX", "VTV", "VUG" 
    # Removed ^VIX3M, ^IRX as they often cause YF issues or are indices with no volume
]

# 2. Crypto (10%)
CRYPTO_ASSETS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", 
    "ADA-USD", "AVAX-USD", "DOGE-USD", "DOT-USD", "LINK-USD"
]

# 3. AI & Growth (30%)
GROWTH_ASSETS = [
    "NVDA", "AMD", "TSM", "AVGO", "ARM", "MU", "INTC", "TXN", "LRCX", "AMAT", 
    "VRT", "ANET", "SMCI", "PLTR", "DELL", "HPE", "CSCO", "IBM", "ORCL", "CEG", 
    "VST", "NRG", "CCJ", "URA", "NEE", "SO", "DUK", "TSLA", "PATH", "ISRG", 
    "BOTZ", "ROBO", "ARKK"
]

# 4. Defensive (20%)
DEFENSIVE_ASSETS = [
    "XLV", "JNJ", "PFE", "MRK", "ABBV", "UNH", "LLY", "AMGN", "XLP", "PG", 
    "KO", "PEP", "COST", "WMT", "PM", "XLU", "O", "AMT", "CCI", "PSA", 
    "DLR", "USMV", "SPLV"
]

ASSET_UNIVERSE = BROAD_ASSETS + CRYPTO_ASSETS + GROWTH_ASSETS + DEFENSIVE_ASSETS

# Constants
TURBULENCE_PERCENTILE = 0.99
TURBULENCE_SCORE_SCALE = 1000
TURBULENCE_ANCHOR = 370
ABSORPTION_THRESHOLD = 0.8 # 80%

# Regime Thresholds
REGIME_ABSORPTION_HIGH = 0.85
REGIME_TURBULENCE_HIGH = 180
REGIME_TURBULENCE_CRASH = 370

# Colors (MetaMint Premium)
COLOR_BG = "#0E1117"
COLOR_CARD = "#1E1E1E" # Darker card
COLOR_TEXT = "#FAFAFA"
COLOR_ACCENT_GREEN = "#00C853" # Neon Green updated
COLOR_ACCENT_RED = "#FF1744"
COLOR_ACCENT_AMBER = "#FFAB00"