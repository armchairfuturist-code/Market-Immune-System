# PROJECT: Market Immune System v3.0 (Consolidated Master PRD)

## 1. Executive Summary
**Goal:** Build a single-page, real-time risk dashboard ("The Market Immune System") that acts as an institutional-grade early warning detector for market turbulence.
**Core Philosophy:** Detect structural breaks (fragility) 7-14 days before price capitulation. "Trade the Price, Respect the Structure."
**Target Output:** A local Streamlit web application with a "Premium Dark" aesthetic.

## 2. Technical Stack & Constraints
*   **Language:** Python 3.10+
*   **Frontend:** `streamlit` (layout="wide")
*   **Data Sources (Free/Freemium):**
    *   `yfinance` (Price/Volume).
    *   `fredapi` (Macro Truth: Yield Curve, Credit Spreads).
    *   `finvizfinance` + `nltk` (Sentiment Analysis).
*   **Math Backend:** `numpy`, `pandas`, `scipy`, `PyPortfolioOpt` (Ledoit-Wolf shrinkage).
*   **Visualization:** `plotly.graph_objects`.
*   **Design:** Custom CSS (Dark backgrounds `#0E1117`, Neon Green `#00C853` accents).

---

## 3. Data Architecture (The "Zero-Trust" Engine)

### 3.1 Asset Universe ("The 99")
The covariance matrix must be built on these ~99 liquid assets across 4 strategic pillars to capture cross-asset contagion.
1.  **Broad Markets (40%):** SPY, QQQ, DIA, IWM, VXX, TLT, IEF, SHY, LQD, HYG, BND, AGG, GLD, SLV, CPER, USO, UNG, DBC, PALL, UUP, FXE, FXY, FXB, CYB, XLF, XLE, XLK, XLY, XLI, XLB, XLRE, ^VIX, ^VIX3M, ^TNX, ^IRX, VTV, VUG.
2.  **Crypto (10%):** BTC-USD, ETH-USD, SOL-USD, BNB-USD, XRP-USD, ADA-USD, AVAX-USD, DOGE-USD, DOT-USD, LINK-USD.
3.  **AI & Growth (30%):** NVDA, AMD, TSM, AVGO, ARM, MU, INTC, TXN, LRCX, AMAT, VRT, ANET, SMCI, PLTR, DELL, HPE, CSCO, IBM, ORCL, CEG, VST, NRG, CCJ, URA, NEE, SO, DUK, TSLA, PATH, ISRG, BOTZ, ROBO, ARKK.
4.  **Defensive (20%):** XLV, JNJ, PFE, MRK, ABBV, UNH, LLY, AMGN, XLP, PG, KO, PEP, COST, WMT, PM, XLU, O, AMT, CCI, PSA, DLR, USMV, SPLV.

### 3.2 Resilience Fixes (Mandatory Implementation)
The data loader must implement three critical fixes to prevent "hallucinations" and missing data:
1.  **The "Monday" Fix (Outer Join):** Crypto trades 24/7; Stocks trade M-F. The fetcher must download them separately and merge via `outer` join + `ffill` to preserve weekend crypto volatility.
2.  **The "Today" Fix (Live Append):** `yfinance` history often omits the live intraday candle. The system must fetch a separate `period="1d"` snapshot and append it to the historical dataframe so the dashboard is always current.
3.  **The "Time Travel" Fix:** The Date Slider in the UI must strictly cap `max_value` at the last available data point to prevent selecting future dates (which returns empty dataframes).

---

## 4. The Mathematical Engine (Core Algorithms)

### 4.1 Statistical Turbulence (0-1000 Index)
*   **Algorithm:** **Mahalanobis Distance** ($D_t$) measuring the "alienness" of today's return vector vs. a 365-day historical baseline.
*   **Stability:** Use **Ledoit-Wolf Shrinkage** (`PyPortfolioOpt`) for the covariance matrix.
*   **Scaling:** Calibrate raw $D_t$ to a 0-1000 Index.
    *   **Anchor:** Historic 99th percentile = Score 370.
    *   **Warning:** 180.
    *   **Critical:** 370.

### 4.2 Absorption Ratio (Fragility)
*   **Algorithm:** Principal Component Analysis (PCA).
*   **Formula:** (Variance Explained by Top 20% Eigenvectors) / (Total Variance).
*   **Interpretation:** Score > 800 (80%) = Assets moving in lockstep (Systemic Fragility/Crash Risk).

### 4.3 Market Structure Metrics
*   **Hurst Exponent:** R/S Analysis on Log Prices. ($H > 0.75$ = Crowded/Brittle Trend).
*   **Amihud Illiquidity:** Price impact per dollar traded. ($Z > 2.0$ = Liquidity Hole).

### 4.4 Capital Rotation (The "Offense")
*   **Logic:** Calculate 60-day Relative Strength (RS) of Sector ETFs vs `SPY`.
*   **Output:** Dynamically identify the current cycle (Early/Mid/Late/Recession) and return the specific actionable tickers (e.g., "Buy XLE, XLB").

---

## 5. Signal Logic & "Plain English" Translation

### 5.1 Regime Detection (The Tactical HUD)
Synthesize metrics into a user-friendly "Regime" state:
*   **FRAGILE RALLY (Melt-Up):** Price > 50MA **AND** Absorption > 850. *"Prices rising, but market locked. Drop likely."*
*   **DIVERGENCE TRAP:** Price > 50MA **AND** Turbulence > 180. *"Price masking internal break. Do not chase."*
*   **SYSTEMIC SELL-OFF:** Price < 50MA **AND** Absorption > 850. *"Everything falling together. Cash is safe."*
*   **CRASH ALERT:** Price < 50MA **AND** Turbulence > 370. *"Extreme volatility. Protect capital."*
*   **NORMAL MARKET:** Turbulence < 180. *"Structure stable."*

---

## 6. UI/UX Specifications (The "Tactical HUD")

### 6.1 Layout Strategy
1.  **Header:** Title + "System Horizon: 7-14 Days" Warning Banner.
2.  **The Tactical HUD (Top Component):**
    *   *Left:* **Regime Status** (e.g., "FRAGILE MELT-UP") + Plain English Analysis.
    *   *Middle:* **Safety Checks** (Stops: Tight/Standard, Leverage: Yes/No, Hedging: Yes/No).
    *   *Right:* **Opportunity Engine** (What to Buy? e.g., "Energy, Materials" from Cycle logic).
3.  **Metrics Row:** Turbulence (0-1000), Absorption (0-1000), Narrative Alpha (AI vs Crypto Delta).
4.  **Macro Row:**
    *   *Yield Curve (FRED):* Detect Inversion (<0) and **De-Inversion** (Crash Trigger).
    *   *Credit Stress (FRED):* High Yield Spreads Z-Score.
    *   *Sentiment (FinViz):* News Sentiment Score (0-100).
5.  **Charts:**
    *   *Main:* Market Health Monitor (Price vs Turbulence) with **Green Vertical Shading** for Divergence.
    *   *Secondary:* Liquidity Stress Gauge.
    *   *Heatmap:* 30-Day Rolling Correlation.
6.  **Sidebar:**
    *   **Catalyst Watch:** "Next Earnings" list using `yf.Ticker.get_earnings_dates()`.

---

## 7. File Structure
```text
/market_immune_system
│
├── main.py                # UI Controller (Streamlit)
├── market_immune_system.py # Math Engine (Turbulence, Absorption, Cycles)
├── macro_connector.py     # External Data (FRED, FinViz, Calendars)
├── requirements.txt       # yfinance, streamlit, plotly, scikit-learn, pyportfolioopt, fredapi, finvizfinance, nltk
```

## 8. Implementation Checklist
1.  **Setup:** Install dependencies. Set FRED API key in environment or use fallback.
2.  **Data Layer:** Implement `MacroConnector` and the robust `fetch_data` (Outer Join) in `MarketImmuneSystem`.
3.  **Math Layer:** Implement Mahalanobis, PCA, and Cycle Rotation logic.
4.  **UI Layer:** Build `main.py` with the `render_tactical_hud` function (HTML/CSS card) and the fixed Date Slider.
5.  **Validation:** Verify "Narrative Battle" populates (Crypto data present) and "Earnings Watch" shows dates.