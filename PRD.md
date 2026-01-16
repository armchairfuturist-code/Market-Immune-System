Here is the **Master Product Requirement Document (PRD)** designed specifically for use with AI-based IDEs (like Cursor, Windsurf, or GitHub Copilot/Workspace).

**How to use this:**
Copy the text below entirely and paste it into your AI IDE's chat or context window. It uses specific terminology (e.g., "Step-by-Step Implementation Strategy," "File Structure") that forces the AI to code systematically rather than hallucinating random scripts.

***

# PROJECT: Market Immune System v2.0 (Master PRD)

## 1. Executive Summary
**Goal:** Build a Python-based financial dashboard ("The Market Immune System") that acts as an early warning detector for market turbulence.
**Core Philosophy:** Detect structural breaks in correlation and volatility to signal "Buy Low / Sell High" opportunities 7-10 days in advance.
**Target Output:** A local Streamlit web application with a "MetaMint" dark-mode aesthetic.

## 2. Technical Stack & Constraints
*   **Language:** Python 3.10+
*   **Frontend:** `streamlit` (must use `st.set_page_config(layout="wide")`)
*   **Data Source (Price):** `yfinance` (Free tier). **Crucial:** Must implement "Zero-Trust" fixes for data integrity.
*   **Data Source (Macro):** `fredapi`. **API Key:** `fc2e25e796936565703717197b34efa8`
*   **Math Backend:** `numpy`, `pandas`, `scipy` (optimize with `numba` if possible), `PyPortfolioOpt` (for Ledoit-Wolf shrinkage).
*   **Visualization:** `plotly.graph_objects` (interactive financial charts).
*   **Design:** Custom CSS for "MetaMint" Dark Mode (Black/Dark Grey backgrounds, Neon Green accents).

## 3. Data Architecture (The "Zero-Trust" Engine)

### 3.1 Asset Universe (The "99")
The system must initialize with this specific list across 4 pillars:
1.  **Broad:** SPY, QQQ, IWM, VXX, TLT, GLD, UUP.
2.  **Crypto:** BTC-USD, ETH-USD, SOL-USD.
3.  **AI/Growth:** NVDA, AMD, TSM, SMCI, VRT, PLTR, CEG, VST, CCJ.
4.  **Defensive:** XLV, XLP, XLU.
*Action:* Auto-fill the remaining slots to reach ~99 assets with top liquid US stocks if needed.

### 3.2 Resilience Fixes (Mandatory Implementation)
The AI must implement these specific functions in `data_loader.py`:
1.  **The "Monday" Fix:** When fetching data, split Crypto (24/7) and Stocks (M-F). Perform an **Outer Join** on the Datetime index and forward-fill (`ffill`) missing stock data on weekends/holidays to honor Bitcoin's volatility signal.
2.  **The "Today" Fix:** Standard `yfinance` history often omits the live intraday candle.
    *   *Logic:* Fetch `history(period="1y")`. Then fetch `history(period="1d", interval="1m")` (or "1h"). Resample the live data to 1 Day and `.loc` append it to the main dataframe.
3.  **The "Time Travel" Fix:** In the UI, the date slider must strictly cap `max_value` at `usage_date.today()` to prevent look-ahead errors.

## 4. Mathematical Core (The Algorithms)

### 4.1 Statistical Turbulence (0-1000 Index)
*   **Input:** Log-returns of the Asset Universe.
*   **Calculation:** Mahalanobis Distance using a **Ledoit-Wolf Shrinkage** covariance matrix (reduces noise).
*   **Normalization:**
    *   Calculate raw Mahalanobis distance.
    *   Rolling 365-day average helps, but final output must be scaled 0-1000.
    *   **Anchor:** Historic 99th percentile = Score 370.

### 4.2 Absorption Ratio (Systemic Fragility)
*   **Method:** Principal Component Analysis (PCA) via `scikit-learn` or `statsmodels`.
*   **Formula:** (Variance of Top 20% Eigenvectors) / (Total Variance).
*   **Logic:** If Score > 800 (80%), assets are moving in lockstep (High Fragility/Crash risk).

### 4.3 Regime Detection Logic
Implement a classifier function `get_market_regime(turbulence, absorption, price_trend)`:
*   **FRAGILE RALLY:** Price Up + Absorption > 850.
*   **STRUCTURAL DIVERGENCE:** Price Up + Turbulence > 180 (The "Trap").
*   **SYSTEMIC SELL-OFF:** Price Down + Absorption > 850.
*   **CRASH ALERT:** Turbulence > 370.
*   **NORMAL:** Turbulence < 180.

## 5. UI/UX Specifications (The "Tactical HUD")

### 5.1 Design System (MetaMint)
*   **Background:** `#0E1117` (Deep Black/Blue).
*   **Card Background:** `#262730` (Dark Grey).
*   **Text:** `#FAFAFA`.
*   **Accents:**
    *   `#00FF80` (Neon Green - Safe/Buy).
    *   `#FF4B4B` (Neon Red - Danger/Sell).
    *   `#FFAA00` (Amber - Warning).

### 5.2 Dashboard Layout
1.  **Top Row (The HUD):** 4 Metric Cards.
    *   *Market Status:* E.g., "RISK ON" (Green) or "CRASH ALERT" (Red).
    *   *Turbulence Score:* Large Number (e.g., "450").
    *   *Fragility (Absorption):* Percentage (e.g., "82%").
    *   *Recession Signal:* (From FRED Yield Curve).
2.  **Main Chart (Divergence Detector):**
    *   Dual Axis Plotly Chart.
    *   Line A: SPX Price (White).
    *   Line B: Turbulence Score (Red area chart).
    *   **Feature:** Add vertical Green Shading to the background whenever "Turbulence > 180 AND SPX > 50-day MA" (Visualizing the divergence).
3.  **Bottom Row:**
    *   *Col 1:* Sector Rotation List (Top 3 sectors to buy now).
    *   *Col 2:* "Narrative Battle" (Crypto vs. AI relative strength chart).

## 6. Suggested File Structure

```text
/market_immune_system
│
├── main.py                # usage: streamlit run main.py
├── requirements.txt       # yfinance, streamlit, plotly, scikit-learn, pyportfolioopt, fredapi
├── config.py              # Asset lists, API Keys, Constants (Thresholds)
│
├── /core
│   ├── data_loader.py     # YFinance & FRED fetching + "Monday/Today" fixes
│   ├── math_engine.py     # Mahalanobis, PCA, Hurst calculations
│   └── text_analysis.py   # (Optional) Simple Finviz sentiment logic
│
└── /ui
    ├── theme.css          # MetaMint CSS injection
    └── charts.py          # Plotly visualization wrappers
```

## 7. Implementation Step-by-Step for AI
1.  **Setup:** Initialize the file structure and `requirements.txt`.
2.  **Data Engine:** Write `data_loader.py` first. Verify the "Monday Fix" works by fetching BTC and SPY and checking the weekend index.
3.  **Math Engine:** Implement `math_engine.py`. Create the Turbulence function with Ledoit-Wolf shrinkage.
4.  **UI Core:** Build `main.py` sidebar and basic layout.
5.  **Integration:** Connect Data -> Math -> UI.
6.  **Refinement:** Apply the "MetaMint.png" CSS in the project folder as a style guide and finalize the Regime Detection Logic logic.