**Objective:** Build a single-page interactive Risk Dashboard ("The Market Immune System") to detect market fragility, structural breaks, and high-probability "buy low" setups.
**Tech Stack:** Python 3.10+, Streamlit (UI), Plotly (Charts), YFinance (Data), PyPortfolioOpt (Covariance), Scikit-Learn (PCA), Scipy (Stats).

## 1. System Architecture
The system functions as a daily monitor that ingests price data, smooths it, calculates covariance structures, and alerts on structural breakdowns (Turbulence) or correlation spikes (absorption), cross-referenced with liquidity health.

### 1.1 Inputs (The Asset Universe)
Define a constant `ASSET_UNIVERSE`. It must include exactly these liquid tickers across 4 distinct groups:
1.  **Broad Markets (40%):** SPY, QQQ, DIA, IWM, VXX, EEM, EFA, TLT, IEF, SHY, LQD, HYG, BND, AGG, GLD, SLV, CPER, USO, UNG, DBC, PALL, UUP, FXE, FXY, FXB, CYB, XLF, XLE, XLK, XLY, XLI, XLB, XLRE.
2.  **Crypto (10%):** BTC-USD, ETH-USD, SOL-USD, BNB-USD, XRP-USD, ADA-USD, AVAX-USD, DOGE-USD, DOT-USD, LINK-USD.
3.  **AI & Growth (30%):** NVDA, AMD, TSM, AVGO, ARM, MU, INTC, TXN, LRCX, AMAT, VRT, ANET, SMCI, PLTR, DELL, HPE, CSCO, IBM, ORCL, CEG, VST, NRG, CCJ, URA, NEE, SO, DUK, TSLA, PATH, ISRG, BOTZ, ROBO, ARKK.
4.  **Defensive (20%):** XLV, JNJ, PFE, MRK, ABBV, UNH, LLY, AMGN, XLP, PG, KO, PEP, COST, WMT, PM, XLU, O, AMT, CCI, PSA, DLR, USMV, SPLV.

### 1.2 Data Ingestion Layer
*   **Library:** `yfinance`.
*   **Range:** Fetch last `750` days.
*   **Specifics:** Retrieve `Close` prices for all assets **AND** `Volume` specifically for `SPY` (used as the system-wide liquidity proxy).
*   **Cleaning:**
    *   Forward Fill (`ffill`) missing data (crucial for aligning Crypto trading 24/7 with Stocks M-F).
    *   Drop columns with >10% missing data.
    *   Calculate **Log Returns** for all math operations.
*   **Caching:** Decorate the fetch function with `@st.cache_data(ttl=3600)` to prevent re-downloading on every UI interaction.

## 2. Mathematical Core (The Evaluation Engine)

### Metric A: Statistical Turbulence (Mahalanobis Distance)
*   **Concept:** Measures how "alien" today's return vector is compared to the historical covariance structure.
*   **Lookback:** Rolling 365-day window for covariance `S`.
*   **Covariance Hardening:** Use **Ledoit-Wolf Shrinkage** (via `pypfopt.risk_models.CovarianceShrinkage`). Do NOT use standard `numpy.cov` to avoid singular matrices.
*   **Formula:** $D_t = \sqrt{ (r_t - \mu) \Sigma^{-1} (r_t - \mu)' }$
*   **Scale Normalization (0-1000 Calibrated):**
    1.  Calculate the **99th Percentile ($P_{99}$)** of the trailing 365-day raw $D_t$ values.
    2.  **Anchor Point:** We define the "Extreme/Critical" threshold as **370** (which is 37% of the 0-1000 scale).
    3.  **Formula:** `Final_Score = (Raw_Dt / P99) * 370`.
    4.  **Cap:** `Final_Score = min(Final_Score, 1000)`.
    *   *Result:* A standard "bad day" (99th percentile) will hit exactly 370. A "Black Swan" (3x a bad day) will hit ~1000.
*   **Smoothing:** Apply a 3-Day EMA to the final Score.

### Metric B: Absorption Ratio (Systemic Fragility)
*   **Concept:** Measures how much the market is moving in lockstep (unification).
*   **Method:** Principal Component Analysis (PCA) via `sklearn.decomposition.PCA`.
*   **Calculation:**
    1.  Fit PCA on the correlation matrix of the last 365 days.
    2.  Calculate Variance Explained by the top 20% of Eigenvectors.
    3.  Scale result: `Absorption_Score = (Sum_Variance_Top_20_Percent) / (Total_Variance) * 1000`.

### Metric C: Fractal Efficiency (The Hurst Exponent)
*   **Concept:** Measures the "memory" of the price series to detect crowded trades. If trends are "too smooth" ($H \to 1.0$), the market is fragile.
*   **Input:** Log prices of `SPY` (last 300 days).
*   **Calculation:**
    1.  Implement a standard **Rescaled Range (R/S) Analysis** or a simplistic **Variance Ratio Test**.
    2.  Estimate $H$ via the slope of `log(lag)` vs `log(volatility)` using `numpy.polyfit`.
*   **Note:** Use vectorized operations where possible to prevent lag.

### Metric D: Liquidity Stress (Amihud Ratio Z-Score)
*   **Concept:** Measures the price impact per dollar traded. If Price moves significantly on Low Volume, a "Liquidity Hole" is forming.
*   **Input:** `SPY` DataFrame (Close Price and Volume).
*   **Calculation:**
    1.  Compute Daily Amihud: `Abs(Daily_Return) / (Price * Volume)`. *Handle 0 volume by replacing with NaN or previous.*
    2.  Smooth: Apply a **10-day Moving Average**.
    3.  Normalize: Calculate the **Z-Score** relative to the trailing 365-day mean/std dev.

## 3. Signal Logic (The Composite "Buy Low" Detector)
Implement a function `generate_signal(spx_return, turbulence_score, absorption_score, hurst, liquidity_z)`:

**Composite Signal States (Based on 0-1000 Calibrated Scale):**
1.  **CONDITION: FRAGILE (The Bubble)**
    *   **Logic:** `Absorption > 800` **OR** (`Hurst > 0.75` AND `Liquidity_Z > 1.5`).
    *   **Message:** "WARNING: Fragile Structure. Crowded Trade + Thin Liquidity."
    *   **Action:** Cash preservation (Code: ORANGE).
2.  **CONDITION: CRASH (The Event)**
    *   **Logic:** `Turbulence > 370` (Extreme) **AND** `Liquidity_Z > 2.0`.
    *   **Message:** "CRASH: Liquidity Evaporation Event."
    *   **Action:** DO NOT CATCH THE KNIFE (Code: BLACK).
3.  **CONDITION: DIVERGENCE (The Trap)**
    *   **Logic:** `SPX_Price > 50-day SMA` (Uptrend) **AND** `Turbulence > 180` (Warning).
    *   **Message:** "DIVERGENCE: Market rising on broken structure."
    *   **Action:** CAUTION (Code: RED).
4.  **CONDITION: OPPORTUNITY (The Buy Signal)**
    *   **Logic:**
        *   `Turbulence` *was* > 370 (recent panic).
        *   `Turbulence` is now < 300 (calming).
        *   `Liquidity_Z` < 1.0 (Market Makers have returned).
        *   `Hurst` < 0.5 (Mean Reversion active).
    *   **Message:** "BUY SIGNAL: Stress clearing, Liquidity returning."
    *   **Action:** ENTER (Code: BLUE/GREEN).

## 4. UI Requirements (Streamlit)
The layout must be modern and dark-mode friendly.

1.  **Sidebar:**
    *   Date Range Slider.
    *   "Refresh Data" button.
    *   List of Top 5 contributing assets to today's Turbulence (calculated via Partial Mahalanobis contribution).
2.  **Main Panel - Header:**
    *   Four columns metric display:
        1.  **Turbulence Score** (0-1000, Normalized)
        2.  **Absorption Ratio** (0-1000)
        3.  **Liquidity Z-Score** (Float)
        4.  **Signal Status** (Colorful text based on Section 3).
3.  **Main Panel - Charts:**
    *   **Chart 1:** "Market Health Monitor" (Plotly). Dual-axis chart.
        *   **Left Axis (Primary):** Turbulence Score (Area/Shadow). Range Fixed `[0, 1000]`.
        *   **Right Axis (Secondary):** SPX Price (Line) and SPX 50-day SMA (Dashed Line).
        *   **Threshold Line 1 (Yellow/Dashed):** Static horizontal line at **180**. Label: "Warning (18%)".
        *   **Threshold Line 2 (Red/Dashed):** Static horizontal line at **370**. Label: "Critical (37%)".
        *   **Overlay:** Green vertical shading when `Turbulence > 180` AND `SPX > 50-day SMA`.
    *   **Chart 2:** "Liquidity Stress Gauge" (Plotly Line Chart).
        *   X-Axis: Date.
        *   Y-Axis: Amihud Z-Score.
        *   Red Zone: Background shading > 2.0.
    *   **Chart 3:** "Fragility Heatmap". Rolling Correlation matrix of the last 30 days.

## 5. Constraint Checklist for Implementation
*   [ ] Handle the `Singular Matrix` error using Pseudo-Inverse (`pinv`) if Shrinkage fails.
*   [ ] Ensure `ASSET_UNIVERSE` tickers are stripped of potential whitespace.
*   [ ] **Crucial:** Include a `try/except` block for the `yfinance` download.
*   [ ] **Crucial:** Handle `ZeroDivisionError` in Amihud calculation if Volume is 0.
*   [ ] Code must be modular: `class MarketImmuneSystem` for logic, `main.py` for UI.
