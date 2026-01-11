**Do not rely solely on the GitHub repositories.** While they mathematically validate the concepts, they are research code (often spaghetti logic, outdated dependencies, or lacking the specific "Live Dashboard" architecture you need).

To achieve a "One Shot" success with Claude Opus 4.5 or Gemini 3 Pro, you need a PRD that bridges the gap between *Academic Theory* and *Production Code*.

I have one critical technical recommendation before you prompt the AI: **Use `Streamlit`**. It is the only framework that will allow an LLM to generate a fully functional, interactive financial dashboard in a single pass without complex callback logic (like Dash/React).

Here is the exact Prompt/PRD to paste into your IDE.

***

# Project Requirements Document (PRD): The "Market Immune System" Dashboard

**Role:** Senior Quantitative Developer / Financial Engineer
**Objective:** Build a single-page interactive Risk Dashboard ("The Market Immune System") to detect market fragility and crash signals.
**Tech Stack:** Python 3.10+, Streamlit (UI), Plotly (Charts), YFinance (Data), PyPortfolioOpt (Covariance), Scikit-Learn (PCA), Scipy (Stats).

## 1. System Architecture
The system functions as a daily monitor that ingests price data, smooths it, calculates covariance structures, and alerts on structural breakdowns (Turbulence) or correlation spikes (absorption).

### 1.1 Inputs (The Asset Universe)
Define a constant `ASSET_UNIVERSE`. It must include exactly these liquid tickers across 4 distinct groups:
1.  **Broad Markets (40%):** SPY, QQQ, DIA, IWM, VXX, EEM, EFA, TLT, IEF, SHY, LQD, HYG, BND, AGG, GLD, SLV, CPER, USO, UNG, DBC, PALL, UUP, FXE, FXY, FXB, CYB, XLF, XLE, XLK, XLY, XLI, XLB, XLRE.
2.  **Crypto (10%):** BTC-USD, ETH-USD, SOL-USD, BNB-USD, XRP-USD, ADA-USD, AVAX-USD, DOGE-USD, DOT-USD, LINK-USD.
3.  **AI & Growth (30%):** NVDA, AMD, TSM, AVGO, ARM, MU, INTC, TXN, LRCX, AMAT, VRT, ANET, SMCI, PLTR, DELL, HPE, CSCO, IBM, ORCL, CEG, VST, NRG, CCJ, URA, NEE, SO, DUK, TSLA, PATH, ISRG, BOTZ, ROBO, ARKK.
4.  **Defensive (20%):** XLV, JNJ, PFE, MRK, ABBV, UNH, LLY, AMGN, XLP, PG, KO, PEP, COST, WMT, PM, XLU, O, AMT, CCI, PSA, DLR, USMV, SPLV.

### 1.2 Data Ingestion Layer
*   **Library:** `yfinance`.
*   **Range:** Fetch last `750` days (ensures sufficient buffer for 365-day rolling window).
*   **Cleaning:**
    *   Forward Fill (`ffill`) missing data (crucial for aligning Crypto trading 24/7 with Stocks M-F).
    *   Drop columns with >10% missing data.
    *   Calculate **Log Returns** for all math operations.
*   **Caching:** Decorate the fetch function with `@st.cache_data(ttl=3600)` to prevent re-downloading on every UI interaction.

## 2. Mathematical Core (The Evaluation Engine)

### Metric A: Statistical Turbulence (Mahalanobis Distance)
*   **Concept:** Measures how "alien" today's return vector is compared to the historical covariance structure.
*   **Lookback:** Rolling 365-day window for covariance `S`.
*   **Covariance Hardening:** Use **Ledoit-Wolf Shrinkage** (via `pypfopt.risk_models.CovarianceShrinkage`). Do NOT use standard `numpy.cov`, as it will produce singular/noisy matrices with 99 assets.
*   **Formula:** $D_t = \sqrt{ (r_t - \mu) \Sigma^{-1} (r_t - \mu)' }$
*   **Normalization:** Convert $D_t$ to a **0-1000 Score** using the **Chi-Squared Cumulative Distribution Function (CDF)** where degrees of freedom = number of assets.
    *   `Score = stats.chi2.cdf(squared_distance, df=N_assets) * 1000`
*   **Smoothing:** Apply a 3-Day EMA to the final Score to reduce noise.

### Metric B: Absorption Ratio (Systemic Fragility)
*   **Concept:** Measures how much the market is moving in lockstep (unification).
*   **Method:** Principal Component Analysis (PCA) via `sklearn.decomposition.PCA`.
*   **Calculation:**
    1.  Fit PCA on the correlation matrix of the last 365 days.
    2.  Calculate Variance Explained by the top 20% of Eigenvectors.
    3.  Scale result: `Absorption_Score = (Sum_Variance_Top_20_Percent) / (Total_Variance) * 1000`.

## 3. Signal Logic (The "Buy Low" Detector)
Implement a function `generate_signal(spx_return, turbulence_score, absorption_score)`:
*   **condition_green:** Turbulence < 750 (Normal).
*   **condition_orange:** Absorption > 800 (Fragile/Unified Market).
*   **condition_red (Divergence):** SPX price > 0 (Green day) **AND** Turbulence > 750. *Message: "DIVERGENCE: Market rising on broken structure."*
*   **condition_black (Crash):** SPX price < -1.0% **AND** Turbulence > 900.
*   **condition_opportunity (Buy Low):** If Turbulence *was* > 900 yesterday and is < 850 today (Mean Reversion).

## 4. UI Requirements (Streamlit)
The layout must be modern and dark-mode friendly.

1.  **Sidebar:**
    *   Date Range Slider.
    *   "Refresh Data" button.
    *   List of Top 5 contributing assets to today's Turbulence (calculated via Partial Mahalanobis contribution).
2.  **Main Panel - Header:**
    *   Three columns displaying: **Current Turbulence (0-1000)**, **Absorption Ratio**, and **Signal Status (Green/Red/Black)** with appropriate color coding.
3.  **Main Panel - Charts:**
    *   **Chart 1:** "Market Health Monitor" (Plotly). Dual-axis chart.
        *   Left Axis: SPY Cumulative Return (Line).
        *   Right Axis: Turbulence Score (Area/Shadow).
        *   Horizontal Line at 750 (Warning Threshold).
    *   **Chart 2:** "Fragility Heatmap". Rolling Correlation matrix of the last 30 days (Plotly Heatmap).

## 5. Constraint Checklist for Implementation
*   [ ] Handle the `Singular Matrix` error using Pseudo-Inverse (`pinv`) if Shrinkage fails.
*   [ ] Ensure `ASSET_UNIVERSE` tickers are stripped of potential whitespace.
*   [ ] Code must be modular: `class MarketImmuneSystem` for logic, `main.py` for UI.
*   [ ] **Crucial:** Include a `try/except` block for the `yfinance` download to handle ticker changes/delistings gracefully without crashing the app.
