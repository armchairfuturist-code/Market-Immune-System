# Gemini Developer Guardrails: Market Immune System

## 🎯 Primary Directive
Before starting any feature work, refactor, or bug fix, always read and follow the requirements in `PRD.md`. This is the absolute source of truth for business logic and mathematical scaling.

## 🛠 Tech Stack Constraints
- **Framework:** Streamlit (Layout: Wide mode only).
- **Data Engine:** `yfinance` for prices (threads=False), `fredapi` for macro truth.
- **Math:** `numpy`, `pandas`, `PyPortfolioOpt` (Ledoit-Wolf Shrinkage).
- **Visualization:** `plotly.graph_objects` (Avoid `st.line_chart` for complex dual-axis data).

## 📂 Project Structure & Logic flow
- `config.py`: Global constants, asset lists, and MetaMint theme colors.
- `main.py`: Entry point. UI layout and high-level synthesis only.
- `core/data_loader.py`: All I/O logic (Stocks, Crypto, Futures). Must implement "Zero-Trust" data alignment and `threads=False` for yfinance.
- `core/math_engine.py`: Pure mathematical functions (Turbulence, Absorption, Hurst, Liquidity). No UI code.
- `core/macro_connector.py`: External API wrappers (FRED, Sentiment).
- `core/cycle_engine.py`: Market Phase detection logic.
- `core/report_generator.py`: Structured data generation for status reports.
- `ui/charts.py`: Plotly figure generators (Divergence Chart, Narrative Battle).
- `ui/theme.css`: Custom CSS for MetaMint premium aesthetic.

## 🛡️ Critical Data Integrity Rules (Zero-Trust)
1. **The Alignment Requirement:** Never assume `SPY` and `BTC-USD` have the same index. Always use `join='outer'` when merging and `ffill()` to propagate Friday stock closes across crypto-active weekends.
2. **The "Today" Fix:** `yf.download` daily candles do not include the current day. You must explicitly fetch a 1-day/1-minute snapshot and resample/append it to provide "Live" status for the current date.
3. **The Covariance Guardrail:** Never use `df.cov()` directly. Always use `pypfopt.risk_models.CovarianceShrinkage(df).ledoit_wolf()` or regularized covariance to prevent singular matrix errors with 99+ assets.
4. **NaN/Inf Handling:** Quantitative calculations (Mahalanobis/PCA) will crash on `inf` values. Always run `.replace([np.inf, -np.inf], np.nan)` and sanitize data (drop sparse columns) before math operations.
5. **Threading Lock:** Always use `threads=False` in `yf.download` calls to prevent internal race conditions (`KeyError`) during multi-ticker fetches.

## 🧠 Dynamic Inference & Anti-Static Logic
1. **No Static Interpretations:** Do not hardcode "example" explanations or dates (e.g., a static CPI calendar or a hardcoded "The market is rising because of X"). 
2. **Variable-Driven Narratives:** All text in the "Analysis" or "Interpretation" fields must be derived from the current state of variables (Hurst, Absorption, Turbulence, Yield Curve, any variable where data is pulling from live api sources. 
3. **Historical Context Awareness:** When the user changes the "Analysis Date," the dashboard's narrative must mutate to reflect that specific historical regime. It must not describe the past using today's news context.
4. **The "Zero-Hallucination" Rule:** If an official data source for a catalyst (like CPI) is unavailable via API, do not create a placeholder dictionary. Instead, omit the metric or provide a verified "Next Event" link to a live source.


## 🎨 UI/UX Styling (MetaMint Theme)
- **Component-Based UI:** Do NOT use raw HTML/JS injection for layout if possible. Use native Streamlit components (`st.container`, `st.metric`) styled via `ui/theme.css` classes (e.g., `.metamint-card`, `.status-badge`) for stability and responsiveness.
- **Translation Layer:** Convert technical cycle names (e.g., "Late Cycle") into actionable asset lists (e.g., "Energy, Staples") for layman readability.
- **Color Logic:** 
    - Green (`#00C853`) for Healthy/Normal/Trend.
    - Amber (`#FFAB00`) for Warning/Trap Zone (280 Threshold).
    - Red (`#FF1744`) for Critical/Crash Alert (370 Threshold).

## 🧠 Performance Optimization
- **Caching:** Use `@st.cache_data` for all data fetching and heavy math. 
- **TTL Strategy:** 
    - `get_market_data`: `ttl=3600` (1h) for fast-moving price/earnings data.
    - `get_macro_data`: `ttl=86400` (24h) for slow-moving macro (FRED) data.
- **Vectorization:** Prefer `numpy` vectorization over pandas `.apply()` for rolling window calculations (especially Hurst and Turbulence).

## 🚀 Proactive Troubleshooting
- If `yfinance` returns a "No data found" error for an asset, do not crash the app. Log the warning and exclude that asset from the covariance matrix for that session.
- If the FRED API key is missing or invalid, the `MacroConnector` must fall back to a safe empty state or calculated proxies rather than crashing.
- **Futures Projection:** Ensure `ES=F` data is available before plotting projection lines; handle missing futures data gracefully.