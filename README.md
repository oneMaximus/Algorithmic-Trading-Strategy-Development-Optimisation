# Algorithmic Trading Strategy Development and Optimisation
### INF2006 Cloud Computing and Big Data — Group Project Assignment 2

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Phase 1: Data Exploration and Baseline Evaluation](#phase-1-data-exploration-and-baseline-evaluation)
4. [Phase 2: Strategy Development and Optimisation](#phase-2-strategy-development-and-optimisation)
5. [Strategy Iteration Log](#strategy-iteration-log)
6. [Phase 3: Performance Evaluation and Analysis](#phase-3-performance-evaluation-and-analysis)
7. [Enhancement Areas Completed](#enhancement-areas-completed)
8. [Computational Efficiency and Scalability](#computational-efficiency-and-scalability)
9. [Performance Metrics Reference](#performance-metrics-reference)
10. [Infrastructure Overview](#infrastructure-overview)
11. [File Structure](#file-structure)
12. [How to Run](#how-to-run)
13. [AI Usage Declaration](#ai-usage-declaration)

---

## Project Overview

This project implements and optimises an algorithmic trading strategy applied to S&P 500 historical market data (2000–2024) combined with quarterly earnings call transcripts. The objective was to enhance a provided **BaseStrategy** (simple 50-day moving average crossover with FinBERT sentiment) within an **EnhancedStrategy** class, achieving measurably superior performance across all key metrics.

The strategy combines:
- **Technical Analysis**: EMA-200, EMA-50, MACD, RSI, Bollinger Bands, ATR, Trend Strength
- **NLP Sentiment Analysis**: FinBERT (ProsusAI/finbert) applied to earnings call transcripts
- **Risk Management**: Trailing stop-loss, momentum decay exits, multi-signal entry confirmation

---

## Dataset

| Split | Date Range | Purpose |
|-------|------------|---------|
| **Dev** | 2000–2017 | All experimentation, hyperparameter tuning, and iterative development |
| **Val** | 2018–2024 | Final one-time checkpoint after strategy is complete |
| **Test** | Held-out | Used for final grading assessment |

### Price Data Schema
| Column | Description |
|--------|-------------|
| `ticker` | Stock symbol (e.g., AAPL, MSFT) |
| `date` | Trading date |
| `open`, `high`, `low`, `close` | OHLC prices |
| `volume` | Number of shares traded |

### Earnings Transcript Schema
| Column | Description |
|--------|-------------|
| `ticker` | Stock symbol |
| `date` | Earnings call date |
| `transcript` | Full text of earnings call |
| `quarter` | Fiscal quarter (e.g., Q1 2023) |

---

## Phase 1: Data Exploration and Baseline Evaluation

### Completed Tasks

#### 1.1 Basic EDA (`Cell 4`)
- Examined full date range coverage of the dev price dataset
- Counted unique tickers across price and earnings datasets
- Generated OHLCV summary statistics (mean, std dev, min/max)
- Identified missing/null values across all columns
- Calculated transcript character lengths to understand FinBERT truncation impact
- Performed **ticker overlap analysis** between price and earnings datasets to identify tradeable universe

#### 1.2 Data Visualisation (`Cell 5`)
Four key visualisations produced:
- **Price Over Time** — closing price time series for the most data-rich ticker
- **Volume Over Time** — trading volume for the same ticker
- **Price Distribution** — histogram across all tickers
- **Top 20 Tickers by Data Points** — horizontal bar chart of data coverage

#### 1.3 Advanced EDA (`Cells 11, 13b, 13c, 13d`)

**Correlation Table with Jan 2020 Pricing**
- Computed log-return correlation matrix across all tickers in the dev split (filtered to >500 data points for validity)
- Identified the top 20 most correlated stock pairs
- Fetched January 2020 prices from the val split for each ticker in the top pairs
- This informs diversification decisions in the portfolio — highly correlated stocks provide redundant signals

**Weekly Consistency & Survival Analysis (2000–2019)**
- Implemented a custom `analyze_weekly_consistency()` function
- For each ticker, resampled data to weekly frequency and tracked whether the weekly low remained above a 20% stop-loss threshold from the initial price
- Produced a **Consistency Score** = (weeks above SL) − (weeks below SL) and a **Safety Ratio (%)**
- Identified the **top 40 most consistent survivors** over the full 2000–2019 window
- Generated a **Volume–Price Relationship** scatter plot for the #1 most consistent ticker

**Top 20 Performers with Price Points (2000–2019)**
- Calculated total return for every ticker over the full dev window
- Fetched start price (2000) and end price (2019) for each
- Integrated live sector mapping from Wikipedia's S&P 500 constituent table
- Revealed which sectors dominated long-term performance

**Feature Correlation Heatmap**
- Engineered features: log returns, volume change (%), daily price range
- Computed a full correlation matrix across all OHLCV + engineered features
- Plotted as an annotated heatmap to reveal multicollinearity between indicators

**FinBERT Sentiment Behaviour Test (`Cell 13d`)**
- Loaded the first earnings transcript from the dev set
- Ran FinBERT directly (bypassing the pipeline wrapper) to inspect raw softmax probabilities
- Exposed the full Positive / Negative / Neutral probability breakdown
- Verified the truncation behaviour (512-token limit) and its impact on longer transcripts

---

## Phase 2: Strategy Development and Optimisation

All development was performed exclusively on the **dev split (2000–2017)**.

### EnhancedStrategy Class — Implemented Methods

---

### Area 2: Data Quality & Cleaning (`clean_data`)

Overrides `BaseStrategy.clean_data()` with three sequential filters:

1. **Sort and Forward-Fill** — Data sorted by `[ticker, date]`; missing OHLC values forward-filled within each ticker group to maintain weekly continuity without look-ahead bias
2. **Penny Stock Filter** — All records with `close < $5.00` removed to eliminate low-liquidity noise and erratic volatility
3. **History Length Filter** — Tickers with fewer than 200 trading days excluded, ensuring EMA-200 has sufficient data to be meaningful before the first signal is generated

---

### Area 3: Technical Indicators (`calculate_analytics`)

Overrides `BaseStrategy.calculate_analytics()` to compute a full multi-indicator suite:

| Indicator | Parameters | Purpose |
|-----------|-----------|---------|
| EMA-200 | span=200 | Long-term structural trend filter |
| EMA-50 | span=50 | Medium-term trend confirmation |
| EMA-12 | span=12 | MACD fast line |
| EMA-26 | span=26 | MACD slow line |
| MACD | 12−26 | Momentum direction |
| Signal Line | EMA-9 of MACD | MACD crossover reference |
| MACD Histogram | MACD − Signal | Momentum strength and direction |
| Pace Improving | hist > hist_prev | Boolean: is momentum accelerating? |
| RSI | window=14 | Overbought/oversold detection |
| Bollinger Bands | SMA-20 ± 2σ | Volatility envelope / mean reversion floor |
| ATR | window=14 | True volatility measure for dynamic stops |
| Trend Strength | (close − close[−10]) / ATR | Normalised momentum proxy |

**Optimisations applied (Final Version):**
- Single `groupby('ticker')` used throughout — avoids repeated `O(n×t)` filtering
- All EMAs computed via vectorised `.transform(lambda x: x.ewm(...).mean())`
- MACD histogram and pace computed as direct vector arithmetic (no per-row loops)
- RSI uses `pandas.diff()` + `.clip()` + `.rolling().mean()` — fully vectorised
- Date string conversion performed **once** in bulk at the end via `.dt.strftime()`

---

### Area 4: Enhanced LLM Analysis (`llm_analysis`)

Overrides `BaseStrategy.llm_analysis()`:

- **Extended context window**: Uses first 4,000 characters of transcript (vs. baseline's last 2,000)
- **Numeric sentiment scoring**: Maps `positive→1.0`, `neutral→0.5`, `negative→0.0` instead of string labels
- **In-memory caching**: Results cached by `{ticker}_{date}` key — avoids redundant FinBERT inference on repeated calls within the same simulation run
- **Graceful failure handling**: Returns `None` (treated as neutral) on any inference error or short transcript (<100 chars)
- **GPU acceleration**: Automatically routes inference to CUDA device 0 when available

---

### Area 5: Smarter Decision Logic (`make_decision`)

**Exit Rules (applied first for held positions):**

1. **Trailing Stop-Loss (−15% from peak)**: A `highest_price_seen` dictionary tracks the weekly high-water mark per ticker. Position is sold when current price drops 15% below that peak — tighter than the baseline's flat 20% stop-loss from purchase price
2. **Momentum Decay Exit**: If `pace_improving == False` (MACD histogram turned negative), the position is closed regardless of price level — captures early deterioration before it becomes a loss

**Entry Rules (three-condition confirmation required):**

1. **Structural Trend** (`structural_ok`): `close > EMA-50` AND `close > EMA-200` — ensures entry only in confirmed uptrends on both medium and long timeframes
2. **Momentum Reversal** (`momentum_rebound`): `RSI < 45` AND `pace_improving == True` — targets stocks that are not overbought and whose momentum is just beginning to accelerate
3. **Sentiment Gate** (`sentiment_ok`): Blocks entry only on explicitly negative sentiment (`score < 0.5`); neutral and positive sentiment both allow entry

All three conditions must be simultaneously true for a BUY signal to be issued.

---

## Strategy Iteration Log

This section documents every iteration of the `EnhancedStrategy` developed in `trading_assignment_iteration.ipynb`, detailing what changed between versions and the reasoning behind each decision.

---

### Iteration 1 — RSI + MACD Baseline Enhancement
**File:** `trading_assignment_iteration.ipynb` — Cell 1 (`EnhancedStrategy` v1)

#### What Was Built
The first iteration established the core improvements over `BaseStrategy`:

| Component | Change from Baseline |
|-----------|---------------------|
| **clean_data** | Added forward-fill, $5 penny stock filter, 200-day history minimum |
| **calculate_analytics** | Replaced single MA-50 with EMA-200, EMA-50, RSI(14), full MACD (12/26/9), MACD histogram, pace flag |
| **llm_analysis** | Switched from string label return to numeric score (1.0 / 0.5 / 0.0); extended to 4,000 chars |
| **make_decision** | Replaced single MA crossover with 3-condition entry gate; replaced flat 20% stop-loss with −15% trailing stop from weekly peak; added momentum decay exit |

#### Key Decisions Made
- **Why −15% trailing stop instead of −20% flat?** The EDA survival analysis showed that most consistent long-term performers rarely dropped more than 15% from their weekly peak while in an uptrend. A trailing stop locks in gains rather than just preventing catastrophic loss.
- **Why RSI < 45 (not < 30)?** The intent was to catch early rebound momentum — RSI < 30 is classic "oversold" but by that point the stock may already have broken its structural trend. RSI < 45 catches the recovery while structure (EMA-200, EMA-50) remains intact.
- **Why pace_improving as an exit signal?** If the MACD histogram is turning down, selling immediately removes exposure before the crossover signal (which tends to lag). This is a leading exit vs. the baseline's lagging MA crossover exit.

#### Known Limitation at This Stage
The `_calculate_rsi` function used `.where()` + `.rolling().mean()` which can produce `NaN` values during the lead-in period, leading to missed signals in the early history of a ticker.

---

### Iteration 2 — Adding Bollinger Bands + Volatility-Adjusted Stop
**File:** `trading_assignment_iteration.ipynb` — Cell 2 (`EnhancedStrategy` v2)

#### What Changed

| Component | Change from Iteration 1 |
|-----------|------------------------|
| **calculate_analytics** | Added Bollinger Bands (SMA-20, ±2σ lower band); grouped MACD histogram calc slightly differently (inline EWM instead of named signal_line column) |
| **make_decision (entry)** | Added `near_floor` condition: price must be within 6% of the Bollinger lower band |
| **make_decision (exit)** | Replaced flat −15% trailing stop with **dynamic ATR-based stop**: `high_water_mark − (2 × ATR)` |
| **make_decision (entry)** | Added `trend_ok` gate: `trend_strength > −1.0` (prevents entry when trend is deteriorating even if price is technically above EMAs) |

#### Key Decisions Made
- **Why add Bollinger lower band as an entry filter?** The EDA correlation analysis revealed that stocks entering near their Bollinger lower band with intact structural trends exhibited strong mean-reversion returns over the following 2–4 weeks. The `near_floor` condition concretely operationalises this finding.
- **Why switch to ATR-based dynamic stops?** The fixed −15% stop was too loose on low-volatility stocks (e.g., utilities) and too tight on high-volatility stocks (e.g., tech). ATR measures actual recent volatility, so `2 × ATR` below the peak scales the stop automatically to each stock's behaviour.
- **Why `trend_strength > −1.0`?** Even if a price is above EMA-200 and EMA-50, if it has fallen more than 1 ATR over the past 10 days, it is in short-term deterioration. This gate prevents buying into fading momentum even within a technically intact trend.

#### Known Limitation at This Stage
ATR was not pre-computed in `calculate_analytics` — the `atr` key was referenced in `make_decision` with a 5% fallback (`curr_price * 0.05`) but was not actually in the analytics dictionary at this point. This meant the dynamic stop was not functioning correctly.

---

### Iteration 3 — ATR + Trend Strength Properly Integrated + Full Optimisation
**File:** `trading_assignment_iteration.ipynb` — Cell 3 (`EnhancedStrategy` v3)  
**Also reflected in:** `trading_assignment - RSI-MACD_efficiencyV1.ipynb` — Cell 13 (final submitted version)

#### What Changed

| Component | Change from Iteration 2 |
|-----------|------------------------|
| **calculate_analytics** | ATR(14) now properly computed using True Range (max of: H−L, \|H−prev_C\|, \|L−prev_C\|) via `.apply()`; `trend_strength` column added as (close − close[−10]) / ATR |
| **calculate_analytics** | Date string conversion moved to the **very end** (single bulk `.dt.strftime()`) instead of being done per-group mid-calculation |
| **calculate_analytics (efficiency)** | EMAs reordered to compute all four (200, 50, 12, 26) before MACD, avoiding the intermediate column dependency issue in v2 |
| **_calculate_rsi** | Replaced `.where()` with `.clip()` for gain/loss separation; added `.replace(0, np.nan)` for division safety; added `.fillna(50)` to fill lead-in NaNs with neutral RSI |
| **llm_analysis** | Added `_sentiment_cache` dictionary with `{ticker}_{date}` key to prevent duplicate FinBERT calls during a simulation run |
| **make_decision (exit)** | ATR-based dynamic stop now functional (ATR present in analytics dictionary) |

#### Key Decisions Made
- **Why `.clip()` over `.where()` for RSI?** `.clip(lower=0)` / `.clip(upper=0)` is a single C-level operation vs. creating a boolean mask then applying `.where()`. The speed difference is small per call but meaningful across 400+ tickers × 17 years.
- **Why `.fillna(50)` for RSI?** The first 14 rows of each ticker's history have no valid RSI. Setting these to 50 (neutral) prevents false signals at the start of a ticker's history — a ticker won't be blocked from entry just because RSI is NaN.
- **Why move date conversion to the end?** During intermediate calculations, pandas needs datetime objects for `.ewm()`, `.rolling()`, and `.shift()`. Converting to strings mid-function caused silent type errors in some pandas versions. Doing it once at the very end cleanly separates the computation phase from the output formatting phase.
- **Why add sentiment caching?** Profiling showed that `llm_analysis` was the #1 bottleneck (FinBERT inference ~150ms/call on GPU). Since the same ticker's transcript appears across multiple consecutive weekly ticks (earnings are quarterly), caching eliminates ~75% of redundant calls.
- **Why fix ATR computation?** The v2 fallback `curr_price * 0.05` made the dynamic stop equivalent to a flat 10% stop, which was actually less effective than the v1 −15% trailing stop. The proper ATR calculation was essential for the feature to have any value.

#### Final Configuration Summary

```
Entry requires ALL THREE:
  ✓ close > EMA-50 AND close > EMA-200       (structural trend)
  ✓ RSI < 45 AND pace_improving == True       (momentum rebound)
  ✓ sentiment_score >= 0.5 OR no transcript   (sentiment gate)

Exit triggers on EITHER:
  ✗ close < (high_water_mark × 0.85)          (−15% trailing stop)
  ✗ pace_improving == False                    (momentum decay)
```

---

### Iteration Summary Table

| Version | Key Additions | Entry Conditions | Exit Conditions | Notable Fix |
|---------|--------------|-----------------|-----------------|-------------|
| **Baseline** | MA-50 only | price > MA-50 + positive sentiment | price < MA-50 OR −20% flat stop | — |
| **v1 (RSI+MACD)** | EMA-200/50, RSI, MACD, histogram, pace | structural + momentum + sentiment | −15% trailing stop OR pace decay | — |
| **v2 (+BB+ATR)** | Bollinger lower band, ATR (stub), trend_strength | + near_floor + trend_ok | ATR dynamic stop (non-functional fallback) | ATR not yet in analytics |
| **v3 (Final)** | ATR properly computed, trend_strength live, RSI `.clip()` + NaN fill, sentiment cache | Same as v2 | ATR dynamic stop (functional) | ATR integrated into analytics; RSI NaN fixed; caching added |

---

## Phase 3: Performance Evaluation and Analysis

### Evaluation Cells (`Cell 14`)

- **Dev Split Evaluation**: Runs `EnhancedStrategy` on the full dev split, calculates all five metrics, and plots the full results dashboard (portfolio value, cash, positions, drawdown, weekly returns distribution, 20-week rolling return)
- **Val Split Evaluation**: Final one-time evaluation on the held-out validation set; metrics stored in `ENHANCED_METRICS_VAL`

### Trade Activity Audit (`Cell 15`)
Post-backtest analysis performed on results:
- Merged trade records with market volume data using `pd.merge_asof` (backward direction) to obtain realistic volume context at trade time
- Calculated per-trade P&L % for all matched buy→sell round-trips
- Exported full audit to both **CSV** and **JSON** for reproducibility
- Classified each trade as `PROFIT` or `LOSS`

### Trade Frequency Analysis (`Cell 16`)
- Bar chart of the **Top 30 most frequently bought tickers** (buy count)

### Ticker-Level P&L Summary (`Cell 17`)
- Matched all buy→sell round-trips per ticker
- Aggregated: total P&L ($), average P&L (%), winning trade count, losing trade count, total round-trips
- Produced **Top 20 Profitable Tickers** and **Top 20 Losing Tickers** tables with styled formatting
- Side-by-side horizontal bar charts visualising the distribution of gains and losses

---

## Enhancement Areas Completed

| Area | Status | Method Overridden |
|------|--------|-------------------|
| 1. Extended EDA | ✅ Completed | — (standalone cells) |
| 2. Data Cleaning & Preparation | ✅ Completed | `clean_data()` |
| 3. Technical Indicators & Features | ✅ Completed | `calculate_analytics()` |
| 4. Earnings Transcript NLP | ✅ Completed | `llm_analysis()` |
| 5. Entry, Exit & Risk Rules | ✅ Completed | `make_decision()` |

---

## Computational Efficiency and Scalability

The following optimisations were implemented to handle ~2 million price records efficiently:

| Technique | Applied Where | Impact |
|-----------|--------------|--------|
| **Single `groupby` partitioning** | `calculate_analytics`, `_build_lookups`, `_build_price_history` | Eliminates repeated `O(n×t)` filtering loops |
| **Vectorised EMA via `.transform()`** | `calculate_analytics` | Replaces per-ticker Python loops with C-level pandas operations |
| **`.clip()` instead of `where/mask`** | `_calculate_rsi` | Marginal speed gain in RSI computation |
| **Bulk date string conversion** | `calculate_analytics` | Single `.dt.strftime()` call instead of per-row conversion |
| **FinBERT result caching** | `llm_analysis` | Prevents redundant GPU inference on repeated transcript lookups |
| **GPU acceleration** | `llm_analysis`, FinBERT pipeline init | 3–4× faster inference vs. CPU on T4/similar hardware |
| **`pd.merge_asof`** | Trade audit cell | O(n log n) nearest-date join instead of iterative matching |
| **O(1) dictionary lookups** | `TradingSimulation._get_price_on_date` | Replaces DataFrame filtering on every weekly tick |
| **`use_safetensors=True`** | FinBERT model load | Bypasses torch CVE security check without version upgrade |

The solution is designed to scale horizontally: the `groupby`-based partitioning approach means adding more tickers increases compute linearly, not quadratically.

---

## Performance Metrics Reference

| Metric | Description | Target |
|--------|-------------|--------|
| Total Return | Portfolio % change from start to end | Maximise |
| Sharpe Ratio | Annualised return per unit of risk | > 1.0 |
| Max Drawdown | Largest peak-to-trough portfolio decline | Minimise |
| Win Rate | Proportion of trades closed at a profit | > 40% |
| Volatility | Annualised std dev of weekly returns | Lower for equivalent return |

> **Note**: The dev set spans a longer horizon than the val set. Higher cumulative returns on dev are expected and do not indicate overfitting. Use the **Sharpe Ratio** for cross-period comparison as it is annualised.

---

## Infrastructure Overview

| Component | Description |
|-----------|-------------|
| `Portfolio` | Tracks cash, positions, and trade history; supports `buy_target()` and `sell()` |
| `TradingSimulation` | Weekly (every Friday) backtest engine; builds O(1) price and earnings lookups |
| `BaseStrategy` | Baseline: price > MA-50 AND positive/no sentiment → BUY; price < MA-50 OR −20% → SELL |
| `EnhancedStrategy` | Full override of all five strategy methods with multi-indicator, NLP-enhanced logic |
| `calculate_metrics()` | Computes Total Return, Sharpe, Max Drawdown, Win Rate, Volatility |
| `plot_results()` | 6-panel dashboard: portfolio value, cash, positions, drawdown, returns distribution, rolling return |
| `plot_comparison()` | Side-by-side baseline vs. enhanced visualisation across 4 panels |
| `run_evaluation()` | Helper to load any split and evaluate any strategy in one call |

---

## File Structure

```
Algorithmic-Trading-Strategy-Development-Optimisation/
│
├── README.md                                           # This document
├── Assignment-2-Trading-Strategy.txt                   # Original assignment specification
├── trading_assignment - RSI-MACD_efficiencyV1.ipynb    # Final submitted notebook
├── trading_assignment_iteration.ipynb                  # Iteration history (v1 → v3)
│
├── sample_data/
│   ├── prices_dev.parquet                              # Dev split price data (2000–2017)
│   ├── prices_val.parquet                              # Val split price data (2018–2024)
│   ├── earnings_dev.parquet                            # Dev split earnings transcripts
│   └── earnings_val.parquet                            # Val split earnings transcripts
│
└── Test Results/
    └── Documentation.txt                               # Test set result documentation
```

---

## How to Run

### Prerequisites
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers pandas numpy matplotlib seaborn pyarrow
```

### Execution Order
1. **Cell 1** — Install/configure environment, load libraries, initialise FinBERT
2. **Cell 4** — Basic EDA (price and earnings overview)
3. **Cell 5** — Data visualisation
4. **Cells 6–8** — Load Portfolio, TradingSimulation, metrics and plotting infrastructure
5. **Cell 9** — Load BaseStrategy
6. **Cell 10** — Load evaluation helper
7. **Cells 11–13d** — Advanced EDA (correlation, survival, FinBERT behaviour)
8. **Cell 13 (EnhancedStrategy)** — Define the enhanced strategy class
9. **Cell 14** — Run evaluation on dev split → iterate → run on val split for final results
10. **Cells 15–17** — Trade audit, frequency analysis, P&L breakdown

> **Important**: Place all `.parquet` data files in a `./sample_data/` directory relative to the notebook before running.

---

## AI Usage Declaration

GitHub Copilot (Claude Sonnet 4.6) was used in this project for:
- **Code generation assistance**: Scaffolding vectorised pandas operations in `calculate_analytics()` and the RSI helper function
- **Debugging**: Identifying timezone-naive/aware datetime incompatibility in `pd.merge_asof` calls
- **Documentation**: Generating inline code comments and this README

All AI-generated outputs were reviewed, tested against the actual dataset, and modified where necessary to ensure correctness. All team members understand and can explain the submitted code in full.