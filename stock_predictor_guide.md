# Stock Return Predictor — Master Build Guide

> A hand-held walkthrough from papers → features → backend → frontend → Docker.
> Written for an ML student comfortable with Python/pandas/sklearn.

---

## 0. Paper Links

| Paper | Link |
|---|---|
| Welch & Goyal (2008) | https://doi.org/10.1093/rfs/hhn010 |
| Gu, Kelly & Xiu (2020) | https://doi.org/10.1093/rfs/hhaa009 |

Both are open-access. The GKX paper's internet appendix (with the full 94-characteristic list) is linked on the Oxford RFS page next to the paper.

---

## 1. Paper 1: Welch & Goyal (2008) — Plain English

### What they did
For 50+ years, finance researchers had been claiming that simple ratios (dividend/price, earnings/price, interest rate spreads, etc.) can predict next year's stock market return. WG rounded up **14 of these "predictor" variables** and asked: *if you had actually used these in real time, back before you knew the future, would they have worked?*

The answer was a resounding **no**.

### The key distinction: in-sample vs. out-of-sample
This is the #1 concept to internalize for your project.

- **In-sample (IS)**: You fit a model on ALL available data, then measure how well it fits that same data. Like a teacher who writes the exam, then grades themselves on it. Of course it looks good.
- **Out-of-sample (OOS)**: You fit on data up to time T, then predict time T+1 — data the model has never seen. This is what a real investor could actually do.

WG showed that almost every "predictor" looks great in-sample but **fails out-of-sample**. The models were curve-fit to history, not capturing genuine signal.

### The 14 macro predictors
These are the variables WG test. You'll implement them as the "Baseline" tab:

| Abbreviation | Variable | What it measures |
|---|---|---|
| `d/p` | Log dividend-price ratio | Dividend yield (smoothed) |
| `d/y` | Log dividend yield | Dividends / lagged price |
| `e/p` | Log earnings-price ratio | Valuation (like inverse P/E) |
| `d/e` | Log dividend payout ratio | What fraction of earnings are paid out |
| `b/m` | Book-to-market ratio (Dow) | Value vs. growth signal |
| `ntis` | Net equity expansion | How much new stock is being issued |
| `tbl` | T-bill rate | Short-term interest rate |
| `lty` | Long-term bond yield | 10yr govt bond yield |
| `ltr` | Long-term bond return | Bond market performance |
| `tms` | Term spread | `lty - tbl` (yield curve slope) |
| `dfy` | Default yield spread | BAA - AAA bond yields |
| `dfr` | Default return spread | Corp bond return - govt bond return |
| `infl` | Inflation | CPI change |
| `svar` | Stock variance | Sum of squared daily S&P returns |

### The OOS R² formula
This is WG's core benchmark metric. Here's the math, step by step:

**Setup**: You're predicting the equity premium `r_t` at each time step.
- The **naive model** (null hypothesis): just predict the historical average of past returns.
- The **regression model**: fit OLS on a predictor, use it to predict.

**Mean Squared Error for each model:**
```
MSE_naive  = (1/T) * Σ (r_t - r̄_{t-1})²     # r̄ is the expanding historical mean
MSE_model  = (1/T) * Σ (r_t - ŷ_t)²           # ŷ is the OLS prediction using predictor x_{t-1}
```

**OOS R²:**
```
R²_OOS = 1 - (MSE_model / MSE_naive)
```

**Interpretation:**
- `R²_OOS > 0`: your model beats the naive historical average → has real signal
- `R²_OOS = 0`: your model is exactly as good as just guessing the historical mean
- `R²_OOS < 0`: your model is WORSE than just using the historical average → useless

WG found that almost all 14 variables have **negative OOS R²** over the last 30 years of their sample. This is the baseline your ML model (from Paper 2) has to beat.

### Why this matters for your webapp
The Baseline tab will implement exactly this: fetch S&P 500 data + FRED macro data, compute rolling OOS R² for each of the 14 predictors, and display whether the historical macro models even work for the ticker the user selected. It's the "null hypothesis" your ML model is trying to beat.

---

## 2. Paper 2: Gu, Kelly & Xiu (2020) — Plain English

### What they did
GKX took the WG critique seriously. If simple macro predictors don't work, what about **machine learning on a huge feature set at the individual stock level**?

They studied nearly 30,000 individual US stocks over 60 years (1957–2016), building a feature set of **94 stock characteristics** (plus macro interactions and industry dummies → 900+ total signals) and ran every major ML method through it: OLS, Lasso/ElasticNet, PCA, PLS, regression trees, random forests, gradient boosting, and neural networks with up to 5 hidden layers.

### What they found
1. **OLS fails with many predictors** — OOS R² goes deeply negative. Too many parameters, too little data.
2. **Linear methods with regularization (Lasso/ElasticNet) work** — positive OOS R² (~0.11% per month).
3. **Trees and neural networks win** — OOS R² of 0.33%–0.40% per month. The key: they capture **nonlinear interactions** between predictors.
4. **The winning signals** are variations on momentum, liquidity (market cap, dollar volume, bid-ask spread), and volatility (realized vol, idiosyncratic vol, beta).
5. **Economic gains are large**: a neural network that times the S&P 500 achieves a Sharpe ratio of 0.77 vs 0.51 for buy-and-hold.

### Why tiny R² still matters
0.33% monthly R² sounds tiny. Here's the intuition:

Stock returns are ~85% noise (unpredictable news). The signal-to-noise ratio is extremely low. A model that explains even 0.33% of the variance out-of-sample is actually capturing real, economically meaningful signal — enough to build profitable strategies on top of.

Think of it like weather forecasting: knowing there's a 55% vs 45% chance of rain sounds barely useful, but if you're running an outdoor event business and trade on that edge every day, it compounds to real profits.

### The feature set you'll implement (simplified to ~20 key features)
GKX's full 94-characteristic list requires CRSP/Compustat (institutional data). For yfinance, you'll implement the **most important subset**:

**Momentum features** (consistently the #1 signal):
- 1-month lagged return (short-term reversal)
- 12-month cumulative return, skipping most recent month (momentum)
- 6-month cumulative return

**Volatility features**:
- 1-month realized volatility (std of daily returns)
- 3-month realized volatility
- Idiosyncratic volatility (residual from market regression)

**Liquidity/size features**:
- Log market cap (price × shares outstanding)
- Dollar volume (rolling 20-day average)
- Amihud illiquidity ratio

**Technical/price features**:
- Price-to-52-week-high ratio
- Moving average crossover signals (MA20/MA50, MA50/MA200)
- RSI (14-day relative strength index)

**Macro interaction features** (from WG):
- Each stock feature × term spread
- Each stock feature × T-bill rate

### The train/val/test split — no data leakage
GKX use a **strict temporal split** (not random):
- **Training**: 1957–1974 (fit model parameters)
- **Validation**: 1975–1986 (tune hyperparameters — e.g., regularization strength)
- **Test**: 1987–2016 (evaluate OOS R², never touched during training)

For your webapp with a single ticker (say 2010–2024):
- **Training**: 2010–2018
- **Validation**: 2019–2021
- **Test**: 2022–2024

**Critical rule**: features at time T can only use data available at or before time T. If you compute a 12-month momentum at 2024-01-01, it must use only returns from 2023-01-01 to 2022-01-31 (the most recent month is excluded — it's the target variable).

---

## 3. Project Architecture

```
stock-predictor/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── data/
│   │   ├── fetcher.py       # yfinance + FRED data
│   │   └── features.py      # Feature engineering
│   ├── models/
│   │   ├── baseline.py      # WG OOS R² benchmark
│   │   ├── ml_pipeline.py   # Ridge → RF → NN pipeline
│   │   ├── dcf.py           # DCF valuation
│   │   └── monte_carlo.py   # GBM simulation
│   └── requirements.txt
├── frontend/
│   └── app.py               # Streamlit app (5 tabs)
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 4. Part 1: Welch & Goyal OOS R² in Python

### Step 1: Fetch the data

```python
# backend/data/fetcher.py
import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
import numpy as np
from datetime import datetime

def fetch_sp500_and_macro(start="1990-01-01", end=None):
    """
    Fetch S&P 500 returns + macro predictors from FRED.
    Returns a monthly DataFrame.
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    
    # --- S&P 500 monthly returns ---
    sp500 = yf.download("^GSPC", start=start, end=end, interval="1mo", auto_adjust=True)
    sp500 = sp500["Close"].rename("sp500_price")
    sp500.index = sp500.index.to_period("M").to_timestamp()
    sp500_ret = sp500.pct_change().rename("mkt_ret")
    
    # --- Risk-free rate (3-month T-bill from FRED) ---
    # TB3MS = 3-Month Treasury Bill Secondary Market Rate (monthly, annualized %)
    tbill = web.DataReader("TB3MS", "fred", start, end)
    tbill = tbill.resample("MS").last() / 100 / 12  # convert to monthly decimal
    tbill.columns = ["rf"]
    
    # --- Equity premium = market return - risk-free rate ---
    df = sp500_ret.to_frame().join(tbill, how="inner")
    df["eq_premium"] = df["mkt_ret"] - df["rf"]
    
    # --- Macro predictors from FRED ---
    fred_series = {
        "DGS10": "lty",         # 10-year Treasury yield
        "TB3MS": "tbl",         # 3-month T-bill (already have, but keep for predictor)
        "BAMLC0A4CBBB": "baa",  # BAA corporate bond yield
        "BAMLC0A1CAAA": "aaa",  # AAA corporate bond yield
        "CPIAUCSL": "cpi",      # CPI (for inflation)
    }
    
    macro_raw = web.DataReader(list(fred_series.keys()), "fred", start, end)
    macro_raw = macro_raw.resample("MS").last()
    macro_raw = macro_raw.rename(columns=fred_series)
    macro_raw = macro_raw / 100  # convert % to decimal where applicable
    
    # Derived predictors
    macro_raw["tms"] = macro_raw["lty"] - macro_raw["tbl"]   # term spread
    macro_raw["dfy"] = macro_raw["baa"] - macro_raw["aaa"]   # default yield spread
    macro_raw["infl"] = macro_raw["cpi"].pct_change()         # inflation rate
    # NOTE: FRED inflation is lagged 1 month before use (info available next month)
    macro_raw["infl"] = macro_raw["infl"].shift(1)
    
    # S&P 500 stock variance: sum of squared DAILY returns per month
    sp500_daily = yf.download("^GSPC", start=start, end=end, interval="1d", auto_adjust=True)
    sp500_daily = sp500_daily["Close"].pct_change().dropna()
    sp500_daily.index = pd.to_datetime(sp500_daily.index)
    svar = sp500_daily.resample("MS").apply(lambda x: (x**2).sum()).rename("svar")
    
    df = df.join(macro_raw[["tbl", "lty", "tms", "dfy", "infl"]], how="left")
    df = df.join(svar, how="left")
    df = df.dropna()
    
    return df
```

### Step 2: The OOS R² calculation

```python
# backend/models/baseline.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def compute_oos_r2(df: pd.DataFrame, predictor: str, 
                   target: str = "eq_premium",
                   min_train_size: int = 60) -> dict:
    """
    Implements the Welch & Goyal (2008) OOS R² benchmark.
    
    Key idea: 
    - At each time t, train OLS on ALL data up to t-1 (expanding window)
    - Predict r_t using x_{t-1} (predictor lagged 1 period — NO look-ahead!)
    - Compare to the naive model: predict r_t = historical average of past returns
    
    Args:
        df: DataFrame with target and predictor columns
        predictor: column name of the predictor variable (x)
        target: column name of what we're predicting (equity premium)
        min_train_size: months of training data before we start predicting (60 = 5 years)
    
    Returns:
        dict with OOS R², cumulative MSE curve for plotting, actual vs predicted series
    """
    df = df[[target, predictor]].dropna().copy()
    
    # CRITICAL: lag the predictor by 1 period
    # x_{t-1} predicts r_t — this prevents look-ahead bias
    df["x_lagged"] = df[predictor].shift(1)
    df = df.dropna()
    
    n = len(df)
    
    # Storage for OOS predictions
    preds_model = []
    preds_naive = []
    actuals = []
    
    for t in range(min_train_size, n):
        # Training data: everything BEFORE time t
        train = df.iloc[:t]
        
        # --- Naive model: expanding historical mean ---
        naive_pred = train[target].mean()
        
        # --- OLS model: fit on training data, predict at time t ---
        X_train = train[["x_lagged"]].values
        y_train = train[target].values
        
        reg = LinearRegression().fit(X_train, y_train)
        
        # Predictor AT time t is x_t, but it was set up as "x_{t-1}" in lagged column
        # So df["x_lagged"].iloc[t] is already x_{t-1} for the prediction of r_t
        X_pred = df[["x_lagged"]].iloc[[t]].values
        model_pred = reg.predict(X_pred)[0]
        
        preds_model.append(model_pred)
        preds_naive.append(naive_pred)
        actuals.append(df[target].iloc[t])
    
    # Convert to arrays
    actuals = np.array(actuals)
    preds_model = np.array(preds_model)
    preds_naive = np.array(preds_naive)
    
    # MSE for each model
    errors_model = actuals - preds_model
    errors_naive = actuals - preds_naive
    
    mse_model = np.mean(errors_model ** 2)
    mse_naive = np.mean(errors_naive ** 2)
    
    # OOS R² (Campbell & Thompson 2008 definition, same as WG)
    oos_r2 = 1 - (mse_model / mse_naive)
    
    # Cumulative OOS R² over time (for plotting the evolution)
    cum_mse_model = np.cumsum(errors_model ** 2)
    cum_mse_naive = np.cumsum(errors_naive ** 2)
    cum_oos_r2 = 1 - (cum_mse_model / cum_mse_naive)
    
    # Build result DataFrame for plotting
    result_index = df.index[min_train_size:]
    results_df = pd.DataFrame({
        "actual": actuals,
        "pred_model": preds_model,
        "pred_naive": preds_naive,
        "cum_oos_r2": cum_oos_r2,
    }, index=result_index)
    
    return {
        "predictor": predictor,
        "oos_r2": round(oos_r2, 6),
        "oos_r2_pct": round(oos_r2 * 100, 4),
        "beats_naive": oos_r2 > 0,
        "n_oos_periods": len(actuals),
        "results_df": results_df,
    }


def run_all_wg_predictors(df: pd.DataFrame) -> pd.DataFrame:
    """Run WG benchmark for all available predictors, return summary table."""
    predictors = [c for c in ["tbl", "lty", "tms", "dfy", "infl", "svar"] if c in df.columns]
    
    rows = []
    for pred in predictors:
        result = compute_oos_r2(df, pred)
        rows.append({
            "Predictor": pred,
            "OOS R² (%)": result["oos_r2_pct"],
            "Beats Naive?": "✅ Yes" if result["beats_naive"] else "❌ No",
            "N OOS Periods": result["n_oos_periods"],
        })
    
    return pd.DataFrame(rows).sort_values("OOS R² (%)", ascending=False)
```

**What the math is doing, step by step:**

At each month T:
1. Take all data from month 1 to month T-1 (the "past")
2. Fit a regression: `equity_premium = α + β × predictor`
3. Plug in last month's predictor value → get predicted equity premium for month T
4. Also compute the expanding historical average of equity premiums (the naive baseline)
5. Record both predictions and the actual outcome

At the end, you have two series of errors. The OOS R² formula `1 - MSE_model/MSE_naive` tells you if the model's errors are smaller than the naive errors. A positive number means the model adds value.

---

## 5. Part 2: Building the Gu, Kelly & Xiu Feature Set

```python
# backend/data/features.py
import pandas as pd
import numpy as np
import yfinance as yf

def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV data for a single ticker."""
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    return df


def build_gkx_features(df: pd.DataFrame, macro_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build GKX-inspired feature set from daily price data.
    
    CRITICAL DESIGN RULE:
    Every feature computed here uses ONLY past data.
    The target variable (next-period return) is constructed SEPARATELY.
    Never let any feature "see" future prices.
    
    Args:
        df: Daily OHLCV DataFrame (columns: open, high, low, close, volume)
        macro_df: Optional monthly macro DataFrame from WG fetcher
    
    Returns:
        Daily feature DataFrame (will be merged with target returns for modeling)
    """
    feat = pd.DataFrame(index=df.index)
    
    # =========================================================
    # 1. RETURN FEATURES — Momentum & Reversal
    # =========================================================
    # Daily log returns (more stable than simple returns for multiplication)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    
    # Short-term reversal: last month's return (days 2–21)
    # In GKX this is "ret_1_0" — 1-month lag, skipping most recent day
    # Why skip most recent? Bid-ask bounce causes spurious negative autocorrelation
    feat["ret_1m"] = df["close"].pct_change(21).shift(1)  # lag 1 day
    
    # Medium-term momentum: 12-month return, skip last month (days 22–252)
    # GKX calls this "ret_12_1" — one of the strongest predictors
    feat["mom_12_1"] = (df["close"].shift(22) / df["close"].shift(252)) - 1
    
    # 6-month momentum (days 22–126)
    feat["mom_6_1"] = (df["close"].shift(22) / df["close"].shift(126)) - 1
    
    # 3-month momentum
    feat["mom_3_1"] = (df["close"].shift(22) / df["close"].shift(63)) - 1
    
    # =========================================================
    # 2. VOLATILITY FEATURES
    # =========================================================
    # 1-month realized volatility: std of daily returns over past 21 days
    # Annualized by multiplying by sqrt(252)
    feat["vol_1m"] = log_ret.rolling(21).std() * np.sqrt(252)
    
    # 3-month realized volatility
    feat["vol_3m"] = log_ret.rolling(63).std() * np.sqrt(252)
    
    # 6-month realized volatility
    feat["vol_6m"] = log_ret.rolling(126).std() * np.sqrt(252)
    
    # Volatility ratio (short-term vol vs long-term vol — measures vol regime)
    feat["vol_ratio"] = feat["vol_1m"] / (feat["vol_3m"] + 1e-8)
    
    # Maximum daily return in past month (max_ret in GKX — lottery stock signal)
    feat["max_ret"] = log_ret.rolling(21).max()
    
    # =========================================================
    # 3. MARKET BETA (estimated via rolling OLS on market returns)
    # =========================================================
    # We'll use SPY as the market proxy
    # Note: In a real pipeline you'd pass market returns in; here we embed the logic
    try:
        spy = yf.download("SPY", start=df.index[0] - pd.Timedelta(days=400), 
                          end=df.index[-1] + pd.Timedelta(days=1), 
                          auto_adjust=True, progress=False)["Close"]
        mkt_ret = spy.pct_change().reindex(df.index)
        
        # Rolling 252-day beta = cov(stock, market) / var(market)
        cov_roll = log_ret.rolling(252).cov(mkt_ret.pct_change())
        var_roll = mkt_ret.pct_change().rolling(252).var()
        feat["beta"] = cov_roll / (var_roll + 1e-10)
        feat["beta_sq"] = feat["beta"] ** 2  # GKX includes beta² as separate signal
        
        # Idiosyncratic volatility: std of residuals from market regression
        # Residual = stock_return - beta * market_return
        feat["ivol"] = (log_ret - feat["beta"] * mkt_ret.pct_change()).rolling(21).std() * np.sqrt(252)
    except Exception:
        feat["beta"] = np.nan
        feat["beta_sq"] = np.nan
        feat["ivol"] = np.nan
    
    # =========================================================
    # 4. LIQUIDITY FEATURES
    # =========================================================
    # Dollar volume: price × daily volume (measures how easy it is to trade)
    dollar_vol = df["close"] * df["volume"]
    # Rolling 20-day average dollar volume, log-transformed (heavy right tail)
    feat["log_dolvol"] = np.log(dollar_vol.rolling(20).mean() + 1)
    
    # Amihud (2002) illiquidity: |return| / dollar_volume
    # Higher = less liquid (large price impact per dollar traded)
    feat["illiq"] = (log_ret.abs() / (dollar_vol + 1)).rolling(21).mean() * 1e6
    
    # Turnover: volume / shares outstanding proxy
    # (yfinance doesn't give shares outstanding directly, use volume ratio instead)
    feat["turn"] = df["volume"].rolling(20).mean() / (df["volume"].rolling(252).mean() + 1)
    
    # =========================================================
    # 5. PRICE-LEVEL FEATURES
    # =========================================================
    # 52-week high ratio: how close is current price to 52-week high?
    # High ratio → recent outperformance → momentum signal
    feat["high52w"] = df["close"] / df["high"].rolling(252).max()
    
    # Log of price (low-priced stocks behave differently)
    feat["log_price"] = np.log(df["close"] + 1)
    
    # =========================================================
    # 6. TECHNICAL INDICATOR FEATURES
    # =========================================================
    # Moving average crossover signals
    ma20 = df["close"].rolling(20).mean()
    ma50 = df["close"].rolling(50).mean()
    ma200 = df["close"].rolling(200).mean()
    
    # Price relative to moving averages (normalized by MA to be scale-invariant)
    feat["price_to_ma20"] = df["close"] / (ma20 + 1e-8) - 1
    feat["price_to_ma50"] = df["close"] / (ma50 + 1e-8) - 1
    feat["price_to_ma200"] = df["close"] / (ma200 + 1e-8) - 1
    feat["ma20_to_ma50"] = ma20 / (ma50 + 1e-8) - 1
    feat["ma50_to_ma200"] = ma50 / (ma200 + 1e-8) - 1
    
    # RSI (14-day Relative Strength Index)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    feat["rsi14"] = 100 - (100 / (1 + rs))
    
    # Bollinger Band position: (price - lower band) / (upper - lower)
    bb_mid = ma20
    bb_std = df["close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    feat["bb_pct"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-8)
    
    # =========================================================
    # 7. MACRO INTERACTION FEATURES
    # =========================================================
    # GKX interacts each stock characteristic with macro variables
    # This lets the model learn "momentum works better when rates are rising" etc.
    if macro_df is not None:
        # Resample macro to daily (forward-fill — we only know macro at month-end)
        macro_daily = macro_df[["tms", "tbl"]].resample("D").ffill().reindex(df.index)
        
        # Interaction: momentum × term spread
        feat["mom_x_tms"] = feat["mom_12_1"] * macro_daily["tms"]
        # Interaction: volatility × T-bill rate
        feat["vol_x_tbl"] = feat["vol_1m"] * macro_daily["tbl"]
        # Raw macro levels as features
        feat["tms"] = macro_daily["tms"]
        feat["tbl"] = macro_daily["tbl"]
    
    # =========================================================
    # FINAL CLEANUP
    # =========================================================
    # Winsorize at 1st/99th percentile to remove outliers
    # (GKX do this cross-sectionally; for single stock, we do it time-series)
    for col in feat.columns:
        q01 = feat[col].quantile(0.01)
        q99 = feat[col].quantile(0.99)
        feat[col] = feat[col].clip(lower=q01, upper=q99)
    
    # Cross-sectional rank normalization (GKX do this to make features comparable)
    # Maps each feature to [-1, 1] based on its rank in the historical distribution
    # For a single stock time series, we use expanding-window rank (no look-ahead!)
    for col in feat.columns:
        # Rank up to current date, scaled to [-1, 1]
        feat[col] = feat[col].expanding().rank(pct=True) * 2 - 1
    
    return feat


def build_target_variable(df: pd.DataFrame, horizon_days: int = 21) -> pd.Series:
    """
    Build the prediction target: forward return over `horizon_days`.
    
    CRITICAL: This is the FUTURE return — it must NEVER appear as a feature.
    The modeling code must align features at time T with target at time T+horizon.
    
    Args:
        df: daily price DataFrame
        horizon_days: prediction horizon (21 = ~1 month, default)
    
    Returns:
        Series of forward returns
    """
    # Forward return: what will the price be in `horizon_days` days?
    fwd_return = df["close"].pct_change(horizon_days).shift(-horizon_days)
    return fwd_return.rename(f"fwd_ret_{horizon_days}d")


def create_modeling_dataset(df: pd.DataFrame, feat: pd.DataFrame, 
                             horizon_days: int = 21) -> pd.DataFrame:
    """
    Join features and target, drop NaNs, and create a clean modeling dataset.
    The alignment: features at time T predict return from T to T+horizon.
    """
    target = build_target_variable(df, horizon_days)
    dataset = feat.join(target)
    # Drop the last `horizon_days` rows (no future data to compute target)
    dataset = dataset.dropna()
    return dataset
```

---

## 6. Part 3: The ML Pipeline (Ridge → Random Forest → NN)

```python
# backend/models/ml_pipeline.py
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.sklearn
import mlflow.pytorch

# ======================================================
# SPLIT LOGIC — THE MOST IMPORTANT PART
# ======================================================

def temporal_train_val_test_split(dataset: pd.DataFrame, 
                                   val_frac: float = 0.15,
                                   test_frac: float = 0.20):
    """
    Strict temporal split — NO shuffling, NO random state.
    
    Why? If you shuffle, training data will include rows from the "future"
    relative to test rows. The model will have seen signals that leaked
    information about the test period. This is look-ahead bias.
    
    Timeline: |----TRAIN----|--VAL--|--TEST--|
                  ~65%         ~15%    ~20%
    
    Returns: train, val, test DataFrames (each with features + target column)
    """
    n = len(dataset)
    train_end = int(n * (1 - val_frac - test_frac))
    val_end = int(n * (1 - test_frac))
    
    train = dataset.iloc[:train_end]
    val = dataset.iloc[train_end:val_end]
    test = dataset.iloc[val_end:]
    
    print(f"Train: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} obs)")
    print(f"Val:   {val.index[0].date()} → {val.index[-1].date()} ({len(val)} obs)")
    print(f"Test:  {test.index[0].date()} → {test.index[-1].date()} ({len(test)} obs)")
    
    return train, val, test


def split_xy(df: pd.DataFrame, target_col: str):
    """Split a DataFrame into X (features) and y (target)."""
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    return X, y


# ======================================================
# MODEL 1: RIDGE REGRESSION
# ======================================================

def train_ridge(train, val, target_col: str = "fwd_ret_21d"):
    """
    Ridge Regression: linear model with L2 regularization.
    
    Why Ridge and not plain OLS?
    With many correlated features, OLS overfits badly.
    Ridge adds a penalty: minimize MSE + λ * sum(β²)
    This shrinks coefficients toward zero, preventing overfit.
    
    λ (alpha in sklearn) controls the strength of shrinkage:
    - λ → 0: approaches OLS (overfit)
    - λ → ∞: all coefficients → 0 (underfit, predicts the mean)
    
    We find the best λ on the validation set.
    """
    X_train, y_train = split_xy(train, target_col)
    X_val, y_val = split_xy(val, target_col)
    
    # Standardize features (important for Ridge — features on same scale)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)  # USE TRAINING STATS — don't fit on val!
    
    # Try a range of alpha values on the validation set
    best_alpha, best_r2 = None, -np.inf
    for alpha in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        ridge = Ridge(alpha=alpha).fit(X_train_s, y_train)
        val_preds = ridge.predict(X_val_s)
        r2 = r2_score(y_val, val_preds)
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha
    
    # Refit with best alpha on train+val combined (standard practice)
    X_tv = scaler.fit_transform(np.vstack([X_train, X_val]))
    y_tv = np.concatenate([y_train, y_val])
    final_ridge = Ridge(alpha=best_alpha).fit(X_tv, y_tv)
    
    print(f"Ridge best alpha: {best_alpha}, Val R²: {best_r2:.6f}")
    
    return final_ridge, scaler, {"best_alpha": best_alpha, "val_r2": best_r2}


# ======================================================
# MODEL 2: RANDOM FOREST
# ======================================================

def train_random_forest(train, val, target_col: str = "fwd_ret_21d"):
    """
    Random Forest: ensemble of decision trees.
    
    Why RF for financial data?
    - Captures nonlinear relationships (e.g., "momentum only works when vol is low")
    - Resistant to outliers
    - Built-in feature importance scores
    
    Key hyperparameters:
    - n_estimators: number of trees (more = better but slower; 200 is a good start)
    - max_features: fraction of features to try at each split (controls tree diversity)
    - max_depth: how deep each tree grows (shallower = less overfit)
    - min_samples_leaf: minimum observations per leaf (prevents tiny, noisy leaves)
    
    GKX found that in financial data, trees with few leaves (< 6) work best —
    the signal is weak, deep trees just memorize noise.
    """
    X_train, y_train = split_xy(train, target_col)
    X_val, y_val = split_xy(val, target_col)
    
    feature_names = [c for c in train.columns if c != target_col]
    
    # Grid search on validation set
    best_r2, best_params, best_model = -np.inf, {}, None
    
    for n_est in [100, 200]:
        for max_depth in [3, 5, 8]:
            for max_feat in [0.3, 0.5, "sqrt"]:
                rf = RandomForestRegressor(
                    n_estimators=n_est,
                    max_depth=max_depth,
                    max_features=max_feat,
                    min_samples_leaf=50,  # Important: prevents overfit on small samples
                    n_jobs=-1,
                    random_state=42,
                ).fit(X_train, y_train)
                
                r2 = r2_score(y_val, rf.predict(X_val))
                if r2 > best_r2:
                    best_r2 = r2
                    best_params = {"n_estimators": n_est, "max_depth": max_depth, 
                                   "max_features": max_feat}
                    best_model = rf
    
    # Feature importance from the best RF model
    importances = pd.Series(
        best_model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)
    
    print(f"RF best params: {best_params}, Val R²: {best_r2:.6f}")
    print("\nTop 5 features:")
    print(importances.head())
    
    return best_model, {"val_r2": best_r2, "params": best_params, 
                        "feature_importances": importances}


# ======================================================
# MODEL 3: NEURAL NETWORK (PyTorch)
# ======================================================

class StockNN(nn.Module):
    """
    3-layer feedforward NN matching GKX's best architecture.
    
    Architecture: Input → 32 → 16 → 8 → 1
    Activation: ELU (smoother than ReLU, handles negative values better)
    Regularization: Batch normalization + Dropout
    
    GKX found 3 hidden layers optimal for their data.
    Deeper networks overfit due to low signal-to-noise in financial data.
    """
    def __init__(self, n_features: int, hidden_dims=[32, 16, 8], dropout=0.3):
        super().__init__()
        
        layers = []
        in_dim = n_features
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_neural_net(train, val, target_col: str = "fwd_ret_21d",
                     epochs: int = 100, lr: float = 1e-3, batch_size: int = 256):
    """Train the neural network with early stopping on validation R²."""
    X_train, y_train = split_xy(train, target_col)
    X_val, y_val = split_xy(val, target_col)
    
    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)
    
    # PyTorch tensors
    train_ds = TensorDataset(torch.tensor(X_train_s), torch.tensor(y_train.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    X_val_t = torch.tensor(X_val_s)
    y_val_t = torch.tensor(y_val.astype(np.float32))
    
    # Model
    model = StockNN(n_features=X_train_s.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    
    best_val_r2 = -np.inf
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t).numpy()
        
        val_r2 = r2_score(y_val, val_preds)
        scheduler.step(-val_r2)
        
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= 20:  # Early stopping
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Val R² = {val_r2:.6f}")
    
    model.load_state_dict(best_state)
    print(f"\nBest Val R²: {best_val_r2:.6f}")
    return model, scaler, {"val_r2": best_val_r2}


# ======================================================
# OOS EVALUATION — THE ONLY NUMBER THAT MATTERS
# ======================================================

def evaluate_oos(model, X_test, y_test, model_type="sklearn", scaler=None):
    """
    Evaluate a model on the held-out test set.
    This is the ONLY evaluation that counts for reporting.
    
    Returns OOS R² and the series of predictions vs actuals.
    """
    if scaler is not None:
        X_test = scaler.transform(X_test)
    
    if model_type == "pytorch":
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test.astype(np.float32))).numpy()
    else:
        preds = model.predict(X_test)
    
    oos_r2 = r2_score(y_test, preds)
    
    return {
        "oos_r2": oos_r2,
        "predictions": preds,
        "actuals": y_test,
    }
```

---

## 7. Part 4: Monte Carlo Simulation (Tab 3)

```python
# backend/models/monte_carlo.py
import numpy as np
import pandas as pd

def run_gbm_simulation(
    current_price: float,
    ml_predicted_return: float,   # annualized expected return from ML model
    ml_predicted_vol: float,      # annualized volatility from ML model
    horizon_days: int = 252,
    n_simulations: int = 10_000,
    dt: float = 1/252,            # daily time steps
) -> dict:
    """
    Geometric Brownian Motion simulation parameterized by ML model outputs.
    
    GBM equation (discrete form):
        S(t+dt) = S(t) × exp[(μ - σ²/2)×dt + σ×√dt×Z]
    where Z ~ N(0,1)
    
    The (μ - σ²/2) term is the "drift correction":
    Why subtract σ²/2? Because the average of log-normally distributed
    returns is NOT the same as the expected log return. This correction
    ensures E[S(T)] = S(0) × exp(μ×T).
    
    In plain English:
    - μ: the expected daily drift (from ML model's return forecast)
    - σ: the daily volatility (from ML model's vol estimate)
    - Z: random shock (a roll of an infinitely-sided die from a normal curve)
    
    Args:
        current_price: today's stock price (S₀)
        ml_predicted_return: ML model's annualized return forecast (μ)
        ml_predicted_vol: ML model's annualized volatility estimate (σ)
        horizon_days: how many trading days to simulate forward
        n_simulations: number of paths to generate (10,000 = stable percentile estimates)
        dt: time step in years (1/252 = one trading day)
    
    Returns:
        dict with price paths array and percentile bands
    """
    mu = ml_predicted_return      # annualized drift
    sigma = ml_predicted_vol      # annualized volatility
    
    # Daily parameters
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    # Generate all random shocks at once (efficient vectorized operation)
    # Shape: (n_simulations, horizon_days)
    Z = np.random.standard_normal((n_simulations, horizon_days))
    
    # Daily log returns for each path
    daily_log_returns = drift + diffusion * Z
    
    # Cumulative price paths
    # cumsum gives cumulative log return; exp converts back to price ratio
    log_price_paths = np.cumsum(daily_log_returns, axis=1)
    price_paths = current_price * np.exp(log_price_paths)
    
    # Add S₀ as the starting point
    # Shape becomes (n_simulations, horizon_days + 1)
    price_paths = np.hstack([
        np.full((n_simulations, 1), current_price),
        price_paths
    ])
    
    # Compute percentile bands for the fan chart
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    bands = {
        f"p{p}": np.percentile(price_paths, p, axis=0)
        for p in percentiles
    }
    
    # Summary statistics at horizon
    final_prices = price_paths[:, -1]
    
    return {
        "price_paths": price_paths,          # full paths (for sample display)
        "percentile_bands": bands,            # for fan chart
        "horizon_days": horizon_days,
        "n_simulations": n_simulations,
        "summary": {
            "current_price": current_price,
            "median_final": float(np.median(final_prices)),
            "p10_final": float(np.percentile(final_prices, 10)),
            "p90_final": float(np.percentile(final_prices, 90)),
            "prob_positive": float(np.mean(final_prices > current_price)),
            "expected_return": float(np.mean(final_prices / current_price - 1)),
        }
    }
```

---

## 8. Part 5: DCF Model (Tab 4)

```python
# backend/models/dcf.py
import yfinance as yf
import numpy as np
from typing import Optional

def fetch_financials(ticker: str) -> dict:
    """
    Pull financial data from yfinance for DCF inputs.
    
    WARNING: yfinance financial data quality varies — treat as approximate.
    For any real investment decision, verify against SEC filings directly.
    """
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # Free cash flow from cash flow statement
    try:
        cf = stock.cashflow
        # Free cash flow = Operating CF - CapEx
        op_cf = cf.loc["Operating Cash Flow"].iloc[0] if "Operating Cash Flow" in cf.index else None
        capex = cf.loc["Capital Expenditure"].iloc[0] if "Capital Expenditure" in cf.index else 0
        fcf = float(op_cf + capex) if op_cf is not None else None  # capex is negative in yfinance
    except Exception:
        fcf = None
    
    return {
        "ticker": ticker,
        "company_name": info.get("longName", ticker),
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "shares_outstanding": info.get("sharesOutstanding"),
        "market_cap": info.get("marketCap"),
        "revenue_growth": info.get("revenueGrowth"),         # YoY revenue growth
        "free_cash_flow": fcf,
        "total_debt": info.get("totalDebt", 0) or 0,
        "cash": info.get("totalCash", 0) or 0,
        "beta": info.get("beta", 1.0) or 1.0,
    }


def run_dcf(
    fcf: float,                    # Trailing twelve-month free cash flow ($)
    shares_outstanding: float,     # Number of shares
    growth_rate_5y: float,         # Expected FCF growth rate for years 1-5
    terminal_growth_rate: float,   # Long-run growth rate after year 5 (should be < WACC)
    wacc: float,                   # Weighted Average Cost of Capital
    net_debt: float = 0,           # Total debt minus cash
) -> dict:
    """
    5-Year Discounted Cash Flow Model.
    
    The core idea: a company is worth the present value of ALL its future cash flows.
    "Present value" means: a dollar received in the future is worth less than a dollar
    today (because you could invest today's dollar and earn returns).
    
    Formula for each year's PV:
        PV_t = FCF_t / (1 + WACC)^t
    
    Terminal value: captures all cash flows beyond year 5 using the Gordon Growth Model:
        TV = FCF_5 × (1 + g) / (WACC - g)
    
    Then: Intrinsic Value per share = (sum of PV_1..5 + PV of TV - net_debt) / shares
    
    Args:
        fcf: current annual free cash flow
        shares_outstanding: total shares outstanding
        growth_rate_5y: annual FCF growth rate for years 1-5 (e.g., 0.10 = 10%)
        terminal_growth_rate: long-run growth rate (e.g., 0.03 = 3%, should be ~GDP growth)
        wacc: discount rate (e.g., 0.09 = 9%)
        net_debt: total debt minus cash (enterprise value adjustment)
    
    Returns:
        dict with intrinsic value per share, margin of safety, and waterfall components
    """
    if wacc <= terminal_growth_rate:
        return {"error": "WACC must be greater than terminal growth rate"}
    
    # Project FCF for years 1-5
    projected_fcf = []
    current_fcf = fcf
    for year in range(1, 6):
        current_fcf = current_fcf * (1 + growth_rate_5y)
        projected_fcf.append(current_fcf)
    
    # Discount each year's FCF to present value
    pv_fcf = []
    for t, fcf_t in enumerate(projected_fcf, start=1):
        pv = fcf_t / (1 + wacc) ** t
        pv_fcf.append(pv)
    
    # Terminal value at end of year 5 (Gordon Growth Model)
    fcf_year6 = projected_fcf[-1] * (1 + terminal_growth_rate)
    terminal_value = fcf_year6 / (wacc - terminal_growth_rate)
    pv_terminal = terminal_value / (1 + wacc) ** 5
    
    # Enterprise value = sum of PV of FCFs + PV of terminal value
    enterprise_value = sum(pv_fcf) + pv_terminal
    
    # Equity value = enterprise value - net debt
    equity_value = enterprise_value - net_debt
    
    # Intrinsic value per share
    intrinsic_value = equity_value / shares_outstanding if shares_outstanding else None
    
    return {
        "intrinsic_value_per_share": intrinsic_value,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "pv_year_fcfs": pv_fcf,           # For waterfall chart: Year 1-5 contributions
        "pv_terminal_value": pv_terminal,  # For waterfall chart: Terminal value contribution
        "terminal_value": terminal_value,
        "projected_fcf": projected_fcf,
        "wacc": wacc,
        "terminal_growth_rate": terminal_growth_rate,
        "growth_rate_5y": growth_rate_5y,
        # Waterfall components (for visualization)
        "waterfall": {
            "FCF Years 1-5": sum(pv_fcf),
            "Terminal Value": pv_terminal,
            "Net Debt Adjustment": -net_debt,
        }
    }
```

---

## 9. Part 6: FastAPI Backend

```python
# backend/main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import Optional
import traceback

# Import our modules
from data.fetcher import fetch_stock_data, fetch_sp500_and_macro
from data.features import build_gkx_features, build_target_variable, create_modeling_dataset
from models.baseline import compute_oos_r2, run_all_wg_predictors
from models.ml_pipeline import (temporal_train_val_test_split, split_xy,
                                  train_ridge, train_random_forest, train_neural_net,
                                  evaluate_oos)
from models.monte_carlo import run_gbm_simulation
from models.dcf import fetch_financials, run_dcf

app = FastAPI(title="Stock Return Predictor API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock this down in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- REQUEST/RESPONSE MODELS ----

class PredictionRequest(BaseModel):
    ticker: str
    start_date: str          # "YYYY-MM-DD"
    end_date: str
    horizon_days: int = 21   # prediction horizon
    mc_horizon: int = 252    # Monte Carlo horizon

class DCFRequest(BaseModel):
    ticker: str
    wacc: float = 0.09
    terminal_growth: float = 0.03
    growth_5y: float = 0.10

class MonteCarloRequest(BaseModel):
    ticker: str
    horizon_days: int = 252
    n_simulations: int = 10_000
    ml_return_override: Optional[float] = None
    ml_vol_override: Optional[float] = None


# ---- ENDPOINTS ----

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/baseline")
def get_baseline(req: PredictionRequest):
    """
    Tab 1: Welch & Goyal OOS R² for each macro predictor.
    Returns a table and cumulative R² curves.
    """
    try:
        macro_df = fetch_sp500_and_macro(start=req.start_date, end=req.end_date)
        summary_table = run_all_wg_predictors(macro_df)
        
        # Build cumulative R² time series for each predictor (for interactive chart)
        curves = {}
        for pred in ["tbl", "lty", "tms", "dfy", "infl", "svar"]:
            if pred in macro_df.columns:
                result = compute_oos_r2(macro_df, pred)
                curves[pred] = {
                    "dates": result["results_df"].index.strftime("%Y-%m-%d").tolist(),
                    "cum_oos_r2": result["results_df"]["cum_oos_r2"].tolist(),
                }
        
        return {
            "summary": summary_table.to_dict(orient="records"),
            "curves": curves,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict")
def get_predictions(req: PredictionRequest):
    """
    Tab 2: ML model predictions + feature importance.
    Runs Ridge → RF → NN pipeline, returns OOS R² comparison.
    """
    try:
        # Fetch data
        stock_df = fetch_stock_data(req.ticker, req.start_date, req.end_date)
        macro_df = fetch_sp500_and_macro(start=req.start_date, end=req.end_date)
        
        # Build features and target
        features = build_gkx_features(stock_df, macro_df)
        dataset = create_modeling_dataset(stock_df, features, req.horizon_days)
        
        if len(dataset) < 200:
            raise HTTPException(status_code=400, detail="Not enough data (need 200+ obs)")
        
        target_col = f"fwd_ret_{req.horizon_days}d"
        
        # Split
        train, val, test = temporal_train_val_test_split(dataset)
        X_test, y_test = split_xy(test, target_col)
        
        results = {}
        
        # Ridge
        ridge_model, ridge_scaler, ridge_meta = train_ridge(train, val, target_col)
        ridge_oos = evaluate_oos(ridge_model, X_test, y_test, "sklearn", ridge_scaler)
        results["ridge"] = {"oos_r2": ridge_oos["oos_r2"], "val_r2": ridge_meta["val_r2"]}
        
        # Random Forest
        rf_model, rf_meta = train_random_forest(train, val, target_col)
        rf_oos = evaluate_oos(rf_model, X_test, y_test, "sklearn")
        results["random_forest"] = {
            "oos_r2": rf_oos["oos_r2"],
            "val_r2": rf_meta["val_r2"],
            "feature_importances": rf_meta["feature_importances"].head(10).to_dict(),
        }
        
        # Neural Network (may be slow without GPU — optional flag)
        try:
            nn_model, nn_scaler, nn_meta = train_neural_net(train, val, target_col, epochs=50)
            nn_oos = evaluate_oos(nn_model, X_test, y_test, "pytorch", nn_scaler)
            results["neural_net"] = {"oos_r2": nn_oos["oos_r2"], "val_r2": nn_meta["val_r2"]}
            # Use best model's predictions for backtest
            best_preds = nn_oos["predictions"]
        except Exception:
            best_preds = rf_oos["predictions"]
        
        # Backtest: long when predicted return > 0, else hold cash
        test_dates = test.index.strftime("%Y-%m-%d").tolist()
        actual_returns = y_test.tolist()
        
        # Strategy: take position proportional to predicted return (sign only for simplicity)
        strategy_returns = np.sign(best_preds) * y_test
        bnh_returns = y_test  # buy-and-hold
        
        # Cumulative returns
        cum_strategy = (1 + strategy_returns).cumprod().tolist()
        cum_bnh = (1 + bnh_returns).cumprod().tolist()
        
        return {
            "model_comparison": results,
            "backtest": {
                "dates": test_dates,
                "strategy_cumulative": cum_strategy,
                "buyhold_cumulative": cum_bnh,
                "strategy_sharpe": float(np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(252/req.horizon_days)),
                "bnh_sharpe": float(np.mean(bnh_returns) / (np.std(bnh_returns) + 1e-8) * np.sqrt(252/req.horizon_days)),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.post("/api/montecarlo")
def get_monte_carlo(req: MonteCarloRequest):
    """Tab 3: Monte Carlo simulation."""
    stock_df = fetch_stock_data(req.ticker, 
                                 pd.Timestamp.today().strftime("%Y-%m-%d"), 
                                 pd.Timestamp.today().strftime("%Y-%m-%d"))
    # Use last available price
    current_price = float(stock_df["close"].iloc[-1])
    
    # Get historical vol if no override
    hist_returns = stock_df["close"].pct_change().dropna()
    hist_vol = float(hist_returns.std() * np.sqrt(252))
    hist_return = float(hist_returns.mean() * 252)
    
    mu = req.ml_return_override if req.ml_return_override is not None else hist_return
    sigma = req.ml_vol_override if req.ml_vol_override is not None else hist_vol
    
    mc_results = run_gbm_simulation(
        current_price=current_price,
        ml_predicted_return=mu,
        ml_predicted_vol=sigma,
        horizon_days=req.horizon_days,
        n_simulations=req.n_simulations,
    )
    
    # Return only percentile bands (not all 10k paths — too large)
    bands = mc_results["percentile_bands"]
    days = list(range(req.horizon_days + 1))
    
    return {
        "days": days,
        "bands": {k: v.tolist() for k, v in bands.items()},
        "summary": mc_results["summary"],
        "params": {"mu": mu, "sigma": sigma},
    }


@app.post("/api/dcf")
def get_dcf(req: DCFRequest):
    """Tab 4: DCF valuation."""
    financials = fetch_financials(req.ticker)
    
    if not financials.get("free_cash_flow") or not financials.get("shares_outstanding"):
        raise HTTPException(status_code=400, detail="Insufficient financial data for DCF")
    
    result = run_dcf(
        fcf=financials["free_cash_flow"],
        shares_outstanding=financials["shares_outstanding"],
        growth_rate_5y=req.growth_5y,
        terminal_growth_rate=req.terminal_growth,
        wacc=req.wacc,
        net_debt=financials["total_debt"] - financials["cash"],
    )
    
    margin_of_safety = None
    if result.get("intrinsic_value_per_share") and financials.get("current_price"):
        mos = (result["intrinsic_value_per_share"] - financials["current_price"]) / result["intrinsic_value_per_share"]
        margin_of_safety = round(mos * 100, 2)
    
    return {
        **result,
        "margin_of_safety_pct": margin_of_safety,
        "current_price": financials["current_price"],
        "company_name": financials["company_name"],
        "disclaimer": "⚠️ This is a simplified model for educational purposes only. Not financial advice. FCF data from yfinance may be inaccurate.",
    }
```

---

## 10. Part 7: Streamlit Frontend

```python
# frontend/app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta

API_BASE = "http://backend:8000"  # or "http://localhost:8000" for local dev

st.set_page_config(page_title="Stock Return Predictor", page_icon="📈", layout="wide")

st.title("📈 Stock Return Predictor")
st.caption("Replicating Welch & Goyal (2008) + Gu, Kelly & Xiu (2020)")

# ---- SIDEBAR INPUTS ----
with st.sidebar:
    st.header("⚙️ Settings")
    ticker = st.text_input("Stock Ticker", value="AAPL").upper().strip()
    start_date = st.date_input("Start Date", value=date(2015, 1, 1))
    end_date = st.date_input("End Date", value=date.today())
    horizon = st.selectbox("Prediction Horizon", [5, 10, 21, 63], index=2, 
                            format_func=lambda x: f"{x} days (~{x//21 or 1} month)")
    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# ---- TABS ----
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Baseline (WG 2008)", 
    "🤖 ML Forecast (GKX 2020)", 
    "🎲 Monte Carlo",
    "💰 DCF Valuation",
])

# ================================================================
# TAB 1: WELCH & GOYAL BASELINE
# ================================================================
with tab1:
    st.header("Welch & Goyal (2008) OOS R² Benchmark")
    st.info("""
    **What this shows:** Can traditional macro predictors beat a naive "predict the historical average" model?
    A positive OOS R² means the predictor has real signal. WG found most predictors fail this test.
    Your ML model (Tab 2) needs to beat the best predictor here.
    """)
    
    if run_btn:
        with st.spinner("Computing OOS R² for macro predictors..."):
            try:
                resp = requests.post(f"{API_BASE}/api/baseline", json={
                    "ticker": ticker,
                    "start_date": str(start_date),
                    "end_date": str(end_date),
                    "horizon_days": horizon,
                }, timeout=120)
                data = resp.json()
                
                # Summary table
                summary = pd.DataFrame(data["summary"])
                st.subheader("📋 OOS R² by Predictor")
                
                # Color-code: green if positive, red if negative
                def color_r2(val):
                    color = "green" if val > 0 else "red"
                    return f"color: {color}"
                
                st.dataframe(
                    summary.style.applymap(color_r2, subset=["OOS R² (%)"]),
                    use_container_width=True
                )
                
                # Cumulative R² chart
                st.subheader("📈 Cumulative OOS R² Over Time")
                fig = go.Figure()
                for pred, curve_data in data["curves"].items():
                    fig.add_trace(go.Scatter(
                        x=curve_data["dates"],
                        y=curve_data["cum_oos_r2"],
                        name=pred,
                        mode="lines",
                    ))
                fig.add_hline(y=0, line_dash="dash", line_color="black", 
                               annotation_text="Naive baseline (R²=0)")
                fig.update_layout(
                    title="Cumulative OOS R² — Above 0 = beats historical average",
                    xaxis_title="Date", yaxis_title="Cumulative OOS R²",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {e}")


# ================================================================
# TAB 2: ML FORECAST + FEATURE IMPORTANCE + BACKTEST
# ================================================================
with tab2:
    st.header("ML Forecast — Ridge → Random Forest → Neural Network")
    st.info("""
    **Pipeline:** Features are built with NO look-ahead bias. The model is trained on early data,
    tuned on validation data, and evaluated ONLY on the held-out test set.
    OOS R² > 0 means the ML model beats the naive historical average.
    """)
    
    if run_btn:
        with st.spinner("Training models... this may take 1-3 minutes"):
            try:
                resp = requests.post(f"{API_BASE}/api/predict", json={
                    "ticker": ticker,
                    "start_date": str(start_date),
                    "end_date": str(end_date),
                    "horizon_days": horizon,
                }, timeout=300)
                data = resp.json()
                
                # Model comparison
                st.subheader("🏆 Model Comparison (OOS R²)")
                comp = data["model_comparison"]
                
                col1, col2, col3 = st.columns(3)
                for col, (name, label) in zip([col1, col2, col3], [
                    ("ridge", "Ridge Regression"),
                    ("random_forest", "Random Forest"),
                    ("neural_net", "Neural Network"),
                ]):
                    if name in comp:
                        r2 = comp[name]["oos_r2"]
                        col.metric(
                            label=label,
                            value=f"{r2*100:.4f}%",
                            delta="✅ Beats naive" if r2 > 0 else "❌ Below naive",
                            delta_color="normal" if r2 > 0 else "inverse",
                        )
                
                # Feature importance bar chart
                if "feature_importances" in comp.get("random_forest", {}):
                    st.subheader("🔍 Feature Importance (Random Forest)")
                    fi = comp["random_forest"]["feature_importances"]
                    fi_df = pd.DataFrame({"feature": list(fi.keys()), "importance": list(fi.values())})
                    fig = px.bar(fi_df, x="importance", y="feature", orientation="h",
                                 title="Top 10 Predictive Features")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Backtest chart
                st.subheader("📊 Backtest: ML Strategy vs Buy-and-Hold")
                bt = data["backtest"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=bt["dates"], y=bt["strategy_cumulative"],
                                          name="ML Strategy", line=dict(color="blue")))
                fig.add_trace(go.Scatter(x=bt["dates"], y=bt["buyhold_cumulative"],
                                          name="Buy & Hold", line=dict(color="gray", dash="dash")))
                fig.update_layout(
                    title=f"Cumulative Returns — ML Strategy vs Buy & Hold ({ticker})",
                    xaxis_title="Date", yaxis_title="Portfolio Value (starting at 1.0)",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                col1.metric("ML Strategy Sharpe", f"{bt['strategy_sharpe']:.2f}")
                col2.metric("Buy & Hold Sharpe", f"{bt['bnh_sharpe']:.2f}")
                
            except Exception as e:
                st.error(f"Error: {e}")


# ================================================================
# TAB 3: MONTE CARLO
# ================================================================
with tab3:
    st.header("Monte Carlo Simulation — 10,000 Price Paths (GBM)")
    
    col1, col2, col3 = st.columns(3)
    mc_horizon = col1.selectbox("Time Horizon (days)", [30, 90, 180, 365], index=1)
    ml_return = col2.number_input("Expected Annual Return (%) — from ML model", 
                                    value=10.0, min_value=-50.0, max_value=100.0) / 100
    ml_vol = col3.number_input("Annual Volatility (%) — from ML model", 
                                 value=25.0, min_value=1.0, max_value=200.0) / 100
    
    if run_btn or st.button("Run Simulation"):
        with st.spinner("Running 10,000 simulations..."):
            resp = requests.post(f"{API_BASE}/api/montecarlo", json={
                "ticker": ticker,
                "horizon_days": mc_horizon,
                "n_simulations": 10_000,
                "ml_return_override": ml_return,
                "ml_vol_override": ml_vol,
            }, timeout=60)
            mc = resp.json()
            
            days = mc["days"]
            bands = mc["bands"]
            
            fig = go.Figure()
            
            # Fan chart
            fig.add_trace(go.Scatter(x=days, y=bands["p90"], name="90th %ile",
                                      line=dict(color="rgba(0,100,255,0.3)"), fill=None))
            fig.add_trace(go.Scatter(x=days, y=bands["p10"], name="10th %ile",
                                      line=dict(color="rgba(0,100,255,0.3)"),
                                      fill="tonexty", fillcolor="rgba(0,100,255,0.1)"))
            fig.add_trace(go.Scatter(x=days, y=bands["p75"], name="75th %ile",
                                      line=dict(color="rgba(0,100,255,0.5)"), fill=None))
            fig.add_trace(go.Scatter(x=days, y=bands["p25"], name="25th %ile",
                                      line=dict(color="rgba(0,100,255,0.5)"),
                                      fill="tonexty", fillcolor="rgba(0,100,255,0.15)"))
            fig.add_trace(go.Scatter(x=days, y=bands["p50"], name="Median",
                                      line=dict(color="blue", width=3)))
            
            fig.update_layout(
                title=f"{ticker} — Monte Carlo Price Paths ({mc_horizon} days, 10,000 simulations)",
                xaxis_title="Trading Days", yaxis_title="Price ($)",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
            
            s = mc["summary"]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Current Price", f"${s['current_price']:.2f}")
            col2.metric("Median Forecast", f"${s['median_final']:.2f}")
            col3.metric("10th Percentile", f"${s['p10_final']:.2f}")
            col4.metric("90th Percentile", f"${s['p90_final']:.2f}")
            
            prob = s["prob_positive"]
            st.metric("Probability of Positive Return", f"{prob*100:.1f}%")


# ================================================================
# TAB 4: DCF VALUATION
# ================================================================
with tab4:
    st.header("💰 Discounted Cash Flow (DCF) Valuation")
    st.warning("""
    ⚠️ **Disclaimer**: This is a simplified educational model. Financial data from yfinance
    may be inaccurate. This is NOT financial advice. Always verify against official SEC filings.
    The intrinsic value is highly sensitive to the WACC and growth rate assumptions.
    """)
    
    col1, col2, col3 = st.columns(3)
    wacc = col1.slider("WACC (%)", min_value=4.0, max_value=20.0, value=9.0, step=0.5) / 100
    growth_5y = col2.slider("5-Year FCF Growth (%)", min_value=-10.0, max_value=40.0, 
                              value=10.0, step=1.0) / 100
    terminal_growth = col3.slider("Terminal Growth (%)", min_value=0.0, max_value=5.0, 
                                    value=3.0, step=0.25) / 100
    
    if run_btn or st.button("Run DCF"):
        with st.spinner("Fetching financials and computing DCF..."):
            resp = requests.post(f"{API_BASE}/api/dcf", json={
                "ticker": ticker,
                "wacc": wacc,
                "terminal_growth": terminal_growth,
                "growth_5y": growth_5y,
            }, timeout=30)
            
            if resp.status_code != 200:
                st.error(f"Error: {resp.json().get('detail', 'Unknown error')}")
            else:
                dcf = resp.json()
                
                st.subheader(f"📊 {dcf.get('company_name', ticker)} — DCF Results")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Market Price", f"${dcf.get('current_price', 'N/A'):.2f}" 
                             if dcf.get("current_price") else "N/A")
                intrinsic = dcf.get("intrinsic_value_per_share")
                col2.metric("Intrinsic Value (DCF)", 
                             f"${intrinsic:.2f}" if intrinsic else "N/A")
                mos = dcf.get("margin_of_safety_pct")
                col3.metric("Margin of Safety", 
                             f"{mos:.1f}%" if mos is not None else "N/A",
                             delta="Undervalued" if mos and mos > 0 else "Overvalued",
                             delta_color="normal" if mos and mos > 0 else "inverse")
                
                # Waterfall chart
                if dcf.get("waterfall"):
                    st.subheader("🔍 Value Decomposition (Waterfall Chart)")
                    waterfall = dcf["waterfall"]
                    labels = list(waterfall.keys())
                    values = list(waterfall.values())
                    
                    fig = go.Figure(go.Waterfall(
                        name="DCF Components",
                        orientation="v",
                        measure=["relative", "relative", "relative"],
                        x=labels,
                        y=[v / 1e9 for v in values],
                        texttemplate="%{value:.1f}B",
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                    ))
                    fig.update_layout(
                        title="How Each Assumption Contributes to Enterprise Value ($B)",
                        yaxis_title="Contribution ($B)",
                    )
                    st.plotly_chart(fig, use_container_width=True)
```

---

## 11. Part 8: Dockerize

**`backend/requirements.txt`**
```
fastapi>=0.104
uvicorn[standard]>=0.24
yfinance>=0.2.36
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
torch>=2.0
pandas-datareader>=0.10
pydantic>=2.0
```

**`Dockerfile` (backend)**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**`docker-compose.yml`**
```yaml
version: "3.9"

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    image: python:3.11-slim
    working_dir: /app
    volumes:
      - ./frontend:/app
    command: >
      sh -c "pip install streamlit plotly requests pandas && 
             streamlit run app.py --server.port=8501 --server.address=0.0.0.0"
    ports:
      - "8501:8501"
    depends_on:
      backend:
        condition: service_healthy
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
```

**To run:**
```bash
# Build and start everything
docker-compose up --build

# App is at:
# Frontend: http://localhost:8501
# API docs: http://localhost:8000/docs
```

---

## 12. What to Build First — Recommended Order

1. ✅ **Step 1** (Week 1): `fetcher.py` — get S&P 500 + macro data working, print a DataFrame
2. ✅ **Step 2** (Week 1): `baseline.py` — implement OOS R², print the WG table for a few predictors  
3. ✅ **Step 3** (Week 2): `features.py` — build the GKX feature set, inspect for NaN/leakage
4. ✅ **Step 4** (Week 2): `ml_pipeline.py` — Ridge only first, verify the split is clean
5. ✅ **Step 5** (Week 3): Add RF + NN, compare OOS R²
6. ✅ **Step 6** (Week 3): `monte_carlo.py` and `dcf.py`
7. ✅ **Step 7** (Week 4): FastAPI backend — test each endpoint with `curl` or Swagger UI
8. ✅ **Step 8** (Week 4): Streamlit frontend — Tab 1 first (simplest), then Tab 2, 3, 4
9. ✅ **Step 9** (Week 5): Docker Compose — get it running end-to-end locally
10. ✅ **Step 10** (Week 5): Deploy (Fly.io, Railway, or AWS ECS)

---

## 13. The #1 Thing That Will Kill Your Project: Data Leakage

Quick checklist to run before every commit:

| Check | How to verify |
|---|---|
| Features only use past data | Print `features.iloc[100]` — does it use any data from rows 101+? |
| Target variable is strictly future | `target.shift(-horizon)` — does it align to future dates? |
| Scaler fitted on training data only | `scaler.fit(X_train)` then `.transform(X_val)` — never `fit_transform(X_val)` |
| Val/test never seen during hyperparameter search | Hyperparams chosen on val, final eval on test only |
| No future data in macro interactions | Macro resampled with `.ffill()` not `.bfill()` |

If your OOS R² is suspiciously high (> 5%), you almost certainly have a leakage bug.
Real expected OOS R² for a single stock: 0.01% to 0.5%.
