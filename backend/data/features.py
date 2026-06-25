import numpy as np
import pandas as pd
import yfinance as yf

def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} from {start} to {end}")

    # Flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        # Keep only the first level (Close, High, Low, ...)
        df.columns = [str(col[0]).lower() for col in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    df.index = pd.to_datetime(df.index)
    return df


# def build_gkx_features(df: pd.DataFrame, macro_df: pd.DataFrame = None) -> pd.DataFrame:
#     feat = pd.DataFrame(index=df.index)

#     # why not pct_change
#     log_ret = np.log(df["close"] / df["close"].shift(1))

#     # is this 21 trading days in a month
#     feat["ret_1m"] = df["close"].pct_change(21).shift(1)

#     feat["mom_12_1"] = (df["close"].shift(22) / df["close"].shift(252)) - 1

#     feat["mom_6_1"] = (df["close"].shift(22) / df["close"].shift(126)) - 1

#     feat["mom_3_1"] = (df["close"].shift(22) / df["close"].shift(63)) - 1

#     # volatility

#     feat["vol_1m"] = log_ret.rolling(22).std() * np.sqrt(252)

#     feat["vol_3m"] = log_ret.rolling(63).std() * np.sqrt(252)

#     feat["vol_6m"] = log_ret.rolling(126).std() * np.sqrt(252)

#     feat["vol_ratio"] = feat["vol_1m"] / (feat["vol_3m"] + 1e-8)

#     feat["max_ret"] = log_ret.rolling(21).max()

#     # market beta

#     try:
#         spy = yf.download("SPY", start=df.index[0] - pd.Timedelta(days=400), 
#                           end=df.index[-1] + pd.Timedelta(days=1), 
#                           auto_adjust=True, progress=False)["Close"]
#         mkt_ret = spy.pct_change().reindex(df.index)

#         # why mkt_ret %change if it's alr the %change from spy
#         cov_roll = log_ret.rolling(252).cov(mkt_ret.pct_change())
#         var_roll = mkt_ret.pct_change().rolling(252).var()
#         # is this the typical equation for beta
#         feat["beta"] = cov_roll / (var_roll + 1e-10)
#         feat["beta_sq"] = feat["beta"] ** 2  
        
#         feat["ivol"] = (log_ret - feat["beta"] * mkt_ret.pct_change()).rolling(21).std() * np.sqrt(252)

#     except Exception:
#         feat["beta"] = np.nan
#         feat["beta_sq"] = np.nan
#         feat["ivol"] = np.nan

#     # liquidity
#     dollar_vol = df["close"] * df["volume"]
#     feat["log_dolvol"] = np.log(dollar_vol.rolling(20).mean() + 1)

#     # feat["illiquidity"] = abs(df["close"] - df["close"].shift(1))/df["volume"]
#     feat["illiq"] = (log_ret.abs() / (dollar_vol + 1)).rolling(21).mean() * 1e6
    
#     feat["turn"] = df["volume"].rolling(20).mean() / (df["volume"].rolling(252).mean() + 1)

#     feat["high52w"] = df["close"]/df["high"].rolling(252).max()

#     feat["log_price"] = np.log(df["close"] + 1)

#     # moving avg signals

#     ma20 = df["close"].rolling(20).mean()
#     ma50 = df["close"].rolling(50).mean()
#     ma200 = df["close"].rolling(200).mean()

#     feat["price_to_ma20"] = df["close"] / (ma20 + 1e-8) - 1
#     feat["price_to_ma50"] = df["close"] / (ma50 + 1e-8) - 1
#     feat["price_to_ma200"] = df["close"] / (ma200 + 1e-8) - 1
#     feat["ma20_to_ma50"] = ma20 / (ma50 + 1e-8) - 1
#     feat["ma50_to_ma200"] = ma50 / (ma200 + 1e-8) - 1

#     # explain this RSI
#     delta = df["close"].diff()
#     gain = delta.where(delta > 0, 0).rolling(14).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
#     rs = gain / (loss + 1e-8)
#     feat["rsi14"] = 100 - (100 / (1 + rs))

#     # bollinger bands
#     bb_mid = ma20
#     bb_std = df["close"].rolling(20).std()
#     bb_upper = bb_mid + 2 * bb_std
#     bb_lower = bb_mid - 2 * bb_std
#     feat["bb_pct"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-8)
    
#     # interaction features
#     if macro_df is not None:
#         macro_daily = macro_df[["tms", "tbl"]].resample("D").ffill().reindex(df.index)
        
#         # Interaction: momentum × term spread
#         feat["mom_x_tms"] = feat["mom_12_1"] * macro_daily["tms"]
#         # Interaction: volatility × T-bill rate
#         feat["vol_x_tbl"] = feat["vol_1m"] * macro_daily["tbl"]
#         # Raw macro levels as features
#         feat["tms"] = macro_daily["tms"]
#         feat["tbl"] = macro_daily["tbl"]
    
#     # normalizing 
#     # explain winsorize
#     for col in feat.columns:
#         q01 = feat[col].quantile(0.01)
#         q99 = feat[col].quantile(0.99)
#         feat[col] = feat[col].clip(lower=q01, upper=q99)

#     for col in feat.columns:
#         # Rank up to current date, scaled to [-1, 1]
#         feat[col] = feat[col].expanding().rank(pct=True) * 2 - 1
    
#     return feat

def build_gkx_features(df: pd.DataFrame, macro_df: pd.DataFrame = None) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
 
    # WHY LOG RETURNS INSTEAD OF PCT_CHANGE:
    # pct_change() gives simple returns: (P_t - P_{t-1}) / P_{t-1}
    # np.log(P_t / P_{t-1}) gives log returns.
    # For rolling std calculations (volatility), log returns are preferred because
    # they are time-additive: log(P3/P1) = log(P3/P2) + log(P2/P1). Simple returns
    # are not additive, which makes averaging them over windows subtly wrong.
    # For momentum features we keep pct_change because GKX report those as simple
    # returns, which is the convention across the literature. The difference is tiny
    # at monthly horizons but matters for intraday or daily compounding.
    log_ret = np.log(df["close"] / df["close"].shift(1))
 
    # YES — 21 is the standard approximation for trading days in a month.
    # A calendar month has ~30 days but only ~21 are trading days (no weekends,
    # no holidays). pct_change(21) computes (price[t] - price[t-21]) / price[t-21].
    # The .shift(1) then moves the whole series forward by 1 row so that on any
    # given day, the value in this column is the return from 22 days ago to 1 day
    # ago — not including today. This prevents look-ahead: we never use today's
    # price as an input to predict today's future return.
    feat["ret_1m"] = df["close"].pct_change(21).shift(1)
 
    feat["mom_12_1"] = (df["close"].shift(22) / df["close"].shift(252)) - 1
 
    feat["mom_6_1"] = (df["close"].shift(22) / df["close"].shift(126)) - 1
 
    feat["mom_3_1"] = (df["close"].shift(22) / df["close"].shift(63)) - 1
 
    # volatility
 
    feat["vol_1m"] = log_ret.rolling(22).std() * np.sqrt(252)
 
    feat["vol_3m"] = log_ret.rolling(63).std() * np.sqrt(252)
 
    feat["vol_6m"] = log_ret.rolling(126).std() * np.sqrt(252)
 
    feat["vol_ratio"] = feat["vol_1m"] / (feat["vol_3m"] + 1e-8)
 
    feat["max_ret"] = log_ret.rolling(21).max()
 
    # market beta
    # Beta measures how much this stock moves when the whole market moves.
    # We estimate it via a rolling 252-day (1 year) regression of the stock's
    # log returns against the market's returns.
    # Beta = Cov(stock, market) / Var(market)
    # That is the standard OLS slope formula applied to a regression with no intercept.
    try:
        spy = yf.download("SPY", start=df.index[0] - pd.Timedelta(days=400),
                          end=df.index[-1] + pd.Timedelta(days=1),
                          auto_adjust=True, progress=False)["Close"]
 
        # spy["Close"] is a price series. One .pct_change() turns it into daily returns.
        # We store that as mkt_ret — it is already a return series at this point.
        # DO NOT call .pct_change() on mkt_ret again below; that would compute the
        # change-in-returns, which is nonsense and is the bug in the original code.
        if isinstance(spy.columns, pd.MultiIndex) if hasattr(spy, 'columns') else False:
            spy = spy.iloc[:, 0]
        mkt_ret = spy.pct_change().reindex(df.index)
 
        # Cov(log_ret, mkt_ret) over a rolling 252-day window
        cov_roll = log_ret.rolling(252).cov(mkt_ret)
        # Var(mkt_ret) over a rolling 252-day window
        var_roll = mkt_ret.rolling(252).var()
 
        # Beta = Cov / Var — the standard formula. The 1e-10 prevents division by
        # zero on days where the market had zero variance (extremely rare but possible
        # in data errors or market halts).
        feat["beta"] = cov_roll / (var_roll + 1e-10)
        feat["beta_sq"] = feat["beta"] ** 2
 
        # Idiosyncratic volatility: the part of the stock's daily move NOT explained
        # by the market. Residual = stock_return - beta * market_return.
        # We compute the rolling std of that residual over 21 days, annualized.
        residual = log_ret - feat["beta"] * mkt_ret
        feat["ivol"] = residual.rolling(21).std() * np.sqrt(252)
 
    except Exception:
        feat["beta"] = np.nan
        feat["beta_sq"] = np.nan
        feat["ivol"] = np.nan
 
    # NULL HANDLING FOR BETA / BETA_SQ / IVOL
    # These three features can produce sporadic NaNs mid-series (not just at startup)
    # if SPY data has gaps or alignment issues. Dropping those rows would create holes
    # in the middle of our time series, corrupting the temporal train/val/test split.
    # Fix: fill mid-series NaNs with the expanding-window median — the median of all
    # values seen so far at each point in time. This is safe because it never uses
    # future values; the expanding window only looks backward.
    # We do NOT use the full-series median (.median()) because that would use future
    # values to fill past NaNs, which is look-ahead bias.
    for col in ["beta", "beta_sq", "ivol"]:
        expanding_median = feat[col].expanding().median()
        feat[col] = feat[col].fillna(expanding_median)
 
    # liquidity
    dollar_vol = df["close"] * df["volume"]
    feat["log_dolvol"] = np.log(dollar_vol.rolling(20).mean() + 1)
 
    # feat["illiquidity"] = abs(df["close"] - df["close"].shift(1))/df["volume"]
    feat["illiq"] = (log_ret.abs() / (dollar_vol + 1)).rolling(21).mean() * 1e6
    
    feat["turn"] = df["volume"].rolling(20).mean() / (df["volume"].rolling(252).mean() + 1)
 
    feat["high52w"] = df["close"]/df["high"].rolling(252).max()
 
    feat["log_price"] = np.log(df["close"] + 1)
 
    # moving avg signals
 
    ma20 = df["close"].rolling(20).mean()
    ma50 = df["close"].rolling(50).mean()
    ma200 = df["close"].rolling(200).mean()
 
    feat["price_to_ma20"] = df["close"] / (ma20 + 1e-8) - 1
    feat["price_to_ma50"] = df["close"] / (ma50 + 1e-8) - 1
    feat["price_to_ma200"] = df["close"] / (ma200 + 1e-8) - 1
    feat["ma20_to_ma50"] = ma20 / (ma50 + 1e-8) - 1
    feat["ma50_to_ma200"] = ma50 / (ma200 + 1e-8) - 1
 
    # RSI — Relative Strength Index (14-day)
    # RSI answers: "over the past 14 days, how much of the total price movement
    # was upward vs downward?" It ranges from 0 to 100.
    # - Above 70: the stock has been rising a lot recently → considered "overbought"
    #   (possibly due for a pullback)
    # - Below 30: the stock has been falling a lot → considered "oversold"
    #   (possibly due for a bounce)
    # HOW IT'S COMPUTED:
    # delta = daily price change (positive on up days, negative on down days)
    # gain = average of positive daily changes over 14 days
    # loss = average of absolute value of negative daily changes over 14 days
    # RS  = gain / loss  (ratio of average up-move to average down-move)
    # RSI = 100 - (100 / (1 + RS))
    # If gains dominate: RS is large → 100/(1+RS) is small → RSI near 100
    # If losses dominate: RS is small → 100/(1+RS) is near 100 → RSI near 0
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    feat["rsi14"] = 100 - (100 / (1 + rs))
 
    # bollinger bands
    bb_mid = ma20
    bb_std = df["close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    feat["bb_pct"] = (df["close"] - bb_lower) / (bb_upper - bb_lower + 1e-8)
    
    # interaction features
    if macro_df is not None:
        macro_daily = macro_df[["tms", "tbl"]].resample("D").ffill().reindex(df.index)
        
        # Interaction: momentum × term spread
        feat["mom_x_tms"] = feat["mom_12_1"] * macro_daily["tms"]
        # Interaction: volatility × T-bill rate
        feat["vol_x_tbl"] = feat["vol_1m"] * macro_daily["tbl"]
        # Raw macro levels as features
        feat["tms"] = macro_daily["tms"]
        feat["tbl"] = macro_daily["tbl"]
    
    # WINSORIZING — WHY AND WHAT IT DOES:
    # Some features will occasionally produce extreme outliers. For example, on a
    # crash day a stock might have 10x its normal volume, making illiq explode.
    # Extreme values destabilize model training — Ridge and NNs are especially
    # sensitive because they minimize squared errors, so one massive outlier can
    # dominate the entire loss function.
    # Winsorizing clips values at the 1st and 99th percentile: anything below the
    # 1st percentile gets set TO the 1st percentile; anything above the 99th gets
    # set TO the 99th. The extreme values are replaced, not dropped.
    #
    # LOOK-AHEAD BUG FIX:
    # The original code used feat[col].quantile(0.01) — the full-series quantile.
    # That means on row 100, you clip using a threshold computed from rows 1–10,000.
    # Row 100's clipping was influenced by data from rows 101–10,000. That is
    # look-ahead bias hidden inside preprocessing.
    #
    # The fix: use an expanding window to compute quantiles, so at each row we only
    # use the distribution of values seen so far. We need a minimum number of rows
    # (here: 50) before the expanding quantile is meaningful; before that we don't
    # clip (leave the value as-is, let the later dropna handle startup NaNs).
    for col in feat.columns:
        expanding_q01 = feat[col].expanding(min_periods=50).quantile(0.01)
        expanding_q99 = feat[col].expanding(min_periods=50).quantile(0.99)
        feat[col] = feat[col].clip(lower=expanding_q01, upper=expanding_q99)
 
    # RANK NORMALIZATION — must come AFTER winsorizing, never before.
    # Converts each feature value to its percentile rank within its own history
    # up to that date, then scales to [-1, 1]. This makes all features comparable
    # regardless of their original units (dollar volume vs RSI vs beta all become
    # a number between -1 and 1). The expanding window means we only rank against
    # past values — no future data is used. NaN values are automatically ignored
    # by pandas' expanding().rank(), so they stay NaN and get cleaned up by the
    # dropna() in create_modeling_dataset().
    for col in feat.columns:
        feat[col] = feat[col].expanding().rank(pct=True) * 2 - 1
        
    # feat = feat.drop(columns=["beta", "beta_sq", "ivol"])

    return feat



# def build_target_variable(df: pd.DataFrame, horizon_days: int = 21) -> pd.Series:
    # Forward return: what will the price be in `horizon_days` days?
    # fwd_return = df["close"].pct_change(horizon_days).shift(-horizon_days)
    # return fwd_return.rename(f"fwd_ret_{horizon_days}d")

def build_target_variable(df: pd.DataFrame, horizon_days: int = 21) -> pd.Series:
    future_ret = df["close"].shift(-horizon_days) / df["close"] - 1
    future_ret.name = f"fwd_ret_{horizon_days}d"
    return future_ret


def create_modeling_dataset(df: pd.DataFrame, feat: pd.DataFrame, horizon_days: int = 21) -> pd.DataFrame:
    target = build_target_variable(df, horizon_days)

    print("feat shape:", feat.shape)
    print("target shape:", target.shape)
    print("feat index type:", type(feat.index))
    print("target index type:", type(target.index))
    print("feat index head:", feat.index[:5])
    print("target index head:", target.index[:5])

    dataset = feat.join(target)
    print("after join shape:", dataset.shape)
    print("NaNs per column:\n", dataset.isna().sum())

    print("all-NaN columns:", dataset.columns[dataset.isna().all()].tolist())
    print("NaN counts:\n", dataset.isna().sum().sort_values(ascending=False).head(10))

    # dataset = dataset.dropna()
    dataset = dataset.dropna(subset=[target.name])
    print("after dropna shape:", dataset.shape)
    return dataset