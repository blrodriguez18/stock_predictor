import numpy as np
import pandas as pd
import yfinance as yf

def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} from {start} to {end}")

    # Flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(part).lower() for part in col if part)
            for col in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    df.index = pd.to_datetime(df.index)
    return df


def build_gkx_features(df: pd.DataFrame, macro_df: pd.DataFrame = None) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)

    # why not pct_change
    log_ret = np.log(df["close"] / df["close"].shift(1))

    # is this 21 trading days in a month
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

    try:
        spy = yf.download("SPY", start=df.index[0] - pd.Timedelta(days=400), 
                          end=df.index[-1] + pd.Timedelta(days=1), 
                          auto_adjust=True, progress=False)["Close"]
        mkt_ret = spy.pct_change().reindex(df.index)

        # why mkt_ret %change if it's alr the %change from spy
        cov_roll = log_ret.rolling(252).cov(mkt_ret.pct_change())
        var_roll = mkt_ret.pct_change().rolling(252).var()
        # is this the typical equation for beta
        feat["beta"] = cov_roll / (var_roll + 1e-10)
        feat["beta_sq"] = feat["beta"] ** 2  
        
        feat["ivol"] = (log_ret - feat["beta"] * mkt_ret.pct_change()).rolling(21).std() * np.sqrt(252)

    except Exception:
        feat["beta"] = np.nan
        feat["beta_sq"] = np.nan
        feat["ivol"] = np.nan

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

    # explain this RSI
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
    
    # normalizing 
    # explain winsorize
    for col in feat.columns:
        q01 = feat[col].quantile(0.01)
        q99 = feat[col].quantile(0.99)
        feat[col] = feat[col].clip(lower=q01, upper=q99)

    for col in feat.columns:
        # Rank up to current date, scaled to [-1, 1]
        feat[col] = feat[col].expanding().rank(pct=True) * 2 - 1
    
    return feat



def build_target_variable(df: pd.DataFrame, horizon_days: int = 21) -> pd.Series:
    # Forward return: what will the price be in `horizon_days` days?
    fwd_return = df["close"].pct_change(horizon_days).shift(-horizon_days)
    return fwd_return.rename(f"fwd_ret_{horizon_days}d")


def create_modeling_dataset(df: pd.DataFrame, feat: pd.DataFrame, horizon_days: int = 21) -> pd.DataFrame:
    target = build_target_variable(df, horizon_days)
    dataset = feat.join(target)
    dataset = dataset.dropna()
    return dataset