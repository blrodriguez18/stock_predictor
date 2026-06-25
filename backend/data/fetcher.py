# Fetching the data

import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
import numpy as np
from datetime import datetime
    

def fetch_sp500_macro(start="1990-01-01", end=None):
    """
    Fetch S&P 500 returns + macro predictors from FRED.
    Returns a monthly DataFrame.
    """

    start = pd.to_datetime(start)
    if end is None:
        end = datetime.today()
    else:
        end = pd.to_datetime(end)

    # --- S&P 500 daily close ---
    sp500_daily = yf.download(
        "^GSPC",
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    # sp500 = sp500["Close"].rename("sp500_price")
    # sp500.index = sp500.index.to_period("M").to_timestamp()
    # sp500_ret = sp500.pct_change().rename("mkt_ret")

    # Make sure Close is a Series
    close = sp500_daily["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.squeeze()

    # if end is None:
    #     end = datetime.today().strftime("%Y-%m-%d")


    # Monthly market return
    monthly_close = close.resample("MS").last()
    mkt_ret = monthly_close.pct_change().rename("mkt_ret")

    # --- Risk-free rate (3-month T-bill from FRED) ---
    tbill = web.DataReader("TB3MS", "fred", start, end)
    tbill = tbill.resample("MS").last()
    tbill.columns = ["rf"]
    tbill["rf"] = tbill["rf"] / 100 / 12  # annualized percent -> monthly decimal

    # --- Equity premium = market return - risk-free rate ---
    df = mkt_ret.to_frame().join(tbill, how="inner")
    df["eq_premium"] = df["mkt_ret"] - df["rf"]
    

    # --- Macro predictors from FRED ---
    fred_series = {
        "DGS10": "lty",
        "TB3MS": "tbl",
        "BAMLC0A4CBBB": "baa",
        "BAMLC0A1CAAA": "aaa",
        "CPIAUCSL": "cpi",
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

    # --- Monthly realized variance from daily S&P 500 returns ---
    # daily_ret = close.pct_change().dropna()
    # svar = daily_ret.resample("MS").apply(lambda x: (x**2).sum()).rename("svar").to_frame()


    # S&P 500 stock variance: sum of squared DAILY returns per month
    sp500_daily = yf.download("^GSPC", start=start, end=end, interval="1d", auto_adjust=True, progress=False)
    if sp500_daily.empty:
        raise ValueError(f"No S&P 500 data returned for {start.date()} to {end.date()}")

    sp500_daily = sp500_daily["Close"].pct_change().dropna()
    print(type(sp500_daily))
    print(sp500_daily.head())
    # close = sp500_daily["Close"]
    # if isinstance(close, pd.DataFrame):
    #     close = close.iloc[:, 0]
    # close = close.squeeze()

    sp500_daily.index = pd.to_datetime(sp500_daily.index)

    daily_ret = close.pct_change().dropna()
    svar = daily_ret.resample("MS").apply(lambda x: (x ** 2).sum()).rename("svar").to_frame()


    # --- Join everything ---
    df = df.join(macro_raw[["tbl", "lty", "baa", "aaa", "infl", "tms", "dfy"]], how="left")
    df = df.join(svar, how="left")
    df = df.dropna()

    return df

    
    
