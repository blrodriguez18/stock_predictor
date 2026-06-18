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

    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    # --- S&P 500 monthly returns ---
    sp500 = yf.download("^GSPC", start=start, end=end, interval="1mo", auto_adjust=True)
    sp500 = sp500["Close"].rename(columns={"Close": "sp500_close"})
    sp500.index = sp500.index.to_period("M").to_timestamp()
    sp500_ret = sp500.pct_change().rename(columns={"^GSPC": "mkt_ret"})

    # --- Risk-free rate (3-month T-bill from FRED) ---
    # TB3MS = 3-Month Treasury Bill Secondary Market Rate (monthly, annualized %)
    tbill = web.DataReader("TB3MS", "fred", start, end)
    tbill = tbill.resample("MS").last()/100/12
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
    macro_raw = macro_raw / 100

    macro_raw["tms"] = macro_raw["lty"] - macro_raw["tbl"]
    macro_raw["dfy"] = macro_raw["baa"] - macro_raw["aaa"]
    macro_raw["infl"] = macro_raw["cpi"].pct_change()
    macro_raw["infl"] = macro_raw["infl"].shift(1)

    daily_sp500 = yf.download("^GSPC", start=start, end=end, interval="1d", auto_adjust=True)
    daily_sp500 = daily_sp500["Close"].pct_change().dropna()
    daily_sp500.index = pd.to_datetime(daily_sp500.index)
    svar = daily_sp500.resample("MS").apply(lambda x: (x**2).sum()).rename(columns={"^GSPC":"svar"})

    df = df.join(macro_raw[["tbl", "lty", "tms", "dfy", "infl"]], how="left")
    # how come we join daily on monthly data?
    df = df.join(svar, how='left')
    df = df.dropna()

    return df