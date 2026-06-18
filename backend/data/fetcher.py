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
    sp500 = sp500["Close"].rename("sp500_close")
    sp500.index = sp500.index.to_period("M").to_timestamp()
    sp500_ret = sp500.pct_change().rename("mkt_ret")

    