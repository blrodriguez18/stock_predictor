"""
vol_surface.py
--------------
Volatility surface analysis + Black-Scholes Greeks + delta-hedge backtest.
Uses only: yfinance, numpy, scipy, pandas, matplotlib

Install deps:
    pip install yfinance numpy scipy pandas matplotlib

Run:
    python vol_surface.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


# ── CONFIG ──────────────────────────────────────────────────────────────────

TICKER      = "SPY"
RISK_FREE   = 0.05          # annualised risk-free rate (approx)
N_EXPIRIES  = 4             # how many expiry dates to pull
BACKTEST_WINDOW = 60        # trading days for delta-hedge backtest


# ── BLACK-SCHOLES HELPERS ────────────────────────────────────────────────────

def bs_price(S, K, T, r, sigma, flag="call"):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if flag == "call" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if flag == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, flag="call"):
    """Returns dict of Black-Scholes Greeks."""
    if T <= 0 or sigma <= 0:
        return {g: 0.0 for g in ["delta", "gamma", "vega", "theta", "rho"]}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    sign = 1 if flag == "call" else -1
    delta = sign * norm.cdf(sign * d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100          # per 1 vol point
    theta = (
        -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        - sign * r * K * np.exp(-r * T) * norm.cdf(sign * d2)
    ) / 365
    rho   = sign * K * T * np.exp(-r * T) * norm.cdf(sign * d2) / 100
    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def implied_vol(market_price, S, K, T, r, flag="call"):
    """Invert BS price to find implied vol via Brent's method."""
    intrinsic = max(S - K, 0) if flag == "call" else max(K - S, 0)
    if market_price <= intrinsic + 1e-6:
        return np.nan
    try:
        iv = brentq(
            lambda sigma: bs_price(S, K, T, r, sigma, flag) - market_price,
            1e-6, 10.0, xtol=1e-6, maxiter=200
        )
        return iv
    except Exception:
        return np.nan


# ── FETCH OPTIONS DATA ───────────────────────────────────────────────────────

def fetch_options(ticker=TICKER):
    print(f"\n{'='*60}")
    print(f"  Fetching options chain for {ticker}")
    print(f"{'='*60}")

    tk   = yf.Ticker(ticker)
    spot = tk.fast_info["last_price"]
    print(f"  Spot price : ${spot:.2f}")

    expiries = tk.options[:N_EXPIRIES]
    print(f"  Expiries   : {expiries}\n")

    records = []
    today   = datetime.today()

    for exp in expiries:
        chain = tk.option_chain(exp)
        T = (datetime.strptime(exp, "%Y-%m-%d") - today).days / 365
        if T <= 0:
            continue

        for flag, df in [("call", chain.calls), ("put", chain.puts)]:
            df = df.copy()
            df["mid"] = (df["bid"] + df["ask"]) / 2
            df = df[(df["mid"] > 0.05) & (df["openInterest"] > 10)]

            for _, row in df.iterrows():
                iv = implied_vol(row["mid"], spot, row["strike"], T, RISK_FREE, flag)
                if iv and 0.01 < iv < 3.0:
                    greeks = bs_greeks(spot, row["strike"], T, RISK_FREE, iv, flag)
                    records.append({
                        "expiry"      : exp,
                        "T"           : round(T, 4),
                        "strike"      : row["strike"],
                        "moneyness"   : round(row["strike"] / spot, 3),
                        "flag"        : flag,
                        "mid"         : round(row["mid"], 2),
                        "iv"          : round(iv, 4),
                        "openInterest": int(row["openInterest"]),
                        **{k: round(v, 4) for k, v in greeks.items()},
                    })

    df = pd.DataFrame(records)
    print(f"  Collected {len(df)} option contracts with valid IVs.\n")
    return df, spot


# ── VOL SURFACE ──────────────────────────────────────────────────────────────

def print_vol_surface(df):
    print("── IMPLIED VOLATILITY SURFACE (ATM ± 10%) ──────────────────")
    atm = df[(df["moneyness"] >= 0.90) & (df["moneyness"] <= 1.10)]
    pivot = (
        atm.groupby(["expiry", "moneyness"])["iv"]
        .mean()
        .unstack("moneyness")
        .round(3)
    )
    print(pivot.to_string())
    print()


def plot_vol_surface(df, spot):
    calls = df[df["flag"] == "call"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{TICKER} Implied Volatility Surface  |  Spot ${spot:.2f}", fontsize=13)

    # Smile per expiry
    ax = axes[0]
    for exp, grp in calls.groupby("expiry"):
        grp = grp.sort_values("moneyness")
        ax.plot(grp["moneyness"], grp["iv"] * 100, marker="o", markersize=3, label=exp)
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8, label="ATM")
    ax.set_xlabel("Moneyness (K/S)")
    ax.set_ylabel("Implied Vol (%)")
    ax.set_title("Vol Smile by Expiry")
    ax.legend(fontsize=8)

    # Term structure at ATM
    ax2 = axes[1]
    atm_ts = (
        calls[(calls["moneyness"] > 0.97) & (calls["moneyness"] < 1.03)]
        .groupby("T")["iv"]
        .mean()
        .sort_index()
    )
    ax2.plot(atm_ts.index * 365, atm_ts.values * 100, marker="s", color="steelblue")
    ax2.set_xlabel("Days to Expiry")
    ax2.set_ylabel("ATM Implied Vol (%)")
    ax2.set_title("ATM Vol Term Structure")

    plt.tight_layout()
    plt.savefig("vol_surface.png", dpi=150)
    print("  Chart saved → vol_surface.png\n")
    plt.show()


# ── GREEKS SUMMARY ───────────────────────────────────────────────────────────

def print_greeks_summary(df, spot):
    print("── GREEKS SNAPSHOT (nearest expiry, calls, ATM ± 5%) ───────")
    nearest = df["expiry"].min()
    atm = df[
        (df["expiry"] == nearest) &
        (df["flag"] == "call") &
        (df["moneyness"] >= 0.95) &
        (df["moneyness"] <= 1.05)
    ].sort_values("moneyness")

    cols = ["strike", "moneyness", "iv", "delta", "gamma", "vega", "theta"]
    print(atm[cols].to_string(index=False))
    print()

    # Vol skew observation
    otm_put_iv = df[
        (df["flag"] == "put") & (df["moneyness"] < 0.95) & (df["expiry"] == nearest)
    ]["iv"].mean()
    atm_iv = df[
        (df["flag"] == "call") & (df["moneyness"].between(0.98, 1.02)) & (df["expiry"] == nearest)
    ]["iv"].mean()

    if not np.isnan(otm_put_iv) and not np.isnan(atm_iv):
        skew = otm_put_iv - atm_iv
        direction = "elevated" if skew > 0.02 else "mild"
        print(f"  Vol skew (OTM put IV − ATM call IV): {skew*100:.1f} vol pts → {direction} downside hedging demand")
    print()


# ── DELTA-HEDGE BACKTEST ─────────────────────────────────────────────────────

def delta_hedge_backtest(ticker=TICKER, window=BACKTEST_WINDOW):
    """
    Simple long ATM call + daily delta hedge over `window` trading days.
    Strategy: buy 1 ATM call, hedge delta each day with underlying.
    PnL = option payoff + cumulative hedge PnL.
    """
    print("── DELTA-HEDGE BACKTEST ────────────────────────────────────")

    hist = yf.download(ticker, period="1y", auto_adjust=True, progress=False)["Close"].squeeze()
    hist = hist.dropna().iloc[-window - 1:]

    S0    = float(hist.iloc[0])
    K     = round(S0)               # ATM strike
    T0    = 30 / 365                # 30-day option
    sigma = float(hist.pct_change().dropna().std() * np.sqrt(252))  # hist vol

    print(f"  Underlying : {ticker}")
    print(f"  Entry spot : ${S0:.2f}  |  Strike: ${K}  |  Hist Vol: {sigma*100:.1f}%")

    option_entry = bs_price(S0, K, T0, RISK_FREE, sigma, "call")
    print(f"  Call price at entry: ${option_entry:.2f}\n")

    cash      = 0.0
    shares    = 0.0
    pnl_log   = []
    prev_price = S0

    for i, (date, price) in enumerate(hist.items()):
        price = float(price)
        T_rem = max((T0 - i / 252), 1 / 252)

        # Daily hedge rebalance
        delta = bs_greeks(price, K, T_rem, RISK_FREE, sigma, "call")["delta"]
        trade  = delta - shares                   # shares to buy/sell
        cash  -= trade * price                    # pay/receive cash
        shares = delta

        daily_pnl = shares * (price - prev_price)
        pnl_log.append({"date": date, "spot": price, "delta": round(delta, 4),
                         "daily_pnl": round(daily_pnl, 4)})
        prev_price = price

    df_bt = pd.DataFrame(pnl_log)

    # Final payoff
    S_final       = float(hist.iloc[-1])
    option_payoff = max(S_final - K, 0)
    hedge_pnl     = df_bt["daily_pnl"].sum()
    total_pnl     = option_payoff - option_entry + hedge_pnl

    print(f"  Exit spot     : ${S_final:.2f}")
    print(f"  Option payoff : ${option_payoff:.2f}")
    print(f"  Hedge PnL     : ${hedge_pnl:.2f}")
    print(f"  Net PnL       : ${total_pnl:.2f}  (vs premium paid ${option_entry:.2f})")

    sharpe = df_bt["daily_pnl"].mean() / (df_bt["daily_pnl"].std() + 1e-9) * np.sqrt(252)
    print(f"  Hedge Sharpe  : {sharpe:.2f}\n")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"{ticker} Delta-Hedge Backtest ({window}d)")
    ax1.plot(df_bt["date"], df_bt["spot"], color="steelblue")
    ax1.set_ylabel("Spot Price ($)")
    ax1.axhline(K, color="grey", linestyle="--", linewidth=0.8, label=f"Strike ${K}")
    ax1.legend()
    ax2.bar(df_bt["date"], df_bt["daily_pnl"],
            color=["green" if x >= 0 else "red" for x in df_bt["daily_pnl"]], width=0.8)
    ax2.set_ylabel("Daily Hedge PnL ($)")
    ax2.axhline(0, color="black", linewidth=0.6)
    plt.tight_layout()
    plt.savefig("backtest.png", dpi=150)
    print("  Chart saved → backtest.png\n")
    plt.show()

    return df_bt


# ── MAIN ─────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     # 1. Pull live options + compute IVs
#     df, spot = fetch_options(TICKER)

#     # 2. Print vol surface to console
#     print_vol_surface(df)

#     # 3. Plot vol smile + term structure
#     plot_vol_surface(df, spot)

#     # 4. Greeks snapshot
#     print_greeks_summary(df, spot)

#     # 5. Delta-hedge backtest
#     delta_hedge_backtest(TICKER, BACKTEST_WINDOW)

#     print("Done. Review vol_surface.png and backtest.png.")

if __name__ == "__main__":
    delta_hedge_backtest(TICKER, BACKTEST_WINDOW)