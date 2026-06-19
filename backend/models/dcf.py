import yfinance as yf
import numpy as np
from typing import Optional


def fetch_financials(ticker: str) -> dict:
    stock = yf.Ticker(ticker)
    info = stock.info

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


