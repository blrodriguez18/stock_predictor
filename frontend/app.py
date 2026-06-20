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