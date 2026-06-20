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

