from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import Optional
import traceback

# Import our modules
from data.fetcher import fetch_sp500_macro
from data.features import fetch_stock_data, build_gkx_features, build_target_variable, create_modeling_dataset
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
        macro_df = fetch_sp500_macro(start=req.start_date, end=req.end_date)
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
        macro_df = fetch_sp500_macro(start=req.start_date, end=req.end_date)
        
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
