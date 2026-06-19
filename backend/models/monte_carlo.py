import numpy as np
import pandas as pd

def run_gbm_simulation(current_price: float,
    ml_predicted_return: float,   # annualized expected return from ML model
    ml_predicted_vol: float,      # annualized volatility from ML model
    horizon_days: int = 252,
    n_simulations: int = 10_000,
    dt: float = 1/252,            # daily time steps
) -> dict:
    
    # explain brownian motion

    mu = ml_predicted_return
    sigma = ml_predicted_vol

    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    Z = np.random.standard_normal((n_simulations, horizon_days))

    daily_log_returns = drift + diffusion * Z
    
    log_price_paths = np.cumsum(daily_log_returns, axis=1)
    price_paths = current_price * np.exp(log_price_paths)

    price_paths = np.hstack([
        np.full((n_simulations, 1), current_price),
        price_paths
    ])

    percentiles = [5, 10, 25, 50, 75, 90, 95]
    bands = {
        f"p{p}": np.percentile(price_paths, p, axis=0)
        for p in percentiles
    }

    final_prices = price_paths[:, -1]

    return {
        "price_paths": price_paths,         
        "percentile_bands": bands,           
        "horizon_days": horizon_days,
        "n_simulations": n_simulations,
        "summary": {
            "current_price": current_price,
            "median_final": float(np.median(final_prices)),
            "p10_final": float(np.percentile(final_prices, 10)),
            "p90_final": float(np.percentile(final_prices, 90)),
            "prob_positive": float(np.mean(final_prices > current_price)),
            "expected_return": float(np.mean(final_prices / current_price - 1)),
        }
    }


