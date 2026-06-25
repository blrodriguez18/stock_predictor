import numpy as np
import pandas as pd
import statistics as stats
from sklearn.linear_model import LinearRegression

def compute_oos_r2(df: pd.DataFrame, predictor: str, target: str = "eq_premium", min_train_size: int = 60) -> dict:

    df = df[[target, predictor]].dropna().copy()

    df["x_lagged"] = df[[predictor]].shift(1).dropna()

    n = len(df)

    preds_model = []
    preds_naive = []
    actuals = []

    for t in range(min_train_size, n):
        train = df.loc[:t]

        naive_pred = train[target].mean()

        lr = LinearRegression()
        X_train = train[["x_lagged"]].values
        y_train = train[target].values

        lr.fit(X_train, y_train)

        X_pred = df.loc[t, "x_lagged"].values
        model_pred = lr.predict(X_pred)[0]

        preds_naive.append(naive_pred)
        preds_model.append(model_pred)
        actuals.append(df.loc[t, target])

    actuals = np.array(actuals)
    preds_model = np.array(preds_model)
    preds_naive = np.array(preds_naive)

    mse_model = np.mean((actuals - preds_model)**2)
    mse_naive = np.mean((actuals - preds_naive)**2)

    oos_r2 = 1 - (mse_model/mse_naive)

    cum_mse_model = np.cumsum((actuals - preds_model)** 2)
    cum_mse_naive = np.cumsum((actuals - preds_model)** 2)
    cum_oos_r2 = 1 - (cum_mse_model / cum_mse_naive)

    result_index = df.index[min_train_size:]
    results_df = pd.DataFrame({
        "actual":actuals,
        "model_pred":preds_model,
        "naive_preds":preds_naive,
        "cum_oos_r2":cum_oos_r2,
    }, index=result_index)

    return {
        "predictor": predictor,
        "oos_r2": round(oos_r2, 6),
        "oos_r2_pct": round(oos_r2 * 100, 4),
        "beats_naive": oos_r2 > 0,
        "n_oos_periods": len(actuals),
        "results_df": results_df,
    }


def run_all_wg_predictors(df: pd.DataFrame) -> pd.DataFrame:
    predictors = predictors = [c for c in ["tbl", "lty", "tms", "dfy", "infl", "svar"] if c in df.columns]

    rows = []
    for pred in predictors:
        result = compute_oos_r2(df, pred)
        rows.append({
            "Predictor": pred,
            "OOS R² (%)": result["oos_r2_pct"],
            "Beats Naive?": "✅ Yes" if result["beats_naive"] else "❌ No",
            "N OOS Periods": result["n_oos_periods"],
        })
    
    return pd.DataFrame(rows).sort_values("OOS R² (%)", ascending=False)


