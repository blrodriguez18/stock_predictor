import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.sklearn
import mlflow.pytorch


def temporal_train_val_test_split(dataset: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.20):
    n = len(dataset)
    train_end = int(n * (1 - val_frac - test_frac))
    val_end = int(n * (1 - test_frac))
    
    train = dataset.iloc[:train_end]
    val = dataset.iloc[train_end:val_end]
    test = dataset.iloc[val_end:]
    
    print(f"Train: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} obs)")
    print(f"Val:   {val.index[0].date()} → {val.index[-1].date()} ({len(val)} obs)")
    print(f"Test:  {test.index[0].date()} → {test.index[-1].date()} ({len(test)} obs)")
    
    return train, val, test


def split_xy(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=["target_col"]).values
    y = df[target_col].values

    return X, y


def train_ridge(train, val, target_col: str="fwd_ret_21d"):
    X_train, y_train = split_xy(train, target_col)
    X_val, y_val = split_xy(val, target_col)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    best_alpha, best_r2 = None, -np.inf
    for alpha in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        ridge = Ridge(alpha=alpha).fit(X_train_s, y_train)
        val_preds = ridge.predict(X_val_s)
        r2 = r2_score(y_val, val_preds)
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha

    X_tv = scaler.fit_transform(np.vstack([X_train, X_val]))
    y_tv = np.concatenate([y_train, y_val])
    final_ridge = Ridge(alpha=best_alpha).fit(X_tv, y_tv)

    print(f"Ridge best alpha: {best_alpha}, Val R²: {best_r2:.6f}")

    return final_ridge, scaler, {"best_alpha": best_alpha, "val_r2": best_r2}


def train_random_forest(train, val, target_col: str = "fwd_ret_21d"):
    X_train, y_train = split_xy(train, target_col)
    X_val, y_val = split_xy(val, target_col)


    feature_names = [c for c in train.columns if c != target_col]

    best_r2, best_params, best_model = -np.inf, {}, None