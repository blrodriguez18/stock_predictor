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


# def temporal_train_val_test_split(dataset: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.20):
#     n = len(dataset)
#     train_end = int(n * (1 - val_frac - test_frac))
#     val_end = int(n * (1 - test_frac))
    
#     train = dataset.iloc[:train_end]
#     val = dataset.iloc[train_end:val_end]
#     test = dataset.iloc[val_end:]
    
#     print(f"Train: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} obs)")
#     print(f"Val:   {val.index[0].date()} → {val.index[-1].date()} ({len(val)} obs)")
#     print(f"Test:  {test.index[0].date()} → {test.index[-1].date()} ({len(test)} obs)")
    
#     return train, val, test


def temporal_train_val_test_split(dataset: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.20):
    n = len(dataset)
    if n == 0:
        raise ValueError("Dataset is empty")

    train_end = int(n * (1 - val_frac - test_frac))
    val_end = int(n * (1 - test_frac))

    train = dataset.iloc[:train_end]
    val = dataset.iloc[train_end:val_end]
    test = dataset.iloc[val_end:]

    print(f"train len={len(train)}, val len={len(val)}, test len={len(test)}")

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(
            f"Empty split: train={len(train)}, val={len(val)}, test={len(test)}"
        )

    print(f"Train: {train.index[0].date()} → {train.index[-1].date()} ({len(train)} obs)")
    print(f"Val:   {val.index[0].date()} → {val.index[-1].date()} ({len(val)} obs)")
    print(f"Test:  {test.index[0].date()} → {test.index[-1].date()} ({len(test)} obs)")

    return train, val, test


def split_xy(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col]).values
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

    best_r2,  best_params, best_model = -np.inf, {}, None
    
    feature_names = [c for c in train.columns if c != target_col]

    for n_est in [100, 200]:
        for max_depth in [3, 5, 8]:
            for max_feat in [0.3, 0.5, "sqrt"]:
                rf = RandomForestRegressor(
                    n_estimators=n_est,
                    max_depth=max_depth,
                    max_features=max_feat,
                    min_samples_leaf=50,
                    n_jobs=-1,
                    random_state=42,
                ).fit(X_train, y_train)
                
                r2 = r2_score(y_val, rf.predict(X_val))
                if r2 > best_r2:
                    best_r2 = r2
                    best_params = {"n_estimators": n_est, "max_depth": max_depth, 
                                   "max_features": max_feat}
                    best_model = rf

    importances = pd.Series(best_model.feature_importances_, index=feature_names).sort_values(ascending=False)

    print(f"RF best params: {best_params}, Val R²: {best_r2:.6f}")
    print("\nTop 5 features:")
    print(importances.head())
    
    return best_model, {"val_r2": best_r2, "params": best_params, "feature_importances": importances}


class StockNN(nn.Module):
    def __init__(self, n_features: int, hidden_dims=[32, 16, 8], dropout=0.3):
        # what is super for
        super().__init__()

        layers = []
        in_dim = n_features

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        # how is NN structured
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeze(-1)
    

def train_neural_net(train, val, target_col: str = "fwd_ret_21d", epochs: int = 100, lr: float = 1e-3, batch_size: int = 256):
    X_train, y_train = split_xy(train, target_col)
    X_val, y_val = split_xy(val, target_col)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    train_ds = TensorDataset(torch.tensor(X_train_s), torch.tensor(y_train.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_val_t = torch.tensor(X_val_s)
    y_val_t = torch.tensor(y_val.astype(np.float32))

    # did the paper specify these parameters
    model = StockNN(n_features=X_train_s.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

    best_val_r2 = -np.inf
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t).numpy()

        val_r2 = r2_score(y_val, val_preds)
        scheduler.step(-val_r2)

        # what does this do
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= 20:  # Early stopping
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Val R² = {val_r2:.6f}")
    
    model.load_state_dict(best_state)
    print(f"\nBest Val R²: {best_val_r2:.6f}")
    return model, scaler, {"val_r2": best_val_r2}


# what is this for
def evaluate_oos(model, X_test, y_test, model_type="sklearn", scaler=None):
    if scaler is not None:
        X_test = scaler.transform(X_test)
    
    if model_type == "pytorch":
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test.astype(np.float32))).numpy()
    else:
        preds = model.predict(X_test)
    
    oos_r2 = r2_score(y_test, preds)
    
    return {
        "oos_r2": oos_r2,
        "predictions": preds,
        "actuals": y_test,
    }