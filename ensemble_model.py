"""
ensemble_model.py
------------------
Builds an actual ensemble (VotingRegressor + StackingRegressor) on top of the
same features/split used in model.py, and compares it against:
    - each individual base model
    - the "best single model" approach from model.py (picks lowest-RMSE model)

Run:
    python ensemble_model.py
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    VotingRegressor, StackingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR / "stock"
CSV_PATH = PROJECT_DIR / "data" / "historical_data.csv"
MODEL_PATH = PROJECT_DIR / "models" / "ensemble_model.pkl"

LAG_DAYS = 5
TEST_SIZE = 0.2

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
FEATURE_COLUMNS = [
    "Close",
    "Close_lag1", "Close_lag2", "Close_lag3", "Close_lag4", "Close_lag5",
    "MA5", "MA10",
    "Daily_Return",
    "Volume", "Volume_MA5",
]


def load_and_prepare():
    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("Date").drop_duplicates(subset="Date", keep="first")
    df[REQUIRED_COLUMNS] = df[REQUIRED_COLUMNS].ffill().bfill()
    df = df.reset_index(drop=True)

    for lag in range(1, LAG_DAYS + 1):
        df[f"Close_lag{lag}"] = df["Close"].shift(lag)
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volume_MA5"] = df["Volume"].rolling(window=5).mean()
    df["Target"] = df["Close"].shift(-1)
    df = df.dropna().reset_index(drop=True)
    return df


def split(df):
    X, y = df[FEATURE_COLUMNS], df["Target"]
    i = int(len(df) * (1 - TEST_SIZE))
    return X.iloc[:i], X.iloc[i:], y.iloc[:i], y.iloc[i:]


def score(name, y_test, preds, results):
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    print(f"{name:24s} | MAE: {mae:8.3f} | RMSE: {rmse:8.3f} | R2: {r2:.4f}")


def get_base_models():
    """Shared list of (name, model) tuples used by the voting/stacking ensembles.
    Imported by explain_model.py and app.py so all three scripts stay in sync."""
    return [
        ("lr", LinearRegression()),
        ("ridge", Ridge(alpha=1.0)),
        ("knn", KNeighborsRegressor(n_neighbors=5)),
        ("rf", RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
        ("gb", GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)),
    ]


def main():
    df = load_and_prepare()
    X_train, X_test, y_train, y_test = split(df)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    base_models = get_base_models()

    results = []
    print("Individual base models:")
    print("-" * 65)
    fitted = {}
    for name, model in base_models:
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        score(name, y_test, preds, results)
        fitted[name] = model

    print("-" * 65)
    print("Ensembles:")

    # Simple averaging ensemble
    voting = VotingRegressor(estimators=base_models)
    voting.fit(X_train_s, y_train)
    voting_preds = voting.predict(X_test_s)
    score("Voting (avg of 5)", y_test, voting_preds, results)

    # Stacked ensemble: base models feed a meta-learner
    stacking = StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
    )
    stacking.fit(X_train_s, y_train)
    stacking_preds = stacking.predict(X_test_s)
    score("Stacking (Ridge meta)", y_test, stacking_preds, results)

    print("-" * 65)
    results_df = pd.DataFrame(results).sort_values("RMSE")
    results_df.to_csv(
    PROJECT_DIR / "models" / "model_evaluation_results.csv",
    index=False
)
    print("\nRanked by RMSE (lower is better):")
    print(results_df.to_string(index=False))

    best_row = results_df.iloc[0]
    print(f"\nBest overall: {best_row['Model']} (RMSE={best_row['RMSE']:.3f}, R2={best_row['R2']:.4f})")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "voting_model": voting,
        "stacking_model": stacking,
        "scaler": scaler,
        "features": FEATURE_COLUMNS,
        "results": results_df,
    }, MODEL_PATH)
    print(f"\nSaved ensembles to {MODEL_PATH}")


if __name__ == "__main__":
    main()