"""
train_model.py
----------------
Trains SEVERAL different regression algorithms to predict next-day closing
price, compares them, and saves the BEST one to disk.

This is a single, self-contained script — no other files needed.

Steps:
    1. Download stock data (yfinance)
    2. Clean it
    3. Create features (lags, moving averages, returns)
    4. Train multiple models:
         - Linear Regression
         - Ridge Regression
         - Random Forest
         - Gradient Boosting
         - K-Nearest Neighbors
    5. Evaluate each on held-out test data (MAE, RMSE, R2)
    6. Pick the best model (lowest RMSE)
    7. Save it to models/best_model.pkl

Run:
    python train_model.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import joblib

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# SETTINGS
# =========================================================
TICKER = "RELIANCE.NS"
START_DATE = "2021-01-01"
END_DATE = "2026-01-01"

BASE_DIR = Path(__file__).resolve().parent
DATA_FOLDER = BASE_DIR / "data"
MODELS_FOLDER = BASE_DIR / "models"
CSV_PATH = DATA_FOLDER / "historical_data.csv"
MODEL_PATH = MODELS_FOLDER / "best_model.pkl"


def _safe_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        df.to_csv(temp_path, index=False)
        if path.exists():
            path.unlink()
        os.replace(temp_path, path)
    except PermissionError:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        if path.exists():
            print(f"Warning: '{path}' is locked by another process. Keeping the existing file and continuing.")
            return
        raise


def _safe_write_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        joblib.dump(obj, temp_path)
        if path.exists():
            path.unlink()
        os.replace(temp_path, path)
    except PermissionError:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        if path.exists():
            print(f"Warning: '{path}' is locked by another process. Keeping the existing file and continuing.")
            return
        raise

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


# =========================================================
# STEP 1: DOWNLOAD DATA
# =========================================================
def download_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    print(f"Downloading {ticker} data from {start_date} to {end_date} ...")

    data = yf.download(
        ticker, start=start_date, end=end_date,
        interval="1d", auto_adjust=False, progress=False,
    )

    if data.empty:
        raise ValueError(f"No data downloaded for '{ticker}'. Check the ticker/date range.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[REQUIRED_COLUMNS]
    data.reset_index(inplace=True)

    if CSV_PATH.exists():
        print(f"Using existing data file at {CSV_PATH} instead of overwriting it.")
        return pd.read_csv(CSV_PATH)

    try:
        _safe_write_csv(data, CSV_PATH)
        print(f"Saved raw data to {CSV_PATH} ({len(data)} records)")
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write to '{CSV_PATH}'. Close any program using this file and try again."
        ) from exc

    return data


# =========================================================
# STEP 2: CLEAN DATA
# =========================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("Date")
    df = df.drop_duplicates(subset="Date", keep="first")
    df[REQUIRED_COLUMNS] = df[REQUIRED_COLUMNS].ffill().bfill()
    df = df.reset_index(drop=True)

    return df


# =========================================================
# STEP 3: FEATURE ENGINEERING
# =========================================================
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for lag in range(1, LAG_DAYS + 1):
        df[f"Close_lag{lag}"] = df["Close"].shift(lag)

    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volume_MA5"] = df["Volume"].rolling(window=5).mean()

    # Target = next day's closing price
    df["Target"] = df["Close"].shift(-1)

    df = df.dropna().reset_index(drop=True)
    return df


# =========================================================
# STEP 4: TRAIN / TEST SPLIT (time-based, no shuffling)
# =========================================================
def split_data(df: pd.DataFrame, test_size: float):
    X = df[FEATURE_COLUMNS]
    y = df["Target"]

    split_index = int(len(df) * (1 - test_size))

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


# =========================================================
# STEP 5: TRAIN MULTIPLE MODELS & COMPARE
# =========================================================
def get_candidate_models():
    """
    Returns a dictionary of {name: model}.
    Linear/Ridge/KNN need scaled features to work well; tree-based
    models (Random Forest, Gradient Boosting) don't need scaling.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        ),
    }


def train_and_compare(X_train, X_test, y_train, y_test):
    # Scale features (helps Linear/Ridge/KNN; harmless for tree models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidates = get_candidate_models()
    results = []
    fitted_models = {}

    print("\nTraining and evaluating models...")
    print("-" * 60)

    for name, model in candidates.items():
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
        fitted_models[name] = model

        print(f"{name:22s} | MAE: {mae:8.3f} | RMSE: {rmse:8.3f} | R2: {r2:.4f}")

    print("-" * 60)

    results_df = pd.DataFrame(results).sort_values("RMSE")
    return results_df, fitted_models, scaler


# =========================================================
# MAIN
# =========================================================
def main():
    # 1. Download + clean + feature-engineer
    raw_df = download_data(TICKER, START_DATE, END_DATE)
    clean_df = clean_data(raw_df)
    feature_df = add_features(clean_df)

    if len(feature_df) < 50:
        raise ValueError("Not enough data to train a reliable model. Widen your date range.")

    # 2. Split
    X_train, X_test, y_train, y_test = split_data(feature_df, TEST_SIZE)

    # 3. Train & compare multiple algorithms
    results_df, fitted_models, scaler = train_and_compare(X_train, X_test, y_train, y_test)

    print("\nModel comparison (sorted by RMSE, lower is better):")
    print(results_df.to_string(index=False))

    # 4. Pick the best model (lowest RMSE)
    best_row = results_df.iloc[0]
    best_name = best_row["Model"]
    best_model = fitted_models[best_name]

    print(f"\nBEST MODEL: {best_name}")
    print(f"   RMSE: {best_row['RMSE']:.3f} | MAE: {best_row['MAE']:.3f} | R2: {best_row['R2']:.4f}")

    # 5. Save the best model + scaler + feature list together
    model_payload = {
        "model": best_model,
        "scaler": scaler,
        "features": FEATURE_COLUMNS,
        "model_name": best_name,
        "ticker": TICKER,
    }
    try:
        _safe_write_pickle(model_payload, MODEL_PATH)
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write to '{MODEL_PATH}'. Close any program using this file and try again."
        ) from exc
    print(f"\nSaved best model to {MODEL_PATH}")

    # 6. Predict tomorrow's closing price using the best model
    latest_row = feature_df[FEATURE_COLUMNS].iloc[[-1]]
    latest_scaled = scaler.transform(latest_row)
    next_day_prediction = best_model.predict(latest_scaled)[0]
    last_close = feature_df["Close"].iloc[-1]

    print(f"\nLast known close : {last_close:.2f}")
    print(f"Predicted next close ({best_name}): {next_day_prediction:.2f}")


if __name__ == "__main__":
    main()