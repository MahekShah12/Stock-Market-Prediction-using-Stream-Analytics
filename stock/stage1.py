"""
stage1.py
----------
Stock Market Prediction using Stream Analytics — STAGE 1 (single file)

What this script does, in order:
    1. Downloads real historical daily stock data from Yahoo Finance (yfinance)
    2. Saves it to data/historical_data.csv
    3. Cleans the data (dates, duplicates, missing values, numeric types)
    4. Prints basic statistics
    5. Saves two charts (closing price, volume)

No ML, no indicators, no streaming — that's for later stages.

Run:
    python stage1.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


# =========================================================
# SETTINGS — change these to try a different stock/date range
# =========================================================
TICKER = "RELIANCE.NS"
START_DATE = "2021-01-01"
END_DATE = "2026-01-01"

DATA_FOLDER = "data"
CSV_PATH = os.path.join(DATA_FOLDER, "historical_data.csv")
CLOSE_CHART_PATH = os.path.join(DATA_FOLDER, "closing_price_chart.png")
VOLUME_CHART_PATH = os.path.join(DATA_FOLDER, "volume_chart.png")

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


# =========================================================
# STEP 1: DOWNLOAD DATA
# =========================================================
def download_stock_data(ticker: str, start_date: str, end_date: str, save_path: str) -> pd.DataFrame:
    print(f"Downloading data for {ticker} from {start_date} to {end_date} ...")

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            f"No data was downloaded for ticker '{ticker}'. "
            "Please check the ticker symbol and date range."
        )

    # Flatten columns if yfinance returns a MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data[REQUIRED_COLUMNS]
    data.reset_index(inplace=True)  # turns Date index into a normal column

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data.to_csv(save_path, index=False)
    print(f"Saved {len(data)} records to {save_path}")

    return data


# =========================================================
# STEP 2: CLEAN / PREPROCESS DATA
# =========================================================
def clean_data(csv_path: str, ticker_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Validate required columns
    all_required = ["Date"] + REQUIRED_COLUMNS
    missing_columns = [col for col in all_required if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in data: {missing_columns}")

    # Convert types
    df["Date"] = pd.to_datetime(df["Date"])
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort + remove duplicate dates
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset="Date", keep="first")

    # Check missing values before handling
    missing_before = df[REQUIRED_COLUMNS].isna().sum()

    # Handle missing values (forward-fill, then back-fill any left at the start)
    df[REQUIRED_COLUMNS] = df[REQUIRED_COLUMNS].ffill().bfill()

    df = df.reset_index(drop=True)

    print("=" * 40)
    print(f"Stock: {ticker_name}")
    print(f"Records: {len(df)}")
    print(f"Start Date: {df['Date'].min().date()}")
    print(f"End Date: {df['Date'].max().date()}")
    print()
    print("Missing values (before cleaning):")
    for col in REQUIRED_COLUMNS:
        print(f"{col}: {missing_before[col]}")
    print("=" * 40)

    return df


# =========================================================
# STEP 3: BASIC ANALYSIS
# =========================================================
def print_basic_analysis(df: pd.DataFrame):
    num_records = len(df)
    highest_close = df["Close"].max()
    lowest_close = df["Close"].min()
    average_close = df["Close"].mean()
    average_volume = df["Volume"].mean()

    first_close = df["Close"].iloc[0]
    last_close = df["Close"].iloc[-1]
    pct_change = ((last_close - first_close) / first_close) * 100

    print("\nBASIC ANALYSIS")
    print("-" * 40)
    print(f"Number of trading records : {num_records}")
    print(f"Highest closing price     : {highest_close:.2f}")
    print(f"Lowest closing price      : {lowest_close:.2f}")
    print(f"Average closing price     : {average_close:.2f}")
    print(f"Average trading volume    : {average_volume:.0f}")
    print(f"Overall % change in close : {pct_change:.2f}%")
    print("-" * 40)

    print("\nFirst 5 records:")
    print(df.head())

    print("\nLast 5 records:")
    print(df.tail())


# =========================================================
# STEP 4: CHARTS
# =========================================================
def make_charts(df: pd.DataFrame, ticker: str):
    # Chart 1: Closing Price
    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["Close"], color="blue")
    plt.title(f"{ticker} - Closing Price Over Time")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.tight_layout()
    plt.savefig(CLOSE_CHART_PATH)
    plt.close()
    print(f"\nSaved: {CLOSE_CHART_PATH}")

    # Chart 2: Trading Volume
    plt.figure(figsize=(10, 5))
    plt.bar(df["Date"], df["Volume"], color="orange")
    plt.title(f"{ticker} - Trading Volume Over Time")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.tight_layout()
    plt.savefig(VOLUME_CHART_PATH)
    plt.close()
    print(f"Saved: {VOLUME_CHART_PATH}")


# =========================================================
# MAIN
# =========================================================
def main():
    download_stock_data(TICKER, START_DATE, END_DATE, CSV_PATH)
    df = clean_data(CSV_PATH, TICKER)
    print_basic_analysis(df)
    make_charts(df, TICKER)


if __name__ == "__main__":
    main()