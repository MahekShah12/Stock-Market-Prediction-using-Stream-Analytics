"""
stock_stream_analytics.py
-------------------------
RELIANCE.NS - Stock Market Prediction using Stream Analytics (final app)

Four sections only:
    1. Market Overview   - 4 KPI cards, price chart, volume chart
    2. Stream Analytics  - rolling mean / volatility / volume, MA5 vs MA20,
                           streaming simulation (latest 50 observations)
    3. ML Prediction     - model comparison table + RMSE chart,
                           actual vs predicted, next-day prediction
    4. Explainability    - SHAP feature importance

Data-leakage handling (ML pipeline):
    * Chronological split: the first 80% of days train, the last 20% test.
      No shuffling anywhere.
    * The scaler is fitted on the training period only.
    * Stacking uses time-ordered out-of-fold predictions (TimeSeriesSplit), so
      a base model never predicts days that lie before the data it was trained
      on. (sklearn's StackingRegressor(cv=5) does not guarantee this.)
    * Every feature only looks backwards (lags / trailing rolling windows).

Run:
    pip install streamlit pandas numpy scikit-learn shap matplotlib
    streamlit run stock_stream_analytics.py
"""

import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

# =========================================================
# SETTINGS
# =========================================================
TICKER = "RELIANCE.NS"
CSV_PATH = Path(__file__).resolve().parent / "stock" / "data" / "historical_data.csv"

TEST_SIZE = 0.2          # last 20% of the timeline = test set
STREAM_WINDOW = 50       # observations used by the streaming simulation
CURRENCY = "₹"

FEATURE_COLUMNS = [
    "Close",
    "Close_lag1", "Close_lag2", "Close_lag3", "Close_lag4", "Close_lag5",
    "MA5", "MA10",
    "Daily_Return",
    "Volume", "Volume_MA5",
]

MODEL_ORDER = [
    "Linear Regression", "Ridge", "KNN", "Random Forest",
    "Gradient Boosting", "Voting Ensemble", "Stacking Ensemble",
]

st.set_page_config(page_title=f"{TICKER} Stream Analytics", page_icon="📈", layout="wide")


# =========================================================
# DATA + FEATURES
# =========================================================
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    cols = ["Open", "High", "Low", "Close", "Volume"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("Date").drop_duplicates("Date", keep="first")
    df[cols] = df[cols].ffill().bfill()
    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Analytics columns + ML features. Everything only looks backwards."""
    d = df.copy()

    # --- Stream analytics metrics
    d["Daily_Return"] = d["Close"].pct_change()
    d["MA5"] = d["Close"].rolling(5).mean()
    d["MA20"] = d["Close"].rolling(20).mean()
    d["MA_Diff"] = d["MA5"] - d["MA20"]
    d["Volatility20"] = d["Daily_Return"].rolling(20).std() * 100   # in %
    d["Volume_MA5"] = d["Volume"].rolling(5).mean()
    d["Volume_MA20"] = d["Volume"].rolling(20).mean()

    # --- ML-only features
    for lag in range(1, 6):
        d[f"Close_lag{lag}"] = d["Close"].shift(lag)
    d["MA10"] = d["Close"].rolling(10).mean()

    # --- Target: next trading day's close (NaN on the very last row)
    d["Target"] = d["Close"].shift(-1)
    return d


# =========================================================
# STREAMING SIMULATION
# =========================================================
class StreamProcessor:
    """Processes observations one by one and updates rolling metrics
    incrementally over a fixed-size sliding window."""

    def __init__(self, window: int, prev_close: float):
        self.prices = deque(maxlen=window)
        self.volumes = deque(maxlen=window)
        self.returns = deque(maxlen=window)
        self.prev_close = prev_close

    def update(self, date, price: float, volume: float) -> dict:
        daily_ret = price / self.prev_close - 1
        self.prev_close = price

        self.prices.append(price)
        self.volumes.append(volume)
        self.returns.append(daily_ret)

        vol = float(np.std(self.returns, ddof=1) * 100) if len(self.returns) > 1 else np.nan
        return {
            "Date": date,
            "Window Size": len(self.prices),
            "Current Price": price,
            "Rolling Mean": float(np.mean(self.prices)),
            "Rolling Volatility (%)": vol,
            "Current Volume": volume,
            "Daily Return (%)": daily_ret * 100,
        }


@st.cache_data
def run_stream(df: pd.DataFrame, window: int = STREAM_WINDOW) -> pd.DataFrame:
    """Replay the latest `window` observations through the StreamProcessor."""
    latest = df.tail(window)
    seed_close = float(df["Close"].iloc[-window - 1])   # the day just before the window
    proc = StreamProcessor(window, seed_close)
    states = [
        proc.update(r.Date, float(r.Close), float(r.Volume))
        for r in latest.itertuples(index=False)
    ]
    return pd.DataFrame(states)


# =========================================================
# ML PIPELINE
# =========================================================
def get_base_models():
    return [
        ("lr", LinearRegression()),
        ("ridge", Ridge(alpha=1.0)),
        ("knn", KNeighborsRegressor(n_neighbors=5)),
        ("rf", RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
        ("gb", GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)),
    ]


class TimeSeriesStacking(BaseEstimator, RegressorMixin):
    """Stacking that respects time order.

    Meta-features are out-of-fold predictions from an expanding-window
    TimeSeriesSplit: each block is predicted only by models trained on
    EARLIER data. Base models are then refit on the full training period.
    """

    def __init__(self, estimators, final_estimator=None, n_splits=5):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.n_splits = n_splits

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        meta = clone(self.final_estimator) if self.final_estimator is not None else Ridge(alpha=1.0)

        oof = np.full((len(X), len(self.estimators)), np.nan)
        for tr, va in TimeSeriesSplit(n_splits=self.n_splits).split(X):
            for j, (_, est) in enumerate(self.estimators):
                oof[va, j] = clone(est).fit(X[tr], y[tr]).predict(X[va])

        seen = ~np.isnan(oof).any(axis=1)      # the first block has no out-of-fold predictions
        self.final_estimator_ = meta.fit(oof[seen], y[seen])
        self.estimators_ = [clone(est).fit(X, y) for _, est in self.estimators]
        return self

    def predict(self, X):
        X = np.asarray(X)
        meta_X = np.column_stack([m.predict(X) for m in self.estimators_])
        return self.final_estimator_.predict(meta_X)


def _metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R²": r2_score(y_true, y_pred),
    }


@st.cache_resource(show_spinner="Training models (chronological split)...")
def train_pipeline(df: pd.DataFrame) -> dict:
    feat = add_features(df)

    # Rows with a complete feature set AND a known target -> supervised data
    labeled = feat.dropna(subset=FEATURE_COLUMNS + ["Target"]).reset_index(drop=True)

    # ---- Chronological split: past -> train, future -> test (NO shuffling)
    split_idx = int(len(labeled) * (1 - TEST_SIZE))
    train, test = labeled.iloc[:split_idx], labeled.iloc[split_idx:]
    assert train["Date"].max() < test["Date"].min(), "Train must end before test starts"

    scaler = StandardScaler().fit(train[FEATURE_COLUMNS])       # fit on TRAIN only
    X_train = scaler.transform(train[FEATURE_COLUMNS])
    X_test = scaler.transform(test[FEATURE_COLUMNS])
    y_train, y_test = train["Target"].to_numpy(), test["Target"].to_numpy()

    names = {
        "lr": "Linear Regression", "ridge": "Ridge", "knn": "KNN",
        "rf": "Random Forest", "gb": "Gradient Boosting",
    }
    models = {names[k]: m.fit(X_train, y_train) for k, m in get_base_models()}
    models["Voting Ensemble"] = VotingRegressor(get_base_models()).fit(X_train, y_train)
    models["Stacking Ensemble"] = TimeSeriesStacking(
        get_base_models(), final_estimator=Ridge(alpha=1.0), n_splits=5
    ).fit(X_train, y_train)

    preds = {n: models[n].predict(X_test) for n in MODEL_ORDER}
    results = pd.DataFrame([{"Model": n, **_metrics(y_test, preds[n])} for n in MODEL_ORDER])

    # ---- Next-day prediction row = the LATEST observation (its target is unknown)
    latest_feat = feat.dropna(subset=FEATURE_COLUMNS).iloc[[-1]]
    X_latest = scaler.transform(latest_feat[FEATURE_COLUMNS])

    return {
        "models": models, "scaler": scaler, "results": results, "preds": preds,
        "train": train, "test": test, "X_train": X_train, "X_test": X_test,
        "latest_feat": latest_feat, "X_latest": X_latest,
    }


# =========================================================
# SHAP  (model-agnostic, so it also works for the ensembles)
# =========================================================
@st.cache_resource(show_spinner="Computing SHAP values...")
def compute_shap(_bundle: dict, model_name: str):
    model = _bundle["models"][model_name]
    masker = shap.maskers.Independent(_bundle["X_train"], max_samples=50)
    explainer = shap.explainers.Permutation(
        model.predict, masker, feature_names=FEATURE_COLUMNS, seed=42, silent=True
    )
    max_evals = 4 * len(FEATURE_COLUMNS) + 1
    sv_global = explainer(_bundle["X_test"][-40:], max_evals=max_evals)   # last 40 test days
    sv_latest = explainer(_bundle["X_latest"], max_evals=max_evals)       # next-day row
    return sv_global, sv_latest


# =========================================================
# LOAD EVERYTHING
# =========================================================
if not CSV_PATH.exists():
    st.error(f"Data file not found: {CSV_PATH}")
    st.stop()

df = load_data()
feat = add_features(df)
bundle = train_pipeline(df)

# =========================================================
# HEADER + SIDEBAR
# =========================================================
st.title(f"📈 {TICKER} — Stock Market Prediction using Stream Analytics")
st.caption(
    f"Daily data: {df['Date'].iloc[0]:%d %b %Y} → {df['Date'].iloc[-1]:%d %b %Y} "
    f"({len(df)} trading days). Streaming-style analytics on historical data — "
    "not a live real-time feed."
)

with st.sidebar:
    st.header("Settings")
    selected_model = st.radio("Selected ensemble model", ["Stacking Ensemble", "Voting Ensemble"])
    st.caption("Used for Actual vs Predicted, Next-Day Prediction and SHAP.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["① Market Overview", "② Stream Analytics", "③ ML Prediction", "④ Explainability"]
)

# =========================================================
# ① MARKET OVERVIEW
# =========================================================
with tab1:
    last, prev = df.iloc[-1], df.iloc[-2]
    daily_ret = (last["Close"] / prev["Close"] - 1) * 100
    vol20 = feat["Volatility20"].iloc[-1]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Last Closing Price", f"{CURRENCY}{last['Close']:,.2f}",
              f"{last['Close'] - prev['Close']:+,.2f}")
    k2.metric("Daily Return", f"{daily_ret:+.2f}%")
    k3.metric("20-Day Volatility", f"{vol20:.2f}%")
    k4.metric("Current Volume", f"{int(last['Volume']):,}")

    rng = st.radio("Range", ["3M", "6M", "1Y", "All"], index=2, horizontal=True)
    n_days = {"3M": 63, "6M": 126, "1Y": 252, "All": len(df)}[rng]
    view = df.tail(n_days).set_index("Date")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Closing Price")
        st.line_chart(view["Close"])
    with c2:
        st.subheader("Volume")
        st.bar_chart(view["Volume"])

# =========================================================
# ② STREAM ANALYTICS
# =========================================================
with tab2:
    view_days = st.select_slider("Chart window (trading days)", [60, 120, 250], value=120)
    v = feat.tail(view_days).set_index("Date")

    # ---- Rolling means + MA crossover
    st.subheader("Rolling Mean and MA5 vs MA20")
    ma5, ma20 = feat["MA5"].iloc[-1], feat["MA20"].iloc[-1]
    diff = ma5 - ma20
    uptrend = diff > 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MA5 (5-day mean)", f"{CURRENCY}{ma5:,.2f}")
    m2.metric("MA20 (20-day mean)", f"{CURRENCY}{ma20:,.2f}")
    m3.metric("MA5 − MA20", f"{CURRENCY}{diff:+,.2f}")
    m4.metric("Current Trend", "Uptrend ▲" if uptrend else "Downtrend ▼",
              "MA5 above MA20" if uptrend else "MA5 below MA20",
              delta_color="normal" if uptrend else "inverse")

    sign = np.sign(feat["MA_Diff"].dropna())
    flips = sign[sign != sign.shift()].iloc[1:]
    if len(flips):
        idx = flips.index[-1]
        kind = "MA5 crossed above MA20" if sign.loc[idx] > 0 else "MA5 crossed below MA20"
        st.caption(f"Last crossover: {feat.loc[idx, 'Date']:%d %b %Y} — {kind}. "
                   "Descriptive analytics only, not a trading signal.")

    st.line_chart(v[["Close", "MA5", "MA20"]])
    st.caption("MA5 − MA20 difference (positive = short-term average above long-term average)")
    st.bar_chart(v["MA_Diff"])

    # ---- Volatility + rolling volume
    g1, g2 = st.columns(2)
    with g1:
        st.subheader("20-Day Rolling Volatility (%)")
        st.line_chart(v["Volatility20"])
        st.caption("Standard deviation of daily returns over the last 20 days.")
    with g2:
        st.subheader("5-Day / 20-Day Rolling Volume")
        st.line_chart(v[["Volume_MA5", "Volume_MA20"]].rename(
            columns={"Volume_MA5": "5-day avg volume", "Volume_MA20": "20-day avg volume"}))

    # ---- Streaming simulation
    st.divider()
    st.subheader("Streaming Simulation")
    st.caption(
        f"Streaming-style analytics: the latest {STREAM_WINDOW} observations are processed "
        "one at a time, updating the rolling metrics after each observation."
    )
    stream = run_stream(df)

    cA, cB = st.columns([1, 3])
    with cA:
        replay = st.button("▶ Replay stream", width="stretch")
    with cB:
        step = st.slider("Observation", 1, len(stream), len(stream),
                         help="Drag to inspect the stream state after each observation.")

    panel = st.empty()

    def render_state(i: int):
        s = stream.iloc[i - 1]
        vol_txt = "n/a" if pd.isna(s["Rolling Volatility (%)"]) else f"{s['Rolling Volatility (%)']:.2f}%"
        with panel.container():
            st.markdown(
                f"**Streaming Window: {STREAM_WINDOW} observations** &nbsp;·&nbsp; "
                f"observation {i}/{len(stream)} &nbsp;·&nbsp; {s['Date']:%d %b %Y} "
                f"&nbsp;·&nbsp; window filled: {int(s['Window Size'])}/{STREAM_WINDOW}"
            )
            a, b, c, d_, e = st.columns(5)
            a.metric("Current Price", f"{CURRENCY}{s['Current Price']:,.2f}")
            b.metric("Rolling Mean", f"{CURRENCY}{s['Rolling Mean']:,.2f}")
            c.metric("Rolling Volatility", vol_txt)
            d_.metric("Current Volume", f"{int(s['Current Volume']):,}")
            e.metric("Daily Return", f"{s['Daily Return (%)']:+.2f}%")

    if replay:
        for i in range(1, len(stream) + 1):
            render_state(i)
            time.sleep(0.08)
    else:
        render_state(step)

    st.line_chart(stream.set_index("Date")[["Current Price", "Rolling Mean"]])

# =========================================================
# ③ ML PREDICTION
# =========================================================
with tab3:
    tr, te = bundle["train"], bundle["test"]
    st.info(
        f"**Chronological split (no shuffling).** Train: {tr['Date'].iloc[0]:%d %b %Y} → "
        f"{tr['Date'].iloc[-1]:%d %b %Y} ({len(tr)} days). "
        f"Test: {te['Date'].iloc[0]:%d %b %Y} → {te['Date'].iloc[-1]:%d %b %Y} ({len(te)} days). "
        "The scaler and all models are fitted on the training period only."
    )

    # ---- Model comparison
    st.subheader("Model Comparison (test set)")
    res = bundle["results"]
    st.dataframe(
        res.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.4f}"})
                 .highlight_min(subset=["MAE", "RMSE"], color="#c8e6c9")
                 .highlight_max(subset=["R²"], color="#c8e6c9"),
        hide_index=True, width="stretch",
    )
    st.subheader("RMSE by Model")
    st.bar_chart(res.set_index("Model")["RMSE"], sort=False)
    st.caption("Lower RMSE / MAE and higher R² are better.")

    # ---- Actual vs predicted (indexed by the date being predicted = next trading day)
    st.subheader(f"Actual vs Predicted Closing Price — {selected_model}")
    target_dates = te["Date"].shift(-1)
    target_dates.iloc[-1] = te["Date"].iloc[-1] + pd.offsets.BDay(1)
    avp = pd.DataFrame(
        {"Actual Closing Price": te["Target"].to_numpy(),
         "Predicted Closing Price": bundle["preds"][selected_model]},
        index=pd.Index(target_dates.to_numpy(), name="Date"),
    )
    st.line_chart(avp)
    st.caption("Test-period comparison on data the models never saw during training.")

    # ---- Next-day prediction
    st.subheader("Next-Day Prediction")
    model = bundle["models"][selected_model]
    pred = float(model.predict(bundle["X_latest"])[0])
    last_close = float(bundle["latest_feat"]["Close"].iloc[0])
    last_date = bundle["latest_feat"]["Date"].iloc[0]
    chg = pred - last_close

    p1, a1, p2, a2, p3 = st.columns([3, 1, 3, 1, 3])
    p1.metric(f"Last Closing Price ({last_date:%d %b %Y})", f"{CURRENCY}{last_close:,.2f}")
    a1.markdown("<h2 style='text-align:center'>→</h2>", unsafe_allow_html=True)
    p2.metric("Selected Ensemble Model", selected_model)
    a2.markdown("<h2 style='text-align:center'>→</h2>", unsafe_allow_html=True)
    p3.metric("Predicted Next-Day Closing Price", f"{CURRENCY}{pred:,.2f}",
              f"{chg:+,.2f} ({chg / last_close * 100:+.2f}%)")
    st.caption("Target: next trading day's closing price. Model output is an estimate")

# =========================================================
# ④ EXPLAINABILITY
# =========================================================
with tab4:
    st.subheader(f"SHAP Feature Importance — {selected_model}")
    sv_global, sv_latest = compute_shap(bundle, selected_model)

    mean_abs = pd.Series(np.abs(sv_global.values).mean(axis=0), index=FEATURE_COLUMNS)
    mean_abs = mean_abs.sort_values(ascending=False)

    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**Top features overall** (mean |SHAP|, last 40 test days)")
        st.bar_chart(mean_abs.head(8), horizontal=True, sort=False)
    with e2:
        st.markdown("**Top features for the next-day prediction**")
        contrib = pd.Series(sv_latest.values[0], index=FEATURE_COLUMNS)
        contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)
        st.bar_chart(contrib, horizontal=True, sort=False)
        st.caption(
            f"Positive = pushed the predicted price up; negative = pushed it down "
            f"(in ₹, relative to the average prediction {CURRENCY}{float(sv_latest.base_values[0]):,.2f})."
        )

    st.caption("SHAP is computed with a model-agnostic permutation explainer directly on the "
               "selected ensemble, so the contributions describe the ensemble's final output.")