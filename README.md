# Stock Market Prediction using Stream Analytics

Predicts the next closing price of **RELIANCE.NS** by combining an offline-trained
ensemble ML model with a simulated real-time market data stream (rolling-window
analytics) inside a Streamlit dashboard.

## Project pipeline

The project is organized in stages. Run them in order the first time; after that,
only the final app needs to be run (it loads the already-trained model).

| Stage | Script | What it does |
|---|---|---|
| 1. Data ingestion | `stock/stage1.py` | Downloads daily OHLCV data for RELIANCE.NS from Yahoo Finance, cleans it, and saves `stock/data/historical_data.csv` plus two exploratory charts. |
| 2. Baseline modeling | `stock/model.py` | Engineers lag/moving-average/return features and trains 5 individual regressors (Linear, Ridge, KNN, Random Forest, Gradient Boosting), keeping the best single model (`stock/models/best_model.pkl`). |
| 3. Ensemble modeling | `ensemble_model.py` | Builds on the same features to train a **Voting** ensemble and a **Stacking** ensemble (Ridge meta-learner) over the 5 base models, and saves the result to `stock/models/ensemble_model.pkl`. This is the model used everywhere downstream. |
| 4. Explainability | `explain.py` | Computes meta-learner weights, native feature importances, and SHAP values (global + "why did it predict this for tomorrow?"), saving plots to `explainability_plots/`. |
| 5. Analysis dashboard | `app.py` | Streamlit app with three tabs: Overview, Model Comparison (RMSE/MAE/R²), and Explainability (SHAP). Retrains models live from the CSV so the numbers are always reproducible. |
| **Final deliverable** | `stock_stream_analytics.py` | The stream-analytics dashboard. Loads the pre-trained ensemble model (no retraining), simulates a market event stream with a configurable rolling window, and shows live rolling stats, forecasts, and charts. |

## How to run

```bash
pip install -r requirements.txt

# One-time setup (only needed if you want to regenerate data/model from scratch)
python stock/stage1.py
python stock/model.py
python ensemble_model.py
python explain.py

# Analysis dashboard
streamlit run app.py

# Main stream analytics app
streamlit run stock_stream_analytics.py
```

## Folder structure

```
.
├── app.py                       # Model comparison + explainability dashboard
├── ensemble_model.py            # Trains the Voting/Stacking ensemble
├── explain.py                   # SHAP / feature-importance explainability
├── stock_stream_analytics.py    # MAIN APP: simulated stream + live forecasting
├── explainability_plots/        # Saved SHAP plots
├── requirements.txt
└── stock/
    ├── stage1.py                # Data download + cleaning
    ├── model.py                 # Baseline model training
    ├── data/                    # historical_data.csv + charts
    └── models/                  # best_model.pkl, ensemble_model.pkl
```

## Notes

- The "stream" in Stream Analytics is a simulated replay of historical daily
  bars processed sequentially through rolling-window statistics and the
  ensemble model — it is an academic/project-style stream simulation, not a
  live exchange tick feed.
- All Python virtual environment files and compiled bytecode caches were
  removed from this package; just `pip install -r requirements.txt` instead.
