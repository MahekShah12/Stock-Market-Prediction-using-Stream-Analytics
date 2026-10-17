"""
app.py
------
Streamlit dashboard for the stock ensemble project.

Tabs:
    1. Overview     - price/volume charts, latest prediction
    2. Model Comparison - RMSE/MAE/R2 across base models + voting + stacking
    3. Explainability   - meta-learner weights, feature importances, SHAP
                          (global + "why tomorrow's prediction?")

Requires: pip install streamlit shap matplotlib
Run:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.linear_model import Ridge
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ensemble_model import load_and_prepare, split, get_base_models, FEATURE_COLUMNS
from explain import fit_everything, explain_meta_weights, explain_native

st.set_page_config(page_title="Stock Ensemble Explorer", layout="wide")


# =========================================================
# CACHED TRAINING (so the UI doesn't retrain on every click)
# =========================================================
@st.cache_resource
def train_all():
    ctx = fit_everything()  # scaler + base models + meta-learner (from explain_model.py)

    X_train_s = ctx["scaler"].transform(ctx["X_train"])
    X_test_s = ctx["scaler"].transform(ctx["X_test"])
    base_models = get_base_models()

    voting = VotingRegressor(estimators=base_models)
    voting.fit(X_train_s, ctx["y_train"])

    stacking = StackingRegressor(
        estimators=base_models, final_estimator=Ridge(alpha=1.0), cv=5,
    )
    stacking.fit(X_train_s, ctx["y_train"])

    results = []
    for name, model in ctx["fitted"].items():
        preds = model.predict(X_test_s)
        results.append(_score_row(name, ctx["y_test"], preds))
    results.append(_score_row("Voting", ctx["y_test"], voting.predict(X_test_s)))
    results.append(_score_row("Stacking", ctx["y_test"], stacking.predict(X_test_s)))
    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)

    return ctx, voting, stacking, results_df


def _score_row(name, y_test, preds):
    return {
        "Model": name,
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2": r2_score(y_test, preds),
    }


ctx, voting_model, stacking_model, results_df = train_all()
df = ctx["df"]

st.title("📈 Stock Ensemble Explorer")
st.caption(f"Ticker data loaded from historical_data.csv — {len(df)} rows after feature engineering")

tab1, tab2, tab3 = st.tabs(["Overview", "Model Comparison", "Explainability"])

# =========================================================
# TAB 1: OVERVIEW
# =========================================================
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Closing Price")
        st.line_chart(df.set_index("Date")["Close"])
    with col2:
        st.subheader("Volume")
        st.bar_chart(df.set_index("Date")["Volume"])

    st.subheader("Next-Close Prediction")
    latest_row = df[FEATURE_COLUMNS].iloc[[-1]]
    latest_scaled = ctx["scaler"].transform(latest_row)
    last_close = df["Close"].iloc[-1]

    pred_choice = st.selectbox(
        "Predict using:",
        ["Stacking (recommended)", "Voting"] + list(ctx["fitted"].keys()),
    )
    if pred_choice == "Stacking (recommended)":
        prediction = stacking_model.predict(latest_scaled)[0]
    elif pred_choice == "Voting":
        prediction = voting_model.predict(latest_scaled)[0]
    else:
        prediction = ctx["fitted"][pred_choice].predict(latest_scaled)[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Last known close", f"{last_close:.2f}")
    c2.metric("Predicted next close", f"{prediction:.2f}", delta=f"{prediction - last_close:+.2f}")
    c3.metric("Model used", pred_choice)

# =========================================================
# TAB 2: MODEL COMPARISON
# =========================================================
with tab2:
    st.subheader("RMSE / MAE / R² across all models")
    st.dataframe(
        results_df.style.highlight_min(subset=["MAE", "RMSE"], color="#1e5631")
                          .highlight_max(subset=["R2"], color="#1e5631"),
        use_container_width=True,
    )
    st.bar_chart(results_df.set_index("Model")["RMSE"])
    st.caption("Lower RMSE/MAE and higher R² are better. Stacking combines all 5 base "
               "models via a Ridge meta-learner; Voting simply averages their predictions.")

# =========================================================
# TAB 3: EXPLAINABILITY
# =========================================================
with tab3:
    st.subheader("Meta-learner weights (Stacking)")
    st.caption("How much the stacking ensemble trusts each base model's prediction.")
    weights = explain_meta_weights(ctx)  # also prints to server console
    weights_df = pd.DataFrame(weights.items(), columns=["Base model", "Meta weight"]) \
        .sort_values("Meta weight", key=abs, ascending=False)
    st.bar_chart(weights_df.set_index("Base model")["Meta weight"])

    st.subheader("Native feature importance / coefficients")
    model_pick = st.selectbox("Model:", [n for n in ctx["fitted"] if n != "knn"])
    model = ctx["fitted"][model_pick]
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    else:
        vals = model.coef_
    imp_df = pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": vals}) \
        .sort_values("Importance", key=abs, ascending=False)
    st.bar_chart(imp_df.set_index("Feature")["Importance"])

    st.subheader("SHAP: why tomorrow's prediction?")
    shap_model_pick = st.selectbox("Explain which model's prediction:", ["ridge", "lr", "rf", "gb"])
    with st.spinner("Computing SHAP values..."):
        model = ctx["fitted"][shap_model_pick]
        latest_df = pd.DataFrame(latest_scaled, columns=FEATURE_COLUMNS)

        if shap_model_pick in ("rf", "gb"):
            explainer = shap.TreeExplainer(model)
        else:
            X_test_s = ctx["scaler"].transform(ctx["X_test"])
            explainer = shap.LinearExplainer(model, pd.DataFrame(X_test_s, columns=FEATURE_COLUMNS))

        sv = explainer(latest_df)
        contrib_df = pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "SHAP contribution": sv.values[0],
        }).sort_values("SHAP contribution", key=abs, ascending=False)

    st.dataframe(contrib_df, use_container_width=True)
    fig, ax = plt.subplots()
    shap.plots.waterfall(sv[0], show=False)
    st.pyplot(fig)
    st.caption("Positive = pushed the predicted next-close price up. Negative = pushed it down.")