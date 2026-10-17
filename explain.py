"""
explain_model.py
------------------
Explainable AI for the stock ensemble:

1. Meta-learner weights   -> how much the stacking ensemble trusts each base model
2. Native feature importances (Random Forest / Gradient Boosting) and
   coefficients (Linear / Ridge) -> which engineered features drive predictions
3. SHAP values             -> per-feature contribution, both globally (which
   features matter across all predictions) and locally (why THIS prediction,
   e.g. tomorrow's forecast, came out the way it did)

Requires: pip install shap
Run:
    python explain_model.py
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from ensemble_model import (
    load_and_prepare, split, get_base_models, FEATURE_COLUMNS,
)

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "explainability_plots"


def fit_everything():
    """Train the scaler + all base models + stacking meta-learner once,
    and return everything needed for explanations."""
    df = load_and_prepare()
    X_train, X_test, y_train, y_test = split(df)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    base_models = get_base_models()
    fitted = {}
    base_preds_train = {}
    for name, model in base_models:
        model.fit(X_train_s, y_train)
        fitted[name] = model
        base_preds_train[name] = model.predict(X_train_s)

    # Meta-learner: same idea as StackingRegressor's final_estimator,
    # fit directly on base models' train predictions so we can read its
    # coefficients afterward (StackingRegressor hides this internally).
    meta_X_train = pd.DataFrame(base_preds_train)
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(meta_X_train, y_train)

    return {
        "df": df, "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "scaler": scaler, "fitted": fitted, "meta_model": meta_model,
    }


# =========================================================
# 1. META-LEARNER WEIGHTS
# =========================================================
def explain_meta_weights(ctx):
    names = list(ctx["fitted"].keys())
    weights = ctx["meta_model"].coef_
    print("\n" + "=" * 60)
    print("META-LEARNER WEIGHTS (how much stacking trusts each model)")
    print("=" * 60)
    for name, w in sorted(zip(names, weights), key=lambda x: -abs(x[1])):
        print(f"  {name:6s} : {w:+.4f}")
    print(f"  intercept: {ctx['meta_model'].intercept_:.4f}")
    return dict(zip(names, weights))


# =========================================================
# 2. NATIVE IMPORTANCES / COEFFICIENTS
# =========================================================
def explain_native(ctx):
    print("\n" + "=" * 60)
    print("NATIVE FEATURE IMPORTANCE / COEFFICIENTS")
    print("=" * 60)

    out = {}
    for name, model in ctx["fitted"].items():
        if hasattr(model, "feature_importances_"):
            vals = model.feature_importances_
        elif hasattr(model, "coef_"):
            vals = model.coef_
        else:
            continue  # e.g. KNN has neither
        ranked = sorted(zip(FEATURE_COLUMNS, vals), key=lambda x: -abs(x[1]))
        print(f"\n{name}:")
        for feat, v in ranked:
            print(f"  {feat:15s} : {v:+.4f}")
        out[name] = dict(ranked)
    return out


# =========================================================
# 3. SHAP (global + local)
# =========================================================
def explain_shap(ctx, save_plots=True):
    print("\n" + "=" * 60)
    print("SHAP EXPLANATIONS")
    print("=" * 60)

    if save_plots:
        PLOTS_DIR.mkdir(exist_ok=True)

    X_test_s = ctx["scaler"].transform(ctx["X_test"])
    X_test_df = pd.DataFrame(X_test_s, columns=FEATURE_COLUMNS)

    shap_values_by_model = {}

    for name, model in ctx["fitted"].items():
        if name in ("rf", "gb"):
            explainer = shap.TreeExplainer(model)
        elif name in ("lr", "ridge"):
            explainer = shap.LinearExplainer(model, X_test_df)
        else:
            continue  # skip KNN: no fast SHAP explainer, and it's the weakest model anyway

        sv = explainer(X_test_df)
        shap_values_by_model[name] = sv

        # Global importance: mean |SHAP value| per feature
        mean_abs = np.abs(sv.values).mean(axis=0)
        ranked = sorted(zip(FEATURE_COLUMNS, mean_abs), key=lambda x: -x[1])
        print(f"\n{name} — mean |SHAP value| per feature:")
        for feat, v in ranked:
            print(f"  {feat:15s} : {v:.4f}")

        if save_plots:
            plt.figure()
            shap.plots.bar(sv, show=False)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f"shap_summary_{name}.png")
            plt.close()

    return shap_values_by_model


def explain_latest_prediction(ctx, model_name="ridge", save_plot=True):
    """Local explanation: why did `model_name` predict what it did for the
    most recent row (i.e. tomorrow's close)?"""
    model = ctx["fitted"][model_name]
    latest_row = ctx["df"][FEATURE_COLUMNS].iloc[[-1]]
    latest_scaled = ctx["scaler"].transform(latest_row)
    latest_df = pd.DataFrame(latest_scaled, columns=FEATURE_COLUMNS)

    if model_name in ("rf", "gb"):
        explainer = shap.TreeExplainer(model)
    elif model_name in ("lr", "ridge"):
        X_test_s = ctx["scaler"].transform(ctx["X_test"])
        explainer = shap.LinearExplainer(model, pd.DataFrame(X_test_s, columns=FEATURE_COLUMNS))
    else:
        print(f"No fast SHAP explainer configured for '{model_name}'.")
        return None

    sv = explainer(latest_df)
    prediction = model.predict(latest_scaled)[0]

    print("\n" + "=" * 60)
    print(f"WHY DID '{model_name}' PREDICT {prediction:.2f} FOR THE NEXT CLOSE?")
    print("=" * 60)
    contributions = sorted(zip(FEATURE_COLUMNS, sv.values[0]), key=lambda x: -abs(x[1]))
    for feat, contrib in contributions:
        direction = "pushed price UP" if contrib > 0 else "pushed price DOWN"
        print(f"  {feat:15s} : {contrib:+.3f}  ({direction})")

    if save_plot:
        PLOTS_DIR.mkdir(exist_ok=True)
        plt.figure()
        shap.plots.waterfall(sv[0], show=False)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"shap_latest_{model_name}.png")
        plt.close()

    return sv


def main():
    ctx = fit_everything()
    explain_meta_weights(ctx)
    explain_native(ctx)
    explain_shap(ctx)
    explain_latest_prediction(ctx, model_name="ridge")
    print(f"\nPlots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()