"""
Model E -- gradient-boosted trees (sklearn HistGradientBoostingRegressor) on
the same lag/covariate features as Model B, predicting log1p(transaction_count).

Rationale: T=8 months per substance is too short a sequence for a deep model
to learn much from (see model_lstm.py's caveats), but a shallow nonlinear
model on the same tabular lag features Model B already uses is cheap to try
and may pick up nonlinearities the linear panel models can't. Unlike the
linear models, HistGradientBoostingRegressor handles missing covariates
natively, so no covariate-driven row drops are needed here.

Same t=3..6 train / t=7 validate / t=8 test / t=9 forecast structure as
model_ab.py, reusing its data loading, splitting, and scoring. model_ab.py
itself is not modified.

Outputs written to Code results/model_gbm_results.txt and Forecasted results/model_gbm_*.csv.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from model_ab import (
    load_panel, split_substances, fit_model_a, fit_model_d_ab_gmm,
    compute_entity_effects, score_counts, oos_r2, SEED, COVARS, _train_window,
)
from combine_comparisons import record_comparison

FEAT_COLS = ["y_l1", "y_l2"] + COVARS


def build_xy(df, subs, t_range):
    d = df[df["entity"].isin(subs) & df["t"].between(*t_range)].dropna(subset=["y", "y_l1", "y_l2"]).copy()
    return d, d[FEAT_COLS], d["y"]


def predict_gbm(model, d):
    yhat_log = model.predict(d[FEAT_COLS])
    return np.clip(np.expm1(yhat_log), 0, None)


def main():
    lines = []
    def out(s=""):
        print(s)
        lines.append(str(s))

    os.makedirs("Code results", exist_ok=True)
    os.makedirs("Forecasted results/model_gbm", exist_ok=True)

    df = load_panel()
    train_subs, holdout_subs = split_substances(df)
    all_subs = set(df["entity"])
    T = int(df["t"].max())
    TRAIN_START, TRAIN_END = _train_window(df)
    VAL_T, TEST_T, FORECAST_T = T - 1, T, T + 1
    FORECAST_MONTH = "2024-09"
    out(f"Panel: {len(df)} rows, {df['drug_substance'].nunique()} substances, T={T}")
    out()

    out("=" * 70)
    out("MODEL E -- HistGradientBoostingRegressor on Model B's lag/covariate features")
    out("=" * 70)

    d_train, X_train, y_train = build_xy(df, train_subs, (TRAIN_START, TRAIN_END))
    model = HistGradientBoostingRegressor(random_state=SEED, early_stopping=True, validation_fraction=0.15)
    model.fit(X_train, y_train)
    out(f"Trained on {len(d_train)} rows, {len(FEAT_COLS)} features: {FEAT_COLS}")
    out(f"n_iter (after early stopping): {model.n_iter_}")
    out()

    # ---------------- Baselines for comparison: Model A (AR1), Model D (AB-GMM) ----------------
    res_a = fit_model_a(df, train_subs, lags=1)
    coefs_a = res_a.params[["y_l1"]].values
    alpha_a = compute_entity_effects(df, coefs_a, ["y_l1"], t_range=(TRAIN_START, TRAIN_END))

    res_d, _frame_d, _diag_d = fit_model_d_ab_gmm(df, train_subs)
    coefs_d = np.array([res_d.params["dy1"]])
    alpha_d = compute_entity_effects(df, coefs_d, ["y_l1"], t_range=(TRAIN_START, TRAIN_END))

    def predict_loglinear(d, alpha, coefs):
        yhat_log = d["entity"].map(alpha).to_numpy(dtype=float) + d["y_l1"].to_numpy(dtype=float) * coefs[0]
        return np.clip(np.expm1(yhat_log), 0, None)

    # ---------------- Test (t=T) ----------------
    d8, _, _ = build_xy(df, all_subs, (TEST_T, TEST_T))
    d8 = d8[d8["entity"].isin(alpha_a.index) & d8["entity"].isin(alpha_d.index)]
    y_true_ct = np.expm1(d8["y"].to_numpy(dtype=float))
    naive_ct = np.expm1(d8["y_l1"].to_numpy(dtype=float))

    pred_gbm_t8 = predict_gbm(model, d8)
    pred_a_t8 = predict_loglinear(d8, alpha_a, coefs_a)
    pred_d_t8 = predict_loglinear(d8, alpha_d, coefs_d)

    scores = {
        "Model E (GBM)": score_counts(y_true_ct, pred_gbm_t8, naive_ct),
        "Model A (AR1)": score_counts(y_true_ct, pred_a_t8, naive_ct),
        "Model D (AB-GMM)": score_counts(y_true_ct, pred_d_t8, naive_ct),
    }
    oosr2 = {
        "Model E (GBM)": oos_r2(y_true_ct, pred_gbm_t8),
        "Model A (AR1)": oos_r2(y_true_ct, pred_a_t8),
        "Model D (AB-GMM)": oos_r2(y_true_ct, pred_d_t8),
    }
    for name, s in scores.items():
        out(f"{name} test (t={TEST_T}): {s}  oos_r2={oosr2[name]:.4f}")
    out()

    comp_df = pd.DataFrame([{"model": name, **s, "oos_r2": oosr2[name]} for name, s in scores.items()])
    record_comparison("Model E (GBM)", "Model E (GBM)", comp_df)
    out()

    test_out = pd.DataFrame({
        "drug_substance": d8["drug_substance"].values,
        "t": TEST_T,
        "actual_transaction_count": y_true_ct,
        "naive_forecast": np.round(naive_ct, 1),
        "model_e_gbm_forecast": np.round(pred_gbm_t8, 1),
        "model_a_forecast": np.round(pred_a_t8, 1),
        "model_d_forecast": np.round(pred_d_t8, 1),
    })
    test_out.to_csv("Forecasted results/model_gbm/model_gbm_test_t8.csv", index=False)
    out(f"Saved -> Forecasted results/model_gbm/model_gbm_test_t8.csv ({len(test_out)} rows)")
    out()

    # ---------------- Forecast (t=T+1) ----------------
    last = df[df["t"] == TEST_T].dropna(subset=["y", "y_l1"]).copy()
    last["y_l2"] = last["y_l1"].values
    last["y_l1"] = last["y"].values  # roll forward: forecast month's y_l1 is T's y, y_l2 is T's old y_l1
    # covariate _l1 columns must reflect t=T's own values, same transform load_panel applies
    # (avg_deal_size_l1 is log1p'd; the rest are used as-is) -- mirrors model_ab.py's _raw_covar_at_t8.
    for c in COVARS:
        base = c.replace("_l1", "")
        vals = last[base].values
        last[c] = np.log1p(vals) if base == "avg_deal_size" else vals

    pred_gbm_next = predict_gbm(model, last)
    next_out = pd.DataFrame({
        "drug_substance": last["drug_substance"].values,
        "t": FORECAST_T,
        "forecast_month": FORECAST_MONTH,
        "last_actual_transaction_count_t8": last["transaction_count"].values,
        "model_e_gbm_forecast": np.round(pred_gbm_next, 1),
    })
    next_out = next_out.sort_values("model_e_gbm_forecast", ascending=False)
    next_out.to_csv("Forecasted results/model_gbm/model_gbm_forecast_t9.csv", index=False)
    out(f"Saved -> Forecasted results/model_gbm/model_gbm_forecast_t9.csv ({len(next_out)} rows)")

    with open("Code results/model_gbm_results.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    out("Saved -> Code results/model_gbm_results.txt")


if __name__ == "__main__":
    main()
