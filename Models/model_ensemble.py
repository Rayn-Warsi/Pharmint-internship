"""
Model F -- Arellano-Bond GMM (model_ab's Model D) + XGBoost residual correction.
XGBoost is trained on the GMM's own log-scale residuals, i.e. it learns the
systematic errors the linear GMM makes as a function of the lag/covariate
features, and that correction is added back to the GMM's prediction.

Model G -- GMM + XGBoost + LightGBM ensemble. Three independently-fit
predictors (GMM log-linear, XGBoost, LightGBM -- the latter two direct
regressors on y, same features as model_gbm.py's Model E) averaged on the
count scale.

Same t=3..6 train / t=7 validate / t=8 test / t=9 forecast structure as
model_ab.py, reusing its data loading, splitting, and scoring. Neither
model_ab.py nor model_gbm.py is modified.

Outputs written to Code results/model_ensemble_results.txt and
Forecasted results/model_ensemble/*.csv, and the two models' key metrics are
appended as columns to Forecasted results/model_comparision.csv.
"""

import os
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from model_ab import (
    load_panel, load_panel_country, split_substances, fit_model_d_ab_gmm,
    compute_entity_effects, score_counts, oos_r2, SEED, COVARS, COVARS_COUNTRY,
    _train_window,
)

FEAT_COLS = ["y_l1", "y_l2"] + COVARS


def gmm_predict_log(d, alpha, phi1):
    return d["entity"].map(alpha).to_numpy(dtype=float) + d["y_l1"].to_numpy(dtype=float) * phi1


def build_xy(df, subs, t_range):
    d = df[df["entity"].isin(subs) & df["t"].between(*t_range)].dropna(subset=["y", "y_l1", "y_l2"]).copy()
    return d, d[FEAT_COLS], d["y"]


def main():
    lines = []
    def out(s=""):
        print(s)
        lines.append(str(s))

    os.makedirs("Code results", exist_ok=True)
    os.makedirs("Forecasted results/model_ensemble", exist_ok=True)

    df = load_panel()
    train_subs, holdout_subs = split_substances(df)
    all_subs = set(df["entity"])
    T = int(df["t"].max())
    TRAIN_START, TRAIN_END = _train_window(df)
    VAL_T, TEST_T, FORECAST_T = T - 1, T, T + 1
    FORECAST_MONTH = "2024-09"
    out(f"Panel: {len(df)} rows, {df['drug_substance'].nunique()} substances, T={T}")
    out()

    # ---------------- Base GMM (Model D, AR(1), AB difference GMM) ----------------
    res_d, _frame_d, _diag_d = fit_model_d_ab_gmm(df, train_subs)
    phi1_d = res_d.params["dy1"]
    alpha_d = compute_entity_effects(df, np.array([phi1_d]), ["y_l1"], t_range=(TRAIN_START, TRAIN_END))
    out(f"Base GMM phi1: {phi1_d:+.4f}")
    out()

    # ---------------- Model F: GMM + XGBoost residual correction ----------------
    out("=" * 70)
    out("MODEL F -- Arellano-Bond GMM + XGBoost residual correction")
    out("=" * 70)
    d_train, X_train, y_train = build_xy(df, train_subs, (TRAIN_START, TRAIN_END))
    d_train = d_train[d_train["entity"].isin(alpha_d.index)]
    X_train = X_train.loc[d_train.index]
    y_train = y_train.loc[d_train.index]

    gmm_pred_train = gmm_predict_log(d_train, alpha_d, phi1_d)
    resid_train = y_train.to_numpy(dtype=float) - gmm_pred_train

    xgb_resid = XGBRegressor(random_state=SEED, n_estimators=300, max_depth=3,
                              learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
    xgb_resid.fit(X_train, resid_train)
    out(f"Residual-correction XGBoost trained on {len(d_train)} rows, features: {FEAT_COLS}")
    out()

    def predict_f(d):
        d = d[d["entity"].isin(alpha_d.index)]
        gmm_log = gmm_predict_log(d, alpha_d, phi1_d)
        resid_hat = xgb_resid.predict(d[FEAT_COLS])
        return d, np.clip(np.expm1(gmm_log + resid_hat), 0, None)

    # ---------------- Direct XGBoost and LightGBM regressors (for Model G) ----------------
    xgb_direct = XGBRegressor(random_state=SEED, n_estimators=300, max_depth=3,
                               learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
    xgb_direct.fit(X_train, y_train)

    lgbm_direct = LGBMRegressor(random_state=SEED, n_estimators=300, max_depth=3,
                                 learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbose=-1)
    lgbm_direct.fit(X_train, y_train)
    out("Direct XGBoost and LightGBM regressors trained on the same rows/features.")
    out()

    def predict_g(d):
        d = d[d["entity"].isin(alpha_d.index)]
        gmm_ct = np.clip(np.expm1(gmm_predict_log(d, alpha_d, phi1_d)), 0, None)
        xgb_ct = np.clip(np.expm1(xgb_direct.predict(d[FEAT_COLS])), 0, None)
        lgbm_ct = np.clip(np.expm1(lgbm_direct.predict(d[FEAT_COLS])), 0, None)
        return d, (gmm_ct + xgb_ct + lgbm_ct) / 3.0

    # ---------------- Validation (t=7) and test (t=8) ----------------
    def score_at(t_target, subs):
        d_t, _, _ = build_xy(df, subs, (t_target, t_target))
        d_f, pred_f = predict_f(d_t)
        d_g, pred_g = predict_g(d_t)
        y_true_f = np.expm1(d_f["y"].to_numpy(dtype=float))
        naive_f = np.expm1(d_f["y_l1"].to_numpy(dtype=float))
        y_true_g = np.expm1(d_g["y"].to_numpy(dtype=float))
        naive_g = np.expm1(d_g["y_l1"].to_numpy(dtype=float))
        return score_counts(y_true_f, pred_f, naive_f), score_counts(y_true_g, pred_g, naive_g)

    val_f_train, val_g_train = score_at(VAL_T, train_subs)
    val_f_hold, val_g_hold = score_at(VAL_T, holdout_subs)
    test_f, test_g = score_at(TEST_T, all_subs)

    d8, _, _ = build_xy(df, all_subs, (TEST_T, TEST_T))
    d8_f, pred_f_t8 = predict_f(d8)
    d8_g, pred_g_t8 = predict_g(d8)
    oos_f = oos_r2(np.expm1(d8_f["y"].to_numpy(dtype=float)), pred_f_t8)
    oos_g = oos_r2(np.expm1(d8_g["y"].to_numpy(dtype=float)), pred_g_t8)

    out(f"Model F test (t={TEST_T}): {test_f}  oos_r2={oos_f:.4f}")
    out(f"Model G test (t={TEST_T}): {test_g}  oos_r2={oos_g:.4f}")
    out()

    test_out = pd.DataFrame({
        "drug_substance": d8_f["drug_substance"].values,
        "t": TEST_T,
        "actual_transaction_count": np.expm1(d8_f["y"].to_numpy(dtype=float)),
        "naive_forecast": np.round(np.expm1(d8_f["y_l1"].to_numpy(dtype=float)), 1),
        "model_f_gmm_xgb_forecast": np.round(pred_f_t8, 1),
        "model_g_ensemble_forecast": np.round(pred_g_t8, 1),
    })
    test_out.to_csv("Forecasted results/model_ensemble/model_ensemble_test_t8.csv", index=False)
    out(f"Saved -> Forecasted results/model_ensemble/model_ensemble_test_t8.csv ({len(test_out)} rows)")
    out()

    # ---------------- Forecast (t=T+1) ----------------
    last = df[df["t"] == TEST_T].dropna(subset=["y", "y_l1"]).copy()
    last["y_l2"] = last["y_l1"].values
    last["y_l1"] = last["y"].values  # roll forward: forecast month's y_l1 is T's y
    for c in COVARS:
        base = c.replace("_l1", "")
        vals = last[base].values
        last[c] = np.log1p(vals) if base == "avg_deal_size" else vals

    alpha_d_full = compute_entity_effects(df, np.array([phi1_d]), ["y_l1"], t_range=(1, T))
    last_f = last[last["entity"].isin(alpha_d_full.index)]
    gmm_log_next = gmm_predict_log(last_f, alpha_d_full, phi1_d)
    resid_next = xgb_resid.predict(last_f[FEAT_COLS])
    pred_f_next = np.clip(np.expm1(gmm_log_next + resid_next), 0, None)

    gmm_ct_next = np.clip(np.expm1(gmm_log_next), 0, None)
    xgb_ct_next = np.clip(np.expm1(xgb_direct.predict(last_f[FEAT_COLS])), 0, None)
    lgbm_ct_next = np.clip(np.expm1(lgbm_direct.predict(last_f[FEAT_COLS])), 0, None)
    pred_g_next = (gmm_ct_next + xgb_ct_next + lgbm_ct_next) / 3.0

    next_out = pd.DataFrame({
        "drug_substance": last_f["drug_substance"].values,
        "t": FORECAST_T,
        "forecast_month": FORECAST_MONTH,
        "last_actual_transaction_count_t8": last_f["transaction_count"].values,
        "model_f_gmm_xgb_forecast": np.round(pred_f_next, 1),
        "model_g_ensemble_forecast": np.round(pred_g_next, 1),
    })
    next_out = next_out.sort_values("model_g_ensemble_forecast", ascending=False)
    next_out.to_csv("Forecasted results/model_ensemble/model_ensemble_forecast_t9.csv", index=False)
    out(f"Saved -> Forecasted results/model_ensemble/model_ensemble_forecast_t9.csv ({len(next_out)} rows)")

    with open("Code results/model_ensemble_results.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    out("Saved -> Code results/model_ensemble_results.txt")

    # ---------------- Append key metrics to the main model_comparision.csv ----------------
    main_cmp_path = "Forecasted results/model_comparision.csv"
    main_cmp = pd.read_csv(main_cmp_path)

    # metric label strings must match model_ab.py's comparison_rows labels exactly --
    # those are now built with TEST_T/VAL_T (this file's are computed from the same
    # panel, so they agree) rather than the old hardcoded t=8/t=7.
    metric_map = [
        (f"test MASE, t={TEST_T}, all substances", test_f["mase"], test_g["mase"]),
        (f"test RMSE (log scale), t={TEST_T}, all substances", test_f["rmse_log"], test_g["rmse_log"]),
        (f"test RMSE (count scale), t={TEST_T}, all substances", test_f["rmse_count"], test_g["rmse_count"]),
        (f"test mean SIGNED error, t={TEST_T}", test_f["mean_signed_err"], test_g["mean_signed_err"]),
        (f"test total forecast, t={TEST_T}", test_f["total_pred"], test_g["total_pred"]),
        (f"test total error %, t={TEST_T}", 100.0 * (test_f["total_pred"] - test_f["total_actual"]) / test_f["total_actual"],
         100.0 * (test_g["total_pred"] - test_g["total_actual"]) / test_g["total_actual"]),
        (f"out-of-sample R2, t={TEST_T}, count scale", oos_f, oos_g),
        (f"validation MASE, t={VAL_T}, train-subs", val_f_train["mase"], val_g_train["mase"]),
        (f"validation MASE, t={VAL_T}, holdout-subs", val_f_hold["mase"], val_g_hold["mase"]),
    ]
    f_col = pd.Series(np.nan, index=main_cmp.index)
    g_col = pd.Series(np.nan, index=main_cmp.index)
    for metric, fval, gval in metric_map:
        mask = main_cmp["metric"] == metric
        f_col[mask] = fval
        g_col[mask] = gval

    main_cmp["Model F (GMM+XGB residual)"] = f_col
    main_cmp["Model G (GMM+XGB+LGBM ensemble)"] = g_col
    note_mask = main_cmp["Model F (GMM+XGB residual)"].isna() & main_cmp["Model G (GMM+XGB+LGBM ensemble)"].isna()
    main_cmp.loc[note_mask, "note"] = main_cmp.loc[note_mask, "note"].astype(str) + \
        " | Model F/G: not applicable (tree ensembles have no coefficients/instruments)"
    main_cmp.to_csv(main_cmp_path, index=False)
    out(f"Updated -> {main_cmp_path} with Model F / Model G columns")


# ---------------------------------------------------------------------------
# Country-grain Model F -- entity = (drug_substance, country_normalized) pair,
# same pattern as model_ab.py's main_country(). Model G (the 3-way ensemble)
# is drug-grain only -- the dashboard's Forecast Explorer only needs Model F.
#
# Unlike the drug-grain build_xy, only y/y_l1 are required (not y_l2): the
# base GMM is AR(1) (uses y_l1 only) and XGBoost handles missing features
# (y_l2, covariates) natively, so thin country pairs aren't dropped the way
# the linear Model B drops them.
# ---------------------------------------------------------------------------

FEAT_COLS_COUNTRY = ["y_l1", "y_l2"] + COVARS_COUNTRY


def main_country():
    lines = []
    def out(s=""):
        print(s)
        lines.append(str(s))

    os.makedirs("Code results", exist_ok=True)
    os.makedirs("Forecasted results/model_ensemble", exist_ok=True)

    df = load_panel_country()
    train_subs, holdout_subs = split_substances(df)
    T = int(df["t"].max())
    TRAIN_START, TRAIN_END = _train_window(df)
    TEST_T, FORECAST_T = T, T + 1
    FORECAST_MONTH = "2024-09"
    n_pairs = df[["drug_substance", "country_normalized"]].drop_duplicates().shape[0]
    out(f"Country panel: {len(df)} rows, {n_pairs} drug x country pairs, T={T}")
    out()

    res_d, _frame_d, _diag_d = fit_model_d_ab_gmm(df, train_subs)
    phi1_d = res_d.params["dy1"]
    alpha_d = compute_entity_effects(df, np.array([phi1_d]), ["y_l1"], t_range=(TRAIN_START, TRAIN_END))
    out(f"Base GMM (country grain) phi1: {phi1_d:+.4f}")
    out()

    out("=" * 70)
    out("MODEL F (country grain) -- Arellano-Bond GMM + XGBoost residual correction")
    out("=" * 70)

    d_train = df[df["entity"].isin(train_subs) & df["t"].between(TRAIN_START, TRAIN_END)].dropna(subset=["y", "y_l1"]).copy()
    d_train = d_train[d_train["entity"].isin(alpha_d.index)]
    resid_train = d_train["y"].to_numpy(dtype=float) - gmm_predict_log(d_train, alpha_d, phi1_d)

    xgb_resid_c = XGBRegressor(random_state=SEED, n_estimators=300, max_depth=3,
                                learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
    xgb_resid_c.fit(d_train[FEAT_COLS_COUNTRY], resid_train)
    out(f"Residual-correction XGBoost trained on {len(d_train)} rows, features: {FEAT_COLS_COUNTRY}")
    out()

    def predict_f_country(d, alpha):
        gmm_log = gmm_predict_log(d, alpha, phi1_d)
        resid_hat = xgb_resid_c.predict(d[FEAT_COLS_COUNTRY])
        return np.clip(np.expm1(gmm_log + resid_hat), 0, None)

    # ---------------- Model G (country grain) -- same 3-way GMM+XGB+LGBM
    # ensemble as the drug-grain build, now trained/predicted at the drug x
    # country pair grain so the Forecast Explorer can offer it per market. ----------------
    out("=" * 70)
    out("MODEL G (country grain) -- GMM + XGBoost + LightGBM ensemble")
    out("=" * 70)
    xgb_direct_c = XGBRegressor(random_state=SEED, n_estimators=300, max_depth=3,
                                 learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
    xgb_direct_c.fit(d_train[FEAT_COLS_COUNTRY], d_train["y"])

    lgbm_direct_c = LGBMRegressor(random_state=SEED, n_estimators=300, max_depth=3,
                                   learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbose=-1)
    lgbm_direct_c.fit(d_train[FEAT_COLS_COUNTRY], d_train["y"])
    out(f"Direct XGBoost and LightGBM regressors (country grain) trained on {len(d_train)} rows, "
        f"features: {FEAT_COLS_COUNTRY}")
    out()

    def predict_g_country(d, alpha):
        gmm_ct = np.clip(np.expm1(gmm_predict_log(d, alpha, phi1_d)), 0, None)
        xgb_ct = np.clip(np.expm1(xgb_direct_c.predict(d[FEAT_COLS_COUNTRY])), 0, None)
        lgbm_ct = np.clip(np.expm1(lgbm_direct_c.predict(d[FEAT_COLS_COUNTRY])), 0, None)
        return (gmm_ct + xgb_ct + lgbm_ct) / 3.0

    # ---------------- Test (t=T), pairs with own train-window history ----------------
    alpha_d_v = compute_entity_effects(df, np.array([phi1_d]), ["y_l1"], t_range=(TRAIN_START, TRAIN_END))
    test = df[df["t"] == TEST_T].dropna(subset=["y", "y_l1"]).copy()
    test = test[test["entity"].isin(alpha_d_v.index)]
    pred_f_test = predict_f_country(test, alpha_d_v)
    pred_g_test = predict_g_country(test, alpha_d_v)
    y_true_ct = test["transaction_count"].to_numpy(dtype=float)
    naive_ct = np.expm1(test["y_l1"].to_numpy(dtype=float))
    score_f = score_counts(y_true_ct, pred_f_test, naive_ct)
    score_g = score_counts(y_true_ct, pred_g_test, naive_ct)
    out(f"Model F test (t={TEST_T}): {score_f}  oos_r2={oos_r2(y_true_ct, pred_f_test):.4f}")
    out(f"Model G test (t={TEST_T}): {score_g}  oos_r2={oos_r2(y_true_ct, pred_g_test):.4f}")
    out()

    test_out = pd.DataFrame({
        "drug_substance": test["drug_substance"].values,
        "country_normalized": test["country_normalized"].values,
        "t": TEST_T,
        "actual_transaction_count": test["transaction_count"].values,
        "n_active_months": test["n_active_months"].values,
        "naive_forecast": np.round(naive_ct, 1),
        "model_f_gmm_xgb_forecast": np.round(pred_f_test, 1),
        "model_g_ensemble_forecast": np.round(pred_g_test, 1),
        "low_confidence": (test["n_active_months"] < 2).to_numpy(),
    })
    test_out.to_csv("Forecasted results/model_ensemble/model_ensemble_test_t8_country.csv", index=False)
    out(f"Saved -> Forecasted results/model_ensemble/model_ensemble_test_t8_country.csv ({len(test_out)} rows)")
    out()

    # ---------------- Forecast (t=T+1): every active pair has a row at t=T ----------------
    alpha_d_full = compute_entity_effects(df, np.array([phi1_d]), ["y_l1"], t_range=(1, T))
    last = df[df["t"] == TEST_T].copy()
    last["y_l2"] = last["y_l1"].values
    last["y_l1"] = last["y"].values  # roll forward: forecast month's y_l1 is T's y
    for c in COVARS_COUNTRY:
        base = c.replace("_l1", "")
        vals = last[base].values
        last[c] = np.log1p(vals) if base == "avg_deal_size" else vals

    no_history = ~last["entity"].isin(alpha_d_full.index)  # single-row-window pairs, no lag at all
    naive_ct_next = last["transaction_count"].to_numpy(dtype=float)
    f_final = naive_ct_next.copy()
    g_final = naive_ct_next.copy()
    has_hist = last[~no_history]
    if len(has_hist):
        f_final[~no_history.to_numpy()] = predict_f_country(has_hist, alpha_d_full)
        g_final[~no_history.to_numpy()] = predict_g_country(has_hist, alpha_d_full)

    low_confidence = (last["n_active_months"] < 2).to_numpy() | no_history.to_numpy()
    fallback_reason = np.where(no_history.to_numpy(), "single_month_no_lag", "")

    next_out = pd.DataFrame({
        "drug_substance": last["drug_substance"].values,
        "country_normalized": last["country_normalized"].values,
        "t": FORECAST_T,
        "forecast_month": FORECAST_MONTH,
        "last_actual_transaction_count_t8": last["transaction_count"].values,
        "n_active_months": last["n_active_months"].values,
        "naive_forecast": np.round(naive_ct_next, 1),
        "model_f_gmm_xgb_forecast": np.round(f_final, 1),
        "model_g_ensemble_forecast": np.round(g_final, 1),
        "low_confidence": low_confidence,
        "fallback_reason": fallback_reason,
    })
    next_out = next_out.sort_values(["drug_substance", "country_normalized"]).reset_index(drop=True)
    next_out.to_csv("Forecasted results/model_ensemble/model_ensemble_forecast_t9_country.csv", index=False)
    out(f"Saved -> Forecasted results/model_ensemble/model_ensemble_forecast_t9_country.csv "
        f"({len(next_out)} rows, {int(low_confidence.sum())} low_confidence)")

    with open("Code results/model_ensemble_results_country.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    out("Saved -> Code results/model_ensemble_results_country.txt")

    # ---------------- Append Model F / Model G rows to the country comparison CSV
    # (same test set/rows as score_f/score_g above) ----------------
    cmp_path = "Forecasted results/model_comparision_country.csv"
    comp_df = pd.read_csv(cmp_path)
    comp_df = comp_df[~comp_df["model"].isin(["Model F (GMM+XGB residual)", "Model G (GMM+XGB+LGBM ensemble)"])]
    comp_df.loc[len(comp_df)] = ["Model F (GMM+XGB residual)", score_f["mase"], score_f["rmse_count"],
                                  score_f["mean_signed_err"], f"n={score_f['n']}"]
    comp_df.loc[len(comp_df)] = ["Model G (GMM+XGB+LGBM ensemble)", score_g["mase"], score_g["rmse_count"],
                                  score_g["mean_signed_err"], f"n={score_g['n']}"]
    comp_df.to_csv(cmp_path, index=False)
    out(f"Updated -> {cmp_path} with Model F / Model G rows")


if __name__ == "__main__":
    import sys
    if "--country" in sys.argv:
        main_country()
    else:
        main()
        main_country()
