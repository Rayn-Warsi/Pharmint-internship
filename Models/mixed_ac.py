"""
Mixed model: per-substance best-of(Model A AR(1), Model C PPML) from model_ab.py.

Each substance is routed to whichever of A or C predicted more accurately on
the t=7 validation month; that same per-substance choice is then applied at
t=8 (test) and t=9 (true forecast). Model A and Model C themselves are fit
exactly as in model_ab.py (imported, unmodified) -- only the per-substance
routing and blended prediction are new.

Outputs written to Code results/mixed_ac_results.txt and Forecasted results/mixed_ac_*.csv.
"""

import os
import numpy as np
import pandas as pd

from model_ab import (
    load_panel, split_substances, fit_model_a, fit_model_c_ppml, fit_model_d_ab_gmm,
    compute_entity_effects, compute_entity_scale_ppml, predict_ppml,
    score_counts, oos_r2, _train_window,
)
from combine_comparisons import record_comparison

A_COLS = ["y_l1"]
C_COLS = ["y_l1"]


def predict_a_row(d, alpha, coefs_a):
    """Per-row Model A predicted count: alpha_entity + y_l1*phi1, expm1'd."""
    yhat_log = d["entity"].map(alpha).to_numpy(dtype=float) + d["y_l1"].to_numpy(dtype=float) * coefs_a[0]
    return np.clip(np.expm1(yhat_log), 0, None)


def route_by_validation(df, alpha_a, coefs_a, scale_c, coefs_c, val_t):
    """Pick A or C per substance by lower absolute error on the validation
    month (T-1). Substances with no row that month (no history) default to A."""
    d = df[df["t"] == val_t].dropna(subset=["y", "y_l1", "transaction_count"]).copy()
    d = d[d["entity"].isin(alpha_a.index) & d["entity"].isin(scale_c.index)]

    y_true = d["transaction_count"].to_numpy(dtype=float)
    pred_a = predict_a_row(d, alpha_a, coefs_a)
    pred_c = predict_ppml(d, scale_c, coefs_c, C_COLS)

    diag = pd.DataFrame({
        "entity": d["entity"].values,
        "err_a": np.abs(y_true - pred_a),
        "err_c": np.abs(y_true - pred_c),
    })
    diag["winner"] = np.where(diag["err_a"] <= diag["err_c"], "A", "C")
    winners = diag.set_index("entity")["winner"]

    all_entities = pd.Index(alpha_a.index).union(scale_c.index)
    missing = all_entities.difference(winners.index)
    if len(missing):
        winners = pd.concat([winners, pd.Series("A", index=missing)])
    return winners, diag


def mixed_predict(d, winners, alpha_a, coefs_a, scale_c, coefs_c):
    pred_a = predict_a_row(d, alpha_a, coefs_a)
    pred_c = predict_ppml(d, scale_c, coefs_c, C_COLS)
    chosen = d["entity"].map(winners).fillna("A")
    return np.where(chosen.to_numpy() == "A", pred_a, pred_c)


def main():
    lines = []
    def out(s=""):
        print(s)
        lines.append(str(s))

    os.makedirs("Code results", exist_ok=True)
    os.makedirs("Forecasted results/mixed_ac", exist_ok=True)

    df = load_panel()
    train_subs, holdout_subs = split_substances(df)
    T = int(df["t"].max())
    TRAIN_START, TRAIN_END = _train_window(df)
    VAL_T, TEST_T, FORECAST_T = T - 1, T, T + 1
    FORECAST_MONTH = "2024-09"
    out(f"Panel: {len(df)} rows, {df['drug_substance'].nunique()} substances, T={T}")
    out()

    out("=" * 70)
    out("MIXED MODEL -- per-substance best-of(A AR(1), C PPML)")
    out("=" * 70)

    res_a = fit_model_a(df, train_subs, lags=1)
    coefs_a = res_a.params[A_COLS].values
    res_c = fit_model_c_ppml(df, train_subs, xcols=C_COLS)
    coefs_c = np.array([res_c["params"][c] for c in C_COLS])
    res_d, _frame_d, _diag_d = fit_model_d_ab_gmm(df, train_subs)
    coefs_d = np.array([res_d.params["dy1"]])

    alpha_a = compute_entity_effects(df, coefs_a, A_COLS, t_range=(TRAIN_START, TRAIN_END))
    scale_c = compute_entity_scale_ppml(df, coefs_c, C_COLS, t_range=(TRAIN_START, TRAIN_END))
    alpha_d = compute_entity_effects(df, coefs_d, ["y_l1"], t_range=(TRAIN_START, TRAIN_END))

    winners, route_diag = route_by_validation(df, alpha_a, coefs_a, scale_c, coefs_c, VAL_T)
    n_a = int((winners == "A").sum())
    n_c = int((winners == "C").sum())
    out(f"Routing (t={VAL_T} validation): {n_a} substances -> A, {n_c} substances -> C")
    route_diag.to_csv("Forecasted results/mixed_ac/mixed_ac_routing.csv", index=False)
    out("Saved -> Forecasted results/mixed_ac/mixed_ac_routing.csv")
    out()

    # ---------------- Test (t=T): mixed vs pure A vs pure C ----------------
    d8 = df[df["t"] == TEST_T].dropna(subset=["y", "y_l1", "transaction_count"]).copy()
    d8 = d8[d8["entity"].isin(alpha_a.index) & d8["entity"].isin(scale_c.index)]

    y_true_ct = d8["transaction_count"].to_numpy(dtype=float)
    naive_ct = np.expm1(d8["y_l1"].to_numpy(dtype=float))
    pred_a_t8 = predict_a_row(d8, alpha_a, coefs_a)
    pred_c_t8 = predict_ppml(d8, scale_c, coefs_c, C_COLS)
    pred_d_t8 = predict_a_row(d8, alpha_d, coefs_d)
    pred_mixed_t8 = mixed_predict(d8, winners, alpha_a, coefs_a, scale_c, coefs_c)

    scores = {
        "Model A (AR1)": score_counts(y_true_ct, pred_a_t8, naive_ct),
        "Model C (PPML)": score_counts(y_true_ct, pred_c_t8, naive_ct),
        "Model D (AB-GMM)": score_counts(y_true_ct, pred_d_t8, naive_ct),
        "Mixed (A/C)": score_counts(y_true_ct, pred_mixed_t8, naive_ct),
    }
    oosr2 = {
        "Model A (AR1)": oos_r2(y_true_ct, pred_a_t8),
        "Model C (PPML)": oos_r2(y_true_ct, pred_c_t8),
        "Model D (AB-GMM)": oos_r2(y_true_ct, pred_d_t8),
        "Mixed (A/C)": oos_r2(y_true_ct, pred_mixed_t8),
    }
    for name, s in scores.items():
        out(f"{name} test (t={TEST_T}): {s}  oos_r2={oosr2[name]:.4f}")
    out()

    comp_df = pd.DataFrame([{"model": name, **s, "oos_r2": oosr2[name]} for name, s in scores.items()])
    record_comparison("Mixed (A/C)", "Mixed (A/C)", comp_df)
    out()

    test_out = pd.DataFrame({
        "drug_substance": d8["drug_substance"].values,
        "t": TEST_T,
        "actual_transaction_count": y_true_ct,
        "naive_forecast": np.round(naive_ct, 1),
        "model_a_forecast": np.round(pred_a_t8, 1),
        "model_c_forecast": np.round(pred_c_t8, 1),
        "model_d_forecast": np.round(pred_d_t8, 1),
        "mixed_forecast": np.round(pred_mixed_t8, 1),
        "chosen_model": d8["entity"].map(winners).fillna("A").values,
    })
    test_out.to_csv("Forecasted results/mixed_ac/mixed_ac_test_t8.csv", index=False)
    out(f"Saved -> Forecasted results/mixed_ac/mixed_ac_test_t8.csv ({len(test_out)} rows)")
    out()

    # ---------------- Forecast (t=T+1): refit alpha/scale on full history ----------------
    alpha_a_full = compute_entity_effects(df, coefs_a, A_COLS, t_range=(1, T))
    scale_c_full = compute_entity_scale_ppml(df, coefs_c, C_COLS, t_range=(1, T))
    alpha_d_full = compute_entity_effects(df, coefs_d, ["y_l1"], t_range=(1, T))

    last = df[df["t"] == TEST_T].dropna(subset=["y", "y_l1"]).copy()
    last = last[last["entity"].isin(alpha_a_full.index) & last["entity"].isin(scale_c_full.index)]
    last["y_l1"] = last["y"].values  # roll forward: forecast month's y_l1 is T's y

    pred_a_next = predict_a_row(last, alpha_a_full, coefs_a)
    pred_c_next = predict_ppml(last, scale_c_full, coefs_c, C_COLS)
    pred_d_next = predict_a_row(last, alpha_d_full, coefs_d)
    pred_mixed_next = mixed_predict(last, winners, alpha_a_full, coefs_a, scale_c_full, coefs_c)

    next_out = pd.DataFrame({
        "drug_substance": last["drug_substance"].values,
        "t": FORECAST_T,
        "forecast_month": FORECAST_MONTH,
        "last_actual_transaction_count_t8": last["transaction_count"].values,
        "model_a_forecast": np.round(pred_a_next, 1),
        "model_c_forecast": np.round(pred_c_next, 1),
        "model_d_forecast": np.round(pred_d_next, 1),
        "mixed_forecast": np.round(pred_mixed_next, 1),
        "chosen_model": last["entity"].map(winners).fillna("A").values,
    })
    next_out = next_out.sort_values("mixed_forecast", ascending=False)
    next_out.to_csv("Forecasted results/mixed_ac/mixed_ac_forecast_t9.csv", index=False)
    out(f"Saved -> Forecasted results/mixed_ac/mixed_ac_forecast_t9.csv ({len(next_out)} rows)")

    with open("Code results/mixed_ac_results.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    out("Saved -> Code results/mixed_ac_results.txt")


if __name__ == "__main__":
    main()
