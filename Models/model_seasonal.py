"""
Model H -- Fourier-seasonal Arellano-Bond GMM. Same AR(1) difference-GMM spec
as model_ab.py's Model D, plus two deterministic calendar terms (sin/cos of
month-of-year) so the panel can express an annual cycle without burning a
parameter per calendar month -- at T=20, Sep-Dec each appear only once, too
thin to identify 12 separate month dummies (see forecast-models.html).
Fourier terms are exogenous and known for any future t, so they need no
lagging or instrumenting, just added straight into the differenced equation.

Model I -- SARIMA, fit two ways per the two-model spec's own "aggregate and
per substance" pattern:
  - aggregate: one series, total transaction_count summed across all
    substances per month. Enough data (T=20) for a real seasonal fit.
  - per substance: one SARIMA per substance with >= MIN_MONTHS_SARIMA months
    of history. Substances below that fall back to naive and are flagged,
    not silently folded into the score.

Outputs written to Code results/model_seasonal_results.txt.
"""

import os
import warnings

import numpy as np
import pandas as pd
from linearmodels.iv import IVGMM
from statsmodels.tsa.statespace.sarimax import SARIMAX

from model_ab import (
    load_panel, split_substances, _train_window, compute_entity_effects,
    predict_and_score, score_counts, oos_r2,
)

FOURIER_PERIOD = 12
MIN_MONTHS_SARIMA = 15  # need enough history to identify a seasonal lag at m=12


def add_fourier(df):
    cal_month = ((df["t"] - 1) % FOURIER_PERIOD) + 1
    df = df.copy()
    df["fourier_sin"] = np.sin(2 * np.pi * cal_month / FOURIER_PERIOD)
    df["fourier_cos"] = np.cos(2 * np.pi * cal_month / FOURIER_PERIOD)
    return df


# ---------------------------------------------------------------------------
# Model H -- Fourier-seasonal AB-GMM
# ---------------------------------------------------------------------------

def fit_model_h_fourier_gmm(df, train_subs):
    need = ["y", "y_l1", "y_l2", "y_l3", "fourier_sin", "fourier_cos"]
    d = df.dropna(subset=need).copy()
    t_start, t_end = _train_window(df)
    d = d[d["entity"].isin(train_subs) & d["t"].between(t_start, t_end)]
    d = d.sort_values(["entity", "t"])

    g = d.groupby("entity")
    dy = g["y"].diff()
    dy1 = g["y_l1"].diff()
    dsin = g["fourier_sin"].diff()
    dcos = g["fourier_cos"].diff()

    frame = pd.concat([
        d[["entity", "t", "y_l2", "y_l3"]].reset_index(drop=True),
        dy.reset_index(drop=True).rename("dy"),
        dy1.reset_index(drop=True).rename("dy1"),
        dsin.reset_index(drop=True).rename("dsin"),
        dcos.reset_index(drop=True).rename("dcos"),
    ], axis=1).dropna()

    endog = frame[["dy1"]]
    exog = frame[["dsin", "dcos"]].copy()
    exog.insert(0, "const", 1.0)
    instruments = frame[["y_l2", "y_l3"]]

    mod = IVGMM(frame["dy"], exog, endog, instruments)
    res = mod.fit(cov_type="clustered", clusters=frame["entity"])
    return res, frame


def score_model_h(df, res, t_target, subs, t_range):
    coefs = np.array([res.params["dy1"], res.params["dsin"], res.params["dcos"]])
    xcols = ["y_l1", "fourier_sin", "fourier_cos"]
    alpha = compute_entity_effects(df, coefs, xcols, t_range=t_range)
    score = predict_and_score(df, alpha, coefs, ["y_l1"], ["fourier_sin", "fourier_cos"], t_target, subs)
    return score, alpha, coefs


def score_model_h_counts(df, res, t_target, subs, t_range):
    """Same fit as score_model_h, but scored with score_counts() (superset of
    predict_and_score()'s dict -- adds mean_signed_err/total_pred/total_actual)
    so it can be folded into model_comparision.csv the same way Models F/G are."""
    coefs = np.array([res.params["dy1"], res.params["dsin"], res.params["dcos"]])
    xcols = ["y_l1", "fourier_sin", "fourier_cos"]
    alpha = compute_entity_effects(df, coefs, xcols, t_range=t_range)
    d = df[(df["t"] == t_target) & df["entity"].isin(subs)].dropna(subset=["y"] + xcols).copy()
    d = d[d["entity"].isin(alpha.index)]
    if len(d) == 0:
        return None, None, None
    yhat = d["entity"].map(alpha).values + d[xcols].values @ coefs
    y_true_ct = np.expm1(d["y"].values)
    y_pred_ct = np.clip(np.expm1(yhat), 0, None)
    naive_ct = np.expm1(d["y_l1"].values)
    return score_counts(y_true_ct, y_pred_ct, naive_ct), y_true_ct, y_pred_ct


# ---------------------------------------------------------------------------
# Model I -- SARIMA, aggregate market total
# ---------------------------------------------------------------------------

def fit_sarima_aggregate(df_train, order=(1, 0, 0), seasonal_order=(1, 0, 0, FOURIER_PERIOD)):
    agg = df_train.groupby("t")["transaction_count"].sum().sort_index()
    y = np.log1p(agg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = SARIMAX(y, order=order, seasonal_order=seasonal_order,
                       enforce_stationarity=False, enforce_invertibility=False)
        res = mod.fit(disp=False)
    return res


def score_sarima_aggregate(res, full_agg, val_t, test_t):
    steps = test_t - val_t
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_log = res.forecast(steps=steps).iloc[-1]
    pred_ct = max(float(np.expm1(pred_log)), 0.0)
    actual_ct = float(full_agg.loc[test_t])
    naive_ct = float(full_agg.loc[val_t])
    mae_ct = abs(actual_ct - pred_ct)
    mae_naive = abs(actual_ct - naive_ct)
    return dict(n=1, rmse_count=mae_ct, mae_count=mae_ct, mae_naive=mae_naive,
                mase=mae_ct / mae_naive if mae_naive else np.nan,
                pred=pred_ct, actual=actual_ct)


# ---------------------------------------------------------------------------
# Model I -- SARIMA, per substance (naive fallback below MIN_MONTHS_SARIMA)
# ---------------------------------------------------------------------------

def fit_sarima_per_substance(df, val_t, test_t, order=(1, 0, 0),
                              seasonal_order=(1, 0, 0, FOURIER_PERIOD)):
    results = []
    for sub, g in df.groupby("drug_substance"):
        g = g.sort_values("t")
        hist = g[g["t"] <= val_t]
        if len(hist) < MIN_MONTHS_SARIMA or test_t not in g["t"].values:
            continue
        y_hist = np.log1p(hist.set_index("t")["transaction_count"])
        actual = float(g.loc[g["t"] == test_t, "transaction_count"].iloc[0])
        naive = float(g.loc[g["t"] == val_t, "transaction_count"].iloc[0])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = SARIMAX(y_hist, order=order, seasonal_order=seasonal_order,
                               enforce_stationarity=False, enforce_invertibility=False)
                res = mod.fit(disp=False)
                pred_log = res.forecast(steps=test_t - val_t).iloc[-1]
            pred = max(float(np.expm1(pred_log)), 0.0)
            fallback = False
        except Exception:
            pred = naive
            fallback = True
        results.append(dict(drug_substance=sub, actual=actual, pred=pred, naive=naive, fallback=fallback))
    return pd.DataFrame(results)


def score_sarima_per_substance(res_df):
    fit_ok = res_df[~res_df["fallback"]]
    if fit_ok.empty:
        return None, None, None
    y_true_ct = fit_ok["actual"].to_numpy(dtype=float)
    y_pred_ct = fit_ok["pred"].to_numpy(dtype=float)
    naive_ct = fit_ok["naive"].to_numpy(dtype=float)
    return score_counts(y_true_ct, y_pred_ct, naive_ct), y_true_ct, y_pred_ct


# ---------------------------------------------------------------------------
# Fold Model H / Model I's key metrics into Forecasted results/model_comparision.csv,
# same direct-column-append pattern model_ensemble.py uses for Models F/G (the
# dashboard's ingest.py reads this CSV, not the satellite-model xlsx path).
# ---------------------------------------------------------------------------

def append_to_comparison_csv(test_t, val_t, columns):
    """columns: {col_name: score_counts()-style dict} for one or more models.
    Rows that don't apply (phi1/Hansen J/etc, unless passed explicitly via
    extra_metrics) are left blank rather than faked, matching F/G's convention."""
    path = "Forecasted results/model_comparision.csv"
    main_cmp = pd.read_csv(path)
    for col_name, (score, extra_metrics) in columns.items():
        if score is None:
            continue
        metric_map = [
            (f"test MASE, t={test_t}, all substances", score["mase"]),
            (f"test RMSE (log scale), t={test_t}, all substances", score["rmse_log"]),
            (f"test RMSE (count scale), t={test_t}, all substances", score["rmse_count"]),
            (f"test mean SIGNED error, t={test_t}", score["mean_signed_err"]),
            (f"test total forecast, t={test_t}", score["total_pred"]),
            (f"test total error %, t={test_t}",
             100.0 * (score["total_pred"] - score["total_actual"]) / score["total_actual"]),
        ]
        metric_map += list(extra_metrics.items())
        col = pd.Series(np.nan, index=main_cmp.index, dtype=object)
        for metric, val in metric_map:
            col[main_cmp["metric"] == metric] = val
        main_cmp[col_name] = col
    main_cmp.to_csv(path, index=False)
    print(f"Updated -> {path} with {', '.join(columns.keys())}")


def main():
    lines = []

    def out(s=""):
        print(s)
        lines.append(str(s))

    df = load_panel()
    df = add_fourier(df)
    train_subs, holdout_subs = split_substances(df)
    T = int(df["t"].max())
    TRAIN_START, TRAIN_END = _train_window(df)
    VAL_T, TEST_T = T - 1, T
    out(f"Panel: {len(df)} rows, {df['drug_substance'].nunique()} substances, T={T}")
    out()

    # ---------------- Model H ----------------
    out("=" * 70)
    out("MODEL H -- Fourier-seasonal Arellano-Bond GMM")
    out("=" * 70)
    res_h, frame_h = fit_model_h_fourier_gmm(df, train_subs)
    out(str(res_h))
    out(f"N obs: {len(frame_h)}, entities: {frame_h['entity'].nunique()}")
    out()
    for label, subs in [("train-subs", train_subs), ("holdout-subs", holdout_subs)]:
        score, _, _ = score_model_h(df, res_h, VAL_T, subs, (TRAIN_START, TRAIN_END))
        out(f"Model H validation (t={VAL_T}, {label}): {score}")
    score_h_test, y_true_h, y_pred_h = score_model_h_counts(
        df, res_h, TEST_T, set(df["entity"]), (TRAIN_START, TRAIN_END))
    out(f"Model H test (t={TEST_T}): {score_h_test}")
    out()

    # ---------------- Model I: SARIMA aggregate ----------------
    out("=" * 70)
    out("MODEL I -- SARIMA, aggregate market total")
    out("=" * 70)
    res_agg = fit_sarima_aggregate(df[df["t"] <= VAL_T])
    out(str(res_agg.summary()))
    full_agg = df.groupby("t")["transaction_count"].sum()
    score_agg = score_sarima_aggregate(res_agg, full_agg, VAL_T, TEST_T)
    out(f"Model I (aggregate) test (t={TEST_T}): {score_agg}")
    out()

    # ---------------- Model I: SARIMA per substance ----------------
    out("=" * 70)
    out(f"MODEL I -- SARIMA, per substance (>= {MIN_MONTHS_SARIMA} months history required)")
    out("=" * 70)
    res_df = fit_sarima_per_substance(df, VAL_T, TEST_T)
    n_fit = int((~res_df["fallback"]).sum()) if len(res_df) else 0
    out(f"Substances scored: {len(res_df)} (real SARIMA fit: {n_fit}, naive fallback: {len(res_df) - n_fit})")
    score_sub, y_true_i, y_pred_i = score_sarima_per_substance(res_df)
    out(f"Model I (per-substance, real fits only) test (t={TEST_T}): {score_sub}")
    out()

    # ---------------- Fold into Forecasted results/model_comparision.csv ----------------
    # Model I's dashboard column is the per-substance fit only -- the aggregate
    # variant above is a single market-total number, not comparable to the other
    # models' per-substance-pooled metrics, so it's reported here but not in the CSV.
    h_extra = {
        "phi1 (coefficient)": res_h.params["dy1"],
        "phi1 p-value": res_h.pvalues["dy1"],
    }
    append_to_comparison_csv(TEST_T, VAL_T, {
        "Model H (Fourier-seasonal GMM)": (score_h_test, h_extra),
        "Model I (SARIMA)": (score_sub, {}),
    })
    if score_h_test:
        out(f"Model H oos_r2: {oos_r2(y_true_h, y_pred_h):.4f}")
    if score_sub:
        out(f"Model I (per-substance) oos_r2: {oos_r2(y_true_i, y_pred_i):.4f}")
    out()

    os.makedirs("Code results", exist_ok=True)
    with open("Code results/model_seasonal_results.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    out("Saved -> Code results/model_seasonal_results.txt")


if __name__ == "__main__":
    main()
