"""
Shared helper for folding a satellite model's test-set scores (Mixed (A/C),
Mixed (D/C), Model E (GBM), Model F (LSTM)) directly into
Forecasted results/model_comparision.xlsx -- mixed_ac.py/mixed_dc.py/
model_gbm.py/model_lstm.py each call record_comparison() in-memory instead of
writing their own *_comparison.csv, so there's no intermediate file to keep
in sync.

Rows that don't apply to these models (phi1/phi2, Hansen J, entities used,
etc. -- those are FE-OLS/GMM diagnostics) are left blank rather than faked.

Run this file directly to rebuild model_comparision.xlsx from
model_comparision.csv alone (base AR(1)/AR(2)/Model B/Model C/Model D columns
only, no satellite models) -- useful right after a fresh model_ab.py run.
"""

import pandas as pd

OUT_PATH = "Forecasted results/model_comparision.xlsx"

EXTRA_ROWS = [
    ("test n (rows scored)", "n"),
    ("test MAE naive baseline (count scale)", "mae_naive"),
    ("test MAE (count scale), all substances", "mae_count"),
]

NOTE_TEXT = (
    "AR(1)/AR(2)/Model B/Model C/Model D score on each model's own t={test_t} rows "
    "(dropna requirements differ per model); Mixed(A/C), Mixed(D/C), Model E (GBM) "
    "and Model F (LSTM) score on a different shared t={test_t} row set. total_actual "
    "differs between the two groups as a result -- MASE and "
    "oos_r2 are still comparable across all 9 models, but total_pred / "
    "total_error_pct are only exactly apples-to-apples within each group, not across both."
)


def _metric_map(test_t):
    """metric label (as it appears in model_comparision's "metric" column) ->
    key in score_counts()-style dicts. The labels embed the test month's
    t-index, which model_ab.py derives from the panel (T) rather than
    hardcoding to 8 -- built here from the same value so the lookup matches."""
    return {
        f"test MASE, t={test_t}, all substances": "mase",
        f"test RMSE (log scale), t={test_t}, all substances": "rmse_log",
        f"test RMSE (count scale), t={test_t}, all substances": "rmse_count",
        f"test mean SIGNED error, t={test_t}": "mean_signed_err",
        f"test total forecast, t={test_t}": "total_pred",
        f"test total error %, t={test_t}": "total_error_pct",
        f"out-of-sample R2, t={test_t}, count scale": "oos_r2",
    }


def _test_t():
    return int(pd.read_csv("Code results/panel_dataset.csv")["t"].max())


def _load_base(test_t):
    """Load model_comparision.xlsx if it exists (stripping its NOTE row so it
    can be rebuilt), else fall back to model_comparision.csv, else an empty
    frame with just a metric column and the extra rows pre-seeded."""
    try:
        df = pd.read_excel(OUT_PATH, sheet_name="model_comparision")
        return df[df["metric"] != "NOTE"].reset_index(drop=True)
    except FileNotFoundError:
        pass
    try:
        return pd.read_csv("Forecasted results/model_comparision.csv")
    except FileNotFoundError:
        rows = [{"metric": label} for label, _ in EXTRA_ROWS]
        return pd.DataFrame(rows, columns=["metric"])


def record_comparison(model_col, col_name, comp_df):
    """Update model_comparision.xlsx in place with one satellite model's
    scores, without writing any intermediate CSV.

    model_col -- the row label in comp_df (e.g. "Mixed (A/C)")
    col_name  -- the column header to write/overwrite in model_comparision
    comp_df   -- DataFrame with a "model" column and score_counts()-style
                 fields (mase, rmse_log, rmse_count, mean_signed_err,
                 total_pred, total_actual, oos_r2, n, mae_naive, mae_count)
    """
    test_t = _test_t()
    metric_map = _metric_map(test_t)
    df = _load_base(test_t)

    if "note" not in df.columns:
        df["note"] = ""

    for label, _ in EXTRA_ROWS:
        if label not in set(df["metric"]):
            blank = {c: "" for c in df.columns}
            blank["metric"] = label
            df = pd.concat([df, pd.DataFrame([blank])], ignore_index=True)

    r = comp_df.set_index("model").loc[model_col]
    values = dict(r)
    values["total_error_pct"] = 100.0 * (r["total_pred"] - r["total_actual"]) / r["total_actual"]

    if col_name not in df.columns:
        df[col_name] = ""

    def cell(metric):
        if metric in metric_map:
            return values.get(metric_map[metric], "")
        for label, key in EXTRA_ROWS:
            if metric == label:
                return values.get(key, "")
        return df.loc[df["metric"] == metric, col_name].iloc[0]

    df[col_name] = df["metric"].map(cell)

    note_row = {c: "" for c in df.columns}
    note_row["metric"] = "NOTE"
    note_row["note"] = NOTE_TEXT.format(test_t=test_t)
    df = pd.concat([df, pd.DataFrame([note_row])], ignore_index=True)

    cols = [c for c in df.columns if c != "note"] + ["note"]
    df = df[cols]

    df.to_excel(OUT_PATH, sheet_name="model_comparision", index=False)
    print(f"Saved -> {OUT_PATH} ({col_name} updated)")


def main():
    """Rebuild model_comparision.xlsx from model_comparision.csv alone (no
    satellite models) -- run mixed_ac.py/mixed_dc.py/model_gbm.py/
    model_lstm.py afterward to fold their columns back in."""
    test_t = _test_t()
    df = pd.read_csv("Forecasted results/model_comparision.csv")
    df["note"] = ""
    note_row = {c: "" for c in df.columns}
    note_row["metric"] = "NOTE"
    note_row["note"] = NOTE_TEXT.format(test_t=test_t)
    df = pd.concat([df, pd.DataFrame([note_row])], ignore_index=True)
    df.to_excel(OUT_PATH, sheet_name="model_comparision", index=False)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
