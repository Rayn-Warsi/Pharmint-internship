"""Bar chart of monthly transaction counts, Jan-Aug 2024 actuals + Sep 2024 forecasts.

Uses the classified drug_substance panel (Code results/panel_dataset.csv), not raw
export rows -- raw rows include a large UNCLASSIFIED bucket (product text that didn't
match a known substance) that the forecast models never see, so summing raw rows
against the forecast total is apples-to-oranges (~127k/month raw vs ~41k/month
classified).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PANEL_SRC = "Code results/panel_dataset.csv"
FORECAST_SRC = "Forecasted results/forecast_t9_september2024.csv"
GBM_SRC = "Forecasted results/model_gbm/model_gbm_forecast_t9.csv"
LSTM_SRC = "Forecasted results/model_lstm/model_lstm_forecast_t9.csv"
ENSEMBLE_SRC = "Forecasted results/model_ensemble/model_ensemble_forecast_t9.csv"
OUT_PNG = "Forecasted results/monthly_transactions_bar.png"
OUT_CSV = "Forecasted results/monthly_transactions.csv"

panel = pd.read_csv(PANEL_SRC)
counts = panel.groupby("t")["transaction_count"].sum().sort_index()
month_names = ["January", "February", "March", "April", "May", "June", "July", "August"]
counts = counts.reindex(range(1, 9), fill_value=0)
counts.index = month_names
counts.to_csv(OUT_CSV, header=["transaction_count"])

fc = pd.read_csv(FORECAST_SRC)
gbm = pd.read_csv(GBM_SRC)
lstm = pd.read_csv(LSTM_SRC)
ens = pd.read_csv(ENSEMBLE_SRC)
sep_totals = {
    "AR(1)": fc["model_a_ar1_forecast"].sum(),
    "AR(2)": fc["model_a_ar2_forecast"].sum(),
    "FE-OLS": fc["model_b_forecast"].sum(),
    "PPML (C)": fc["model_c_ppml_forecast"].sum(),
    "AB-GMM (D)": fc["model_d_gmm_forecast"].sum(),
    "GBM (E)": gbm["model_e_gbm_forecast"].sum(),
    "LSTM (F)": lstm["model_f_lstm_forecast"].sum(),
    "GMM+XGB (F)": ens["model_f_gmm_xgb_forecast"].sum(),
    "GMM+XGB+LGBM (G)": ens["model_g_ensemble_forecast"].sum(),
}

fig, ax = plt.subplots(figsize=(13, 7))
x_labels = month_names + ["September (forecast)"]
x = np.arange(len(x_labels))

ax.bar(x[:8], counts.values, color="#4C72B0", width=0.8, label="Actual")

# 9 forecasts side-by-side (not nested -- outlined concentric bars are
# unreadable once totals are this close together across models)
colors = ["#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860",
          "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
names = list(sep_totals.keys())
n = len(names)
sub_w = 0.8 / n
offsets = (np.arange(n) - (n - 1) / 2) * sub_w
for name, color, off in zip(names, colors, offsets):
    ax.bar(x[8] + off, sep_totals[name], width=sub_w * 0.95, color=color, label=name)

ax.set_title("Transactions by Month (Jan-Aug 2024 actual, Sep 2024 forecast)")
ax.set_xlabel("Month")
ax.set_ylabel("Transaction Count")
ax.set_xticks(list(x))
ax.set_xticklabels(x_labels, rotation=30, ha="right")
ax.yaxis.set_major_locator(mticker.MultipleLocator(5000))
ax.set_ylim(0, max(counts.values.max(), max(sep_totals.values())) * 1.08)
ax.legend(ncol=2, fontsize=9)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150)
print(f"Saved -> {OUT_PNG}")
print(f"Saved -> {OUT_CSV}")
print(counts)
print(sep_totals)