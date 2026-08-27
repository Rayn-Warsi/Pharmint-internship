"""
Model F -- a small shared-weight LSTM over the log1p(transaction_count) sequence.

Caveat stated up front: T=8 months per substance is a very short sequence for
a recurrent model to learn from. This uses ONE shared-weight LSTM across all
792 substances (not one LSTM per substance -- that would have far too little
data per model), fed a length-2 window [y_{t-2}, y_{t-1}] -> target y_t, the
same information Model A's AR(2) spec uses, but let the network learn a
nonlinear function of it instead of a fixed linear coefficient. Expectation
per model_ab.py's own AR(2) findings: two lags did not help the linear models
either, so there's no strong reason a nonlinear function of the same two lags
should do better -- this is here to check that empirically, not because it's
expected to win.

Same t=3..6 train / t=7 validate / t=8 test / t=9 forecast structure as
model_ab.py, reusing its data loading, splitting, and scoring. model_ab.py
itself is not modified.

Outputs written to Code results/model_lstm_results.txt and Forecasted results/model_lstm_*.csv.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from model_ab import (
    load_panel, split_substances, fit_model_a, fit_model_d_ab_gmm,
    compute_entity_effects, score_counts, oos_r2, SEED, _train_window,
)
from combine_comparisons import record_comparison

HIDDEN = 16
EPOCHS = 300
LR = 1e-2


class LSTMRegressor(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)


def build_sequences(df, subs, t_range):
    d = df[df["entity"].isin(subs) & df["t"].between(*t_range)].dropna(subset=["y", "y_l1", "y_l2"]).copy()
    seq = np.ascontiguousarray(d[["y_l2", "y_l1"]].to_numpy(dtype=np.float32)[:, :, None])  # (N, 2, 1)
    target = np.ascontiguousarray(d["y"].to_numpy(dtype=np.float32))
    return d, torch.from_numpy(seq), torch.from_numpy(target)


def predict_lstm(model, seq_tensor):
    model.eval()
    with torch.no_grad():
        return model(seq_tensor).numpy()


def main():
    lines = []
    def out(s=""):
        print(s)
        lines.append(str(s))

    os.makedirs("Code results", exist_ok=True)
    os.makedirs("Forecasted results/model_lstm", exist_ok=True)
    torch.manual_seed(SEED)

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
    out("MODEL F -- shared-weight LSTM, input [y_t-2, y_t-1] -> y_t")
    out("=" * 70)

    d_train, X_train, y_train = build_sequences(df, train_subs, (TRAIN_START, TRAIN_END))
    d_val, X_val, y_val = build_sequences(df, all_subs, (VAL_T, VAL_T))
    out(f"Train sequences: {len(d_train)}, validation (t={VAL_T}) sequences: {len(d_val)}")

    model = LSTMRegressor()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            out(f"  epoch {epoch:3d}  train_mse={loss.item():.4f}  val_mse(t={VAL_T})={val_loss:.4f}")

    model.load_state_dict(best_state)
    out(f"Best validation (t={VAL_T}) MSE (log scale): {best_val:.4f}")
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
    d8, X_test, _ = build_sequences(df, all_subs, (TEST_T, TEST_T))
    d8 = d8[d8["entity"].isin(alpha_a.index) & d8["entity"].isin(alpha_d.index)]
    X_test = torch.from_numpy(np.ascontiguousarray(d8[["y_l2", "y_l1"]].to_numpy(dtype=np.float32)[:, :, None]))

    y_true_ct = np.expm1(d8["y"].to_numpy(dtype=float))
    naive_ct = np.expm1(d8["y_l1"].to_numpy(dtype=float))

    pred_lstm_t8 = np.clip(np.expm1(predict_lstm(model, X_test)), 0, None)
    pred_a_t8 = predict_loglinear(d8, alpha_a, coefs_a)
    pred_d_t8 = predict_loglinear(d8, alpha_d, coefs_d)

    scores = {
        "Model F (LSTM)": score_counts(y_true_ct, pred_lstm_t8, naive_ct),
        "Model A (AR1)": score_counts(y_true_ct, pred_a_t8, naive_ct),
        "Model D (AB-GMM)": score_counts(y_true_ct, pred_d_t8, naive_ct),
    }
    oosr2 = {
        "Model F (LSTM)": oos_r2(y_true_ct, pred_lstm_t8),
        "Model A (AR1)": oos_r2(y_true_ct, pred_a_t8),
        "Model D (AB-GMM)": oos_r2(y_true_ct, pred_d_t8),
    }
    for name, s in scores.items():
        out(f"{name} test (t={TEST_T}): {s}  oos_r2={oosr2[name]:.4f}")
    out()

    comp_df = pd.DataFrame([{"model": name, **s, "oos_r2": oosr2[name]} for name, s in scores.items()])
    record_comparison("Model F (LSTM)", "Model F (LSTM)", comp_df)
    out()

    test_out = pd.DataFrame({
        "drug_substance": d8["drug_substance"].values,
        "t": TEST_T,
        "actual_transaction_count": y_true_ct,
        "naive_forecast": np.round(naive_ct, 1),
        "model_f_lstm_forecast": np.round(pred_lstm_t8, 1),
        "model_a_forecast": np.round(pred_a_t8, 1),
        "model_d_forecast": np.round(pred_d_t8, 1),
    })
    test_out.to_csv("Forecasted results/model_lstm/model_lstm_test_t8.csv", index=False)
    out(f"Saved -> Forecasted results/model_lstm/model_lstm_test_t8.csv ({len(test_out)} rows)")
    out()

    # ---------------- Forecast (t=T+1): sequence [y_{T-1}, y_T] -> y_{T+1} ----------------
    last = df[df["t"] == TEST_T].dropna(subset=["y", "y_l1"]).copy()
    seq_next = torch.from_numpy(np.ascontiguousarray(last[["y_l1", "y"]].to_numpy(dtype=np.float32)[:, :, None]))
    pred_lstm_next = np.clip(np.expm1(predict_lstm(model, seq_next)), 0, None)

    next_out = pd.DataFrame({
        "drug_substance": last["drug_substance"].values,
        "t": FORECAST_T,
        "forecast_month": FORECAST_MONTH,
        "last_actual_transaction_count_t8": last["transaction_count"].values,
        "model_f_lstm_forecast": np.round(pred_lstm_next, 1),
    })
    next_out = next_out.sort_values("model_f_lstm_forecast", ascending=False)
    next_out.to_csv("Forecasted results/model_lstm/model_lstm_forecast_t9.csv", index=False)
    out(f"Saved -> Forecasted results/model_lstm/model_lstm_forecast_t9.csv ({len(next_out)} rows)")

    with open("Code results/model_lstm_results.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    out("Saved -> Code results/model_lstm_results.txt")


if __name__ == "__main__":
    main()
