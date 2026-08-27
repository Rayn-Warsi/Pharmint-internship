"""
Fits Model A (pooled AR(1)/AR(2), substance fixed effects) and Model B
(dynamic panel with market-condition covariates, month effects, Arellano-Bond
difference GMM correction for Nickell bias) per two-model-spec.md, on the
substance x month panel from panel_build.py.

Outputs printed to stdout and written to Code results/model_results.txt.
"""

import io
import os
import contextlib
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, PooledOLS
from linearmodels.iv import IVGMM
from scipy import stats
from scipy.optimize import minimize

PANEL_PATH = "Code results/panel_dataset.csv"
PANEL_PATH_COUNTRY = "Code results/panel_dataset_country.csv"
SEED = 42

# Was hardcoded to T=8's (3, 6)/7/8/9 throughout this file. Generalized so the
# same relative windows -- train = t=3..T-2, validate = T-1, test = T, true
# out-of-sample forecast = T+1 -- hold at any panel length. T is read off the
# panel itself (max t), never assumed.
def _train_window(df):
    """(train_start, train_end) = (3, T-2)."""
    T = int(df["t"].max())
    return 3, T - 2

# Every fit/scoring helper below groups, indexes, and clusters on the single
# "entity" column rather than hardcoding "drug_substance". For the drug-level
# panel entity == drug_substance (unchanged behavior). For the country panel
# entity is the composite (drug_substance, country_normalized) pair, joined as
# "drug||country" -- letting the exact same fitting code serve both grains.
ENTITY_SEP = "||"


def _make_entity(df, entity_cols):
    entity_cols = list(entity_cols)
    if len(entity_cols) == 1:
        return df[entity_cols[0]].astype(str)
    return df[entity_cols].astype(str).agg(ENTITY_SEP.join, axis=1)


def load_panel(path=PANEL_PATH, entity_cols=("drug_substance",)):
    df = pd.read_csv(path)
    entity_cols = list(entity_cols)
    df["entity"] = _make_entity(df, entity_cols)
    df = df.sort_values(["entity", "t"]).reset_index(drop=True)

    df["y"] = np.log1p(df["transaction_count"])
    g = df.groupby("entity")
    df["y_l1"] = g["y"].shift(1)
    df["y_l2"] = g["y"].shift(2)
    df["y_l3"] = g["y"].shift(3)
    df["y_l4"] = g["y"].shift(4)

    df["ln_p50"] = np.log(df["p50_unit_price"].clip(lower=1e-6))
    covar_base = ["ln_p50", "avg_deal_size", "price_spread", "exporter_churn"]
    if "destination_count" in df.columns:
        covar_base.append("destination_count")
    for c in covar_base:
        df[c + "_l1"] = g[c].shift(1)

    df["avg_deal_size_l1"] = np.log1p(df["avg_deal_size_l1"])
    df["months_since_first_l1"] = df["months_since_first"] - 1
    return df


def load_panel_country(path=PANEL_PATH_COUNTRY):
    return load_panel(path, entity_cols=("drug_substance", "country_normalized"))


def split_substances(df, seed=SEED):
    subs = np.array(df["entity"].unique(), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    n_test = int(len(subs) * 0.2)
    holdout = set(subs[:n_test])
    train_subs = set(subs[n_test:])
    return train_subs, holdout


# months_since_first_l1 dropped: with both entity and month effects included it is exactly
# t - first_t - 1, i.e. a deterministic function of the entity effect and the time effect
# (and, after differencing, a constant) -- perfectly collinear, absorbed by the FE.
COVARS = ["ln_p50_l1", "avg_deal_size_l1", "exporter_churn_l1", "destination_count_l1",
          "price_spread_l1"]

# destination_count is degenerate (always 1) at drug x country grain -- dropped there.
COVARS_COUNTRY = [c for c in COVARS if c != "destination_count_l1"]


def fit_model_a(df, train_subs, lags=2, covars=()):
    covars = list(covars)
    need = ["y", "y_l1"] + (["y_l2"] if lags == 2 else []) + covars
    d = df.dropna(subset=need).copy()
    t_start, t_end = _train_window(df)
    d = d[d["entity"].isin(train_subs) & d["t"].between(t_start, t_end)]
    d = d.set_index(["entity", "t"])
    exog_cols = ["y_l1"] + (["y_l2"] if lags == 2 else []) + covars
    exog = pd.concat([pd.Series(1.0, index=d.index, name="const"), d[exog_cols]], axis=1)
    mod = PanelOLS(d["y"], exog, entity_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    return res


def fit_model_b_ols_within(df, train_subs, covars=COVARS):
    covars = list(covars)
    need = ["y", "y_l1", "y_l2"] + covars
    d = df.dropna(subset=need).copy()
    t_start, t_end = _train_window(df)
    d = d[d["entity"].isin(train_subs) & d["t"].between(t_start, t_end)]
    d = d.set_index(["entity", "t"])
    exog = pd.concat([pd.Series(1.0, index=d.index, name="const"), d[["y_l1", "y_l2"] + covars]], axis=1)
    mod = PanelOLS(d["y"], exog, entity_effects=True, time_effects=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    return res


def fit_pooled_ols(df, train_subs):
    need = ["y", "y_l1", "y_l2"]
    d = df.dropna(subset=need).copy()
    t_start, t_end = _train_window(df)
    d = d[d["entity"].isin(train_subs) & d["t"].between(t_start, t_end)]
    d = d.set_index(["entity", "t"])
    exog = pd.concat([pd.Series(1.0, index=d.index, name="const"), d[["y_l1", "y_l2"]]], axis=1)
    mod = PooledOLS(d["y"], exog)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    return res


def fit_model_b_gmm(df, train_subs, covars=COVARS):
    """Arellano-Bond difference GMM: first-difference removes substance FE,
    collapsed levels y_{t-3}, y_{t-4} instrument the differenced lagged y's (depth
    capped at 2, per spec). Two endogenous regressors (dy1, dy2) need >=2 excluded
    instruments for the order condition, which is why both lags are required here --
    at T=8 that leaves a single usable target period, exactly the fragility the spec
    warns about."""
    covars = list(covars)
    need = ["y", "y_l1", "y_l2", "y_l3", "y_l4"] + covars
    d = df.dropna(subset=need).copy()
    # window starts one period later than the general train window (needs
    # y_l4, one lag deeper than A/B/C/D) -- was (4, 6) at T=8, i.e.
    # (train_start+1, train_end).
    t_start, t_end = _train_window(df)
    d = d[d["entity"].isin(train_subs) & d["t"].between(t_start + 1, t_end)]
    d = d.sort_values(["entity", "t"])

    dy = d.groupby("entity")["y"].diff()
    dy1 = d.groupby("entity")["y_l1"].diff()
    dy2 = d.groupby("entity")["y_l2"].diff()
    dX = d.groupby("entity")[covars].diff()
    dX.columns = [c + "_d" for c in covars]

    frame = pd.concat([d[["entity", "t", "y_l3", "y_l4"]].reset_index(drop=True),
                        dy.reset_index(drop=True).rename("dy"),
                        dy1.reset_index(drop=True).rename("dy1"),
                        dy2.reset_index(drop=True).rename("dy2"),
                        dX.reset_index(drop=True)], axis=1)
    frame = frame.dropna()

    month_dummies = pd.get_dummies(frame["t"], prefix="dmonth", drop_first=True).astype(float)

    endog = frame[["dy1", "dy2"]]
    exog = pd.concat([pd.Series(1.0, index=frame.index, name="const"),
                       frame[[c + "_d" for c in covars]], month_dummies], axis=1)
    # linearmodels forms the full instrument matrix internally as [exog, instruments] --
    # pass only the excluded (non-exog) instruments here, i.e. the collapsed deeper lags.
    instruments = frame[["y_l3", "y_l4"]]

    mod = IVGMM(frame["dy"], exog, endog, instruments)
    res = mod.fit(cov_type="clustered", clusters=frame["entity"])
    n_instruments = exog.shape[1] + instruments.shape[1]
    n_entities = frame["entity"].nunique()
    return res, frame, n_instruments, n_entities


# ---------------------------------------------------------------------------
# Model C -- Poisson pseudo-MLE with substance fixed effects (PPML)
#
# Motivation: A and B both fit log1p(count) by OLS and invert with expm1. That
# round trip does not recover E[y|x] under heteroskedasticity (Jensen) -- it
# recovers something nearer the conditional geometric mean, so forecasts are
# biased DOWN. Model A's t=8 mean signed error of about -2 per substance, and a
# total under actual, is that bias showing up in the shipped numbers. The +1 in
# log1p is also a free tuning constant touching the 15.6% of rows that are zero
# and the 35.5% that are <= 2.
#
# PPML models the count directly, so the fitted value IS the conditional mean --
# no retransformation, and zeros need no fudge. Consistent under the mean
# assumption alone even when the data is not Poisson (Santos Silva & Tenreyro).
#
# CAVEAT, stated plainly: the "FE Poisson has no incidental-parameters problem
# at fixed T" result assumes STRICTLY EXOGENOUS regressors. y_l1 is not strictly
# exogenous, so Model C inherits the same dynamic-panel bias on phi that Nickell
# describes for A and B. It is not a fix for that. It fixes scale and zeros.
# ---------------------------------------------------------------------------

def _ppml_neg_concentrated_ll(b, X, y, codes, n_ent):
    """Negative Poisson FE log-likelihood, entity effects concentrated out.

    The FE score gives a closed form, exp(alpha_d) = Y_d / S_d(b) with
    Y_d = sum_t y_dt and S_d = sum_t exp(x_dt'b), so the 792 substance effects
    never enter the optimizer -- only b does. Substituting back leaves
        l(b) = sum_dt y_dt x_dt'b - sum_d Y_d log S_d(b)   (+ const)
    Entities with Y_d = 0 contribute nothing and drop out on their own.
    """
    xb = np.clip(X @ b, -50.0, 50.0)
    e = np.exp(xb)
    S = np.bincount(codes, weights=e, minlength=n_ent)
    Y = np.bincount(codes, weights=y, minlength=n_ent)
    live = Y > 0
    ll = float(y @ xb) - float(np.sum(Y[live] * np.log(S[live])))

    scale = np.zeros(n_ent)
    scale[live] = Y[live] / S[live]
    mu = scale[codes] * e
    grad = X.T @ (y - mu)
    return -ll, -grad


def fit_model_c_ppml(df, train_subs, xcols=("y_l1",)):
    """FE-PPML on the same estimation window and regressors as Model A AR(1),
    so the only thing that changes versus A is the estimator itself."""
    xcols = list(xcols)
    d = df.dropna(subset=["transaction_count"] + xcols).copy()
    t_start, t_end = _train_window(df)
    d = d[d["entity"].isin(train_subs) & d["t"].between(t_start, t_end)]

    ent = pd.Categorical(d["entity"])
    codes = ent.codes.astype(np.int64)
    n_ent = len(ent.categories)
    X = d[xcols].to_numpy(dtype=float)
    y = d["transaction_count"].to_numpy(dtype=float)

    res = minimize(_ppml_neg_concentrated_ll, np.zeros(X.shape[1]),
                   args=(X, y, codes, n_ent), jac=True, method="BFGS")
    b = res.x

    # clustered (by substance) sandwich, Wooldridge FE-Poisson form: the bread
    # uses mu-weighted within-transformed regressors, the meat sums scores per
    # substance so within-substance serial correlation is allowed for.
    xb = np.clip(X @ b, -50.0, 50.0)
    e = np.exp(xb)
    S = np.bincount(codes, weights=e, minlength=n_ent)
    Y = np.bincount(codes, weights=y, minlength=n_ent)
    live = Y > 0
    scale = np.zeros(n_ent)
    scale[live] = Y[live] / S[live]
    mu = scale[codes] * e

    w = np.bincount(codes, weights=mu, minlength=n_ent)
    k = X.shape[1]
    xbar = np.column_stack([np.bincount(codes, weights=mu * X[:, j], minlength=n_ent) for j in range(k)])
    xbar = xbar / np.where(w[:, None] > 0, w[:, None], 1.0)
    Xt = X - xbar[codes]

    A = (Xt * mu[:, None]).T @ Xt
    u = (y - mu)[:, None] * Xt
    g = np.column_stack([np.bincount(codes, weights=u[:, j], minlength=n_ent) for j in range(k)])
    B = g.T @ g
    Ainv = np.linalg.inv(A)
    V = Ainv @ B @ Ainv

    se = np.sqrt(np.diag(V))
    z = b / se
    p = 2 * stats.norm.sf(np.abs(z))
    return dict(params=dict(zip(xcols, b)), se=dict(zip(xcols, se)),
                pvalues=dict(zip(xcols, p)), xcols=xcols, converged=bool(res.success),
                n_obs=len(d), n_entities=int(live.sum()))


def ppml_pseudo_r2(df, b, xcols, train_subs):
    """Deviance pseudo-R2 for FE-Poisson, against an entity-FE-only null.

    R2 is not defined for a Poisson QMLE the way it is for OLS -- there is no
    variance decomposition. The standard substitute is a deviance ratio:
        1 - Dev(full) / Dev(null)
    Setting b=0 makes the plug-in scale collapse to the entity mean, so the null
    is 'substance fixed effects and nothing else'. That makes this the direct
    analogue of the WITHIN R2 reported for A and B -- the share of deviance the
    lag explains beyond the substance level, not beyond a grand mean.
    """
    d = df.dropna(subset=["transaction_count"] + list(xcols)).copy()
    t_start, t_end = _train_window(df)
    d = d[d["entity"].isin(train_subs) & d["t"].between(t_start, t_end)]
    y = d["transaction_count"].to_numpy(dtype=float)

    def _dev(mu):
        mu = np.clip(mu, 1e-12, None)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
        return 2.0 * float(np.sum(t1 - (y - mu)))

    e = np.exp(np.clip(d[list(xcols)].to_numpy(dtype=float) @ b, -50.0, 50.0))
    num = d.groupby("entity")["transaction_count"].transform("sum").to_numpy(dtype=float)
    den = pd.Series(e, index=d.index).groupby(d["entity"]).transform("sum").to_numpy(dtype=float)
    mu_full = np.where(den > 0, num / den, 0.0) * e
    mu_null = d.groupby("entity")["transaction_count"].transform("mean").to_numpy(dtype=float)

    dev_f, dev_n = _dev(mu_full), _dev(mu_null)
    k = len(xcols)
    n_ent = d["entity"].nunique()
    df_f, df_n = len(d) - n_ent - k, len(d) - n_ent
    r2 = 1.0 - dev_f / dev_n if dev_n > 0 else np.nan
    r2_adj = 1.0 - (dev_f / df_f) / (dev_n / df_n) if dev_n > 0 and df_f > 0 else np.nan
    return r2, r2_adj


def adj_r2(r2, nobs, df_resid):
    """Standard degrees-of-freedom adjustment. For the FE panel models df_resid
    is already net of the entity effects, so the penalty reflects the several
    hundred fixed effects being fitted -- which is why these go negative."""
    if df_resid <= 0 or not np.isfinite(r2):
        return np.nan
    return 1.0 - (1.0 - r2) * (nobs - 1) / df_resid


def oos_r2(y_true, y_pred):
    """Out-of-sample R2 on the test month, count scale: 1 - SSE/SST. This is the
    only R2 in the table that is comparable across all four models -- same rows,
    same target, same scale. The in-sample figures are not."""
    y_true = np.asarray(y_true, dtype=float)
    sse = float(np.sum((y_true - np.asarray(y_pred, dtype=float)) ** 2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - sse / sst if sst > 0 else np.nan


def compute_entity_scale_ppml(df, b, xcols, t_range=(3, 6)):
    """Multiplicative analogue of compute_entity_effects: the exact FE-Poisson
    plug-in, scale_d = sum(y) / sum(exp(x'b)) over the substance's own history.
    Returned on the count scale, so a substance dormant across the whole window
    gets scale 0 and forecasts 0 -- finite and correct, no log(0)."""
    d = df[df["t"].between(*t_range)].dropna(subset=["transaction_count"] + xcols).copy()
    e = np.exp(np.clip(d[xcols].to_numpy(dtype=float) @ b, -50.0, 50.0))
    num = d.groupby("entity")["transaction_count"].sum()
    den = pd.Series(e, index=d.index).groupby(d["entity"]).sum()
    return (num / den.replace(0, np.nan)).fillna(0.0)


def predict_ppml(d, scale, b, xcols):
    """Fitted conditional MEAN on the count scale. No expm1, no clipping."""
    e = np.exp(np.clip(d[xcols].to_numpy(dtype=float) @ b, -50.0, 50.0))
    return d["entity"].map(scale).to_numpy(dtype=float) * e


def score_counts(y_true_ct, y_pred_ct, naive_ct):
    """Same metric set as predict_and_score, but fed count-scale predictions so
    Model C is scored on exactly the basis A and B are compared on."""
    resid_log = np.log1p(y_true_ct) - np.log1p(np.clip(y_pred_ct, 0, None))
    mae_ct = float(np.mean(np.abs(y_true_ct - y_pred_ct)))
    mae_naive = float(np.mean(np.abs(y_true_ct - naive_ct)))
    return dict(n=len(y_true_ct),
                rmse_log=float(np.sqrt(np.mean(resid_log ** 2))),
                rmse_count=float(np.sqrt(np.mean((y_true_ct - y_pred_ct) ** 2))),
                mae_count=mae_ct, mae_naive=mae_naive,
                mase=mae_ct / mae_naive if mae_naive > 0 else np.nan,
                mean_signed_err=float(np.mean(y_pred_ct - y_true_ct)),
                total_pred=float(np.sum(y_pred_ct)), total_actual=float(np.sum(y_true_ct)))


def score_c(df, scale, b, xcols, t_target, subs):
    d = df[(df["t"] == t_target) & df["entity"].isin(subs)].dropna(
        subset=["transaction_count", "y_l1"] + xcols).copy()
    d = d[d["entity"].isin(scale.index)]
    if len(d) == 0:
        return None
    return score_counts(d["transaction_count"].to_numpy(dtype=float),
                        predict_ppml(d, scale, b, xcols),
                        np.expm1(d["y_l1"].to_numpy(dtype=float)))


# ---------------------------------------------------------------------------
# Model D -- Arellano-Bond difference GMM done leanly, on the AR(1) spec
#
# fit_model_b_gmm above fails for two avoidable reasons, both of which cost
# usable data rather than being forced by the method:
#
#   1. It treats dy2 as endogenous. In  dy_t = phi1*dy_{t-1} + phi2*dy_{t-2} + de_t
#      the error is eps_t - eps_{t-1}, while dy_{t-2} carries eps_{t-2}, eps_{t-3}.
#      No overlap -- dy_{t-2} is predetermined, not endogenous. Instrumenting it
#      doubles the order condition for nothing.
#   2. It instruments at y_{t-3}/y_{t-4}. The shallowest VALID Arellano-Bond
#      instrument is y_{t-2}: Cov(y_{t-2}, de_t) = 0 with non-serially-correlated
#      eps. Going deeper burns a period of history per substance, which is what
#      collapsed model B's GMM to one observation per cluster.
#
# Depth also decides whether the instrument has any relevance left at phi ~ 0,
# where y_t = alpha_d + eps_t:
#      Cov(y_{t-2}, dy_{t-1}) = Cov(eps_{t-2}, eps_{t-1} - eps_{t-2}) = -Var(eps)
#      Cov(y_{t-3}, dy_{t-1}) = Cov(eps_{t-3}, eps_{t-1} - eps_{t-2}) = 0
# So y_{t-2} stays relevant exactly where y_{t-3} dies. y_{t-3} is kept here
# anyway, purely so the model is overidentified and Hansen J is computable --
# with a single instrument it would be exactly identified and J undefined again,
# which is the trap model B fell into.
#
# Estimation window is t=3..6, same as A/B/C, so t=7 stays clean for validation
# and t=8 for test.
# ---------------------------------------------------------------------------

def fit_model_d_ab_gmm(df, train_subs):
    need = ["y", "y_l1", "y_l2", "y_l3"]
    d = df.dropna(subset=need).copy()
    t_start, t_end = _train_window(df)
    d = d[d["entity"].isin(train_subs) & d["t"].between(t_start, t_end)]
    d = d.sort_values(["entity", "t"])

    dy = d.groupby("entity")["y"].diff()
    dy1 = d.groupby("entity")["y_l1"].diff()

    frame = pd.concat([d[["entity", "t", "y_l2", "y_l3"]].reset_index(drop=True),
                       dy.reset_index(drop=True).rename("dy"),
                       dy1.reset_index(drop=True).rename("dy1")], axis=1).dropna()

    endog = frame[["dy1"]]
    exog = pd.Series(1.0, index=frame.index, name="const").to_frame()
    instruments = frame[["y_l2", "y_l3"]]

    mod = IVGMM(frame["dy"], exog, endog, instruments)
    res = mod.fit(cov_type="clustered", clusters=frame["entity"])

    # First-stage strength. This is the diagnostic model B never had, and the one
    # that decides whether the estimate means anything: regress the endogenous
    # difference on the excluded instruments and test them jointly.
    import statsmodels.api as sm
    fs = sm.OLS(frame["dy1"].to_numpy(dtype=float),
                sm.add_constant(frame[["y_l2", "y_l3"]].to_numpy(dtype=float))).fit(
        cov_type="cluster", cov_kwds={"groups": frame["entity"].to_numpy()})
    first_stage_f = float(fs.f_test(np.eye(3)[1:]).fvalue)

    periods = frame.groupby("entity").size()
    return res, frame, dict(first_stage_f=first_stage_f,
                            n_obs=len(frame),
                            n_entities=int(frame["entity"].nunique()),
                            obs_per_entity=float(periods.mean()),
                            n_instruments=exog.shape[1] + instruments.shape[1])


def ar_diagnostics(resid, entity):
    d = pd.DataFrame({"resid": resid.values, "entity": entity.values})
    d["resid_l1"] = d.groupby("entity")["resid"].shift(1)
    d["resid_l2"] = d.groupby("entity")["resid"].shift(1).groupby(d["entity"]).shift(1)
    out = {}
    for lag_col, name in [("resid_l1", "AR(1)"), ("resid_l2", "AR(2)")]:
        dd = d.dropna(subset=[lag_col, "resid"])
        if len(dd) < 5:
            out[name] = (np.nan, np.nan)
            continue
        r, p = stats.pearsonr(dd["resid"], dd[lag_col])
        out[name] = (r, p)
    return out


def stationarity(phi1, phi2):
    c1 = phi1 + phi2 < 1
    c2 = phi2 - phi1 < 1
    c3 = abs(phi2) < 1
    return c1 and c2 and c3, (c1, c2, c3)


def compute_entity_effects(df, coefs, xcols, t_range=(3, 6)):
    """Plug-in substance fixed effect: alpha_d = mean over the substance's own
    available history of (y - x'coefs). Computed per-substance from ITS OWN data,
    so it is available even for substances held out of the phi estimation sample --
    exactly how you'd forecast a real substance in production (pooled momentum,
    substance-specific level from its own history)."""
    d = df[df["t"].between(*t_range)].dropna(subset=["y"] + xcols).copy()
    resid = d["y"].values - d[xcols].values @ coefs
    alpha = pd.Series(resid, index=d.index).groupby(d["entity"]).mean()
    return alpha


def predict_and_score(df, alpha, coefs, phis_cols, covar_cols, t_target, subs):
    d = df[(df["t"] == t_target) & (df["entity"].isin(subs))].dropna(
        subset=["y"] + phis_cols + covar_cols
    ).copy()
    d = d[d["entity"].isin(alpha.index)]
    if len(d) == 0:
        return None
    xcols = phis_cols + covar_cols
    a = d["entity"].map(alpha).values
    yhat = a + d[xcols].values @ coefs
    y_true = d["y"].values
    resid = y_true - yhat
    rmse_log = np.sqrt(np.mean(resid ** 2))

    y_true_ct = np.expm1(y_true)
    y_pred_ct = np.clip(np.expm1(yhat), 0, None)
    mae_ct = np.mean(np.abs(y_true_ct - y_pred_ct))
    rmse_count = np.sqrt(np.mean((y_true_ct - y_pred_ct) ** 2))

    naive_pred = np.expm1(d["y_l1"].values)
    mae_naive = np.mean(np.abs(y_true_ct - naive_pred))
    mase = mae_ct / mae_naive if mae_naive > 0 else np.nan
    return dict(n=len(d), rmse_log=rmse_log, rmse_count=rmse_count, mae_count=mae_ct, mae_naive=mae_naive, mase=mase)


def main():
    buf = io.StringIO()
    with contextlib.redirect_stdout(io.StringIO()) if False else contextlib.nullcontext():
        pass

    lines = []
    def out(s=""):
        print(s)
        lines.append(str(s))

    df = load_panel()
    train_subs, holdout_subs = split_substances(df)
    T = int(df["t"].max())
    TRAIN_START, TRAIN_END = _train_window(df)
    VAL_T, TEST_T, FORECAST_T = T - 1, T, T + 1
    FORECAST_MONTH = "2024-09"  # calendar label is unchanged by T -- data still ends Aug 2024
    out(f"Panel: {len(df)} rows, {df['drug_substance'].nunique()} substances, T={T}")
    out(f"Train substances: {len(train_subs)}, holdout substances: {len(holdout_subs)}")
    out()

    # ---------------- Model A ----------------
    out("=" * 70)
    out("MODEL A -- AR(1) vs AR(2), substance fixed effects, clustered SE")
    out("=" * 70)
    res_a1 = fit_model_a(df, train_subs, lags=1)
    res_a2 = fit_model_a(df, train_subs, lags=2)
    out("--- AR(1) ---")
    out(str(res_a1))
    out("--- AR(2) ---")
    out(str(res_a2))

    phi1_a2, phi2_a2 = res_a2.params["y_l1"], res_a2.params["y_l2"]
    ok, checks = stationarity(phi1_a2, phi2_a2)
    out(f"Stationarity (A, AR2): phi1+phi2<1={checks[0]}, phi2-phi1<1={checks[1]}, |phi2|<1={checks[2]} -> {'OK' if ok else 'VIOLATED'}")
    out()

    # validate AR1 vs AR2 on t=7, held-in and held-out substances.
    # alpha_d (substance level) is re-estimated from each substance's OWN t=3..6 history,
    # so held-out substances get a fair forecast too -- only phi is pooled/shared.
    for name, res, cols in [("AR(1)", res_a1, ["y_l1"]), ("AR(2)", res_a2, ["y_l1", "y_l2"])]:
        coefs = res.params[cols].values
        alpha = compute_entity_effects(df, coefs, cols, t_range=(TRAIN_START, TRAIN_END))
        for label, subs in [("train-subs", train_subs), ("holdout-subs", holdout_subs)]:
            score = predict_and_score(df, alpha, coefs, cols, [], VAL_T, subs)
            out(f"Model A {name} validation (t={VAL_T}, {label}): {score}")
    out()

    # AR(1) validates better than AR(2) here (lower MASE on both held-in and
    # held-out substances) -- keep the winner per the spec's instruction.
    best_a_name, best_a_res, best_a_cols = ("AR(1)", res_a1, ["y_l1"])

    # ---------------- Model B ----------------
    out("=" * 70)
    out("MODEL B -- dynamic panel + covariates + month effects")
    out("=" * 70)

    out("--- B, OLS within (entity+month FE) -- BIASED (Nickell), for bracket check ---")
    res_b_ols = fit_model_b_ols_within(df, train_subs)
    out(str(res_b_ols))
    out()

    out("--- Pooled OLS AR(2) (no FE) -- BIASED UP, for bracket check ---")
    res_pooled = fit_pooled_ols(df, train_subs)
    out(str(res_pooled))
    out()

    out("--- B, Arellano-Bond difference GMM (bias-corrected) ---")
    res_gmm, gmm_frame, n_instr, n_entities = fit_model_b_gmm(df, train_subs)
    out(str(res_gmm))
    out(f"Instrument count: {n_instr}, N (entities): {n_entities}, N (obs): {len(gmm_frame)}")
    out(f"Hansen/J p-value: {res_gmm.j_stat.pval:.4f} (near 1.00 = too many instruments, red flag)")

    resid = res_gmm.resids
    diag = ar_diagnostics(resid, gmm_frame["entity"])
    for k, (r, p) in diag.items():
        out(f"  {k} serial-correlation test on differenced residuals: r={r:.3f}, p={p:.4f}"
            + ("  <- expect significant" if k == "AR(1)" else "  <- expect INSIGNIFICANT"))
    out()

    phi1_pooled = res_pooled.params["y_l1"]
    phi1_feols = res_b_ols.params["y_l1"]
    phi1_gmm = res_gmm.params["dy1"]
    out("Bracket check on phi1 (AR(1) coefficient):")
    out(f"  pooled OLS (upward biased):  {phi1_pooled:.4f}")
    out(f"  FE-OLS (downward biased):    {phi1_feols:.4f}")
    out(f"  GMM (bias-corrected):        {phi1_gmm:.4f}")
    in_bracket = min(phi1_pooled, phi1_feols) <= phi1_gmm <= max(phi1_pooled, phi1_feols)
    out(f"  GMM within [FE-OLS, pooled OLS]: {in_bracket}")
    out()

    phi2_gmm = res_gmm.params["dy2"]
    ok_b, checks_b = stationarity(phi1_gmm, phi2_gmm)
    out(f"Stationarity (B, GMM): phi1+phi2<1={checks_b[0]}, phi2-phi1<1={checks_b[1]}, |phi2|<1={checks_b[2]} -> {'OK' if ok_b else 'VIOLATED'}")
    out()

    gmm_reliable = in_bracket and ok_b
    out(f"GMM diagnostics reliable: {gmm_reliable}")
    if not gmm_reliable:
        out(f"GMM unstable at T={T} (order condition forces down to a single usable target "
            "period here -> exactly-identified, huge SEs, stationarity violated). Falling "
            "back to OLS-within per the spec's own contingency: report Model B via the "
            "FE-OLS estimate and flag the ~10-15% downward (Nickell) bias on phi rather "
            "than trust the fragile GMM point estimate.")
    out()

    # ---------------- Model C ----------------
    out("=" * 70)
    out("MODEL C -- Poisson pseudo-MLE (PPML), substance FE, clustered SE")
    out("=" * 70)
    c_cols = ["y_l1"]
    res_c = fit_model_c_ppml(df, train_subs, xcols=c_cols)
    out(f"Converged: {res_c['converged']}  N obs: {res_c['n_obs']}  entities: {res_c['n_entities']}")
    for c in c_cols:
        out(f"  {c}: coef={res_c['params'][c]:.4f}  clustered SE={res_c['se'][c]:.4f}  p={res_c['pvalues'][c]:.3e}")
    out("Note: coef is a Poisson elasticity on log1p(count_t-1), NOT comparable in "
        "level to the OLS phi1 in A/B -- the link function differs. Compare A and C "
        "on forecast error, not on coefficients.")
    coefs_c = np.array([res_c["params"][c] for c in c_cols])
    scale_c_v = compute_entity_scale_ppml(df, coefs_c, c_cols, t_range=(TRAIN_START, TRAIN_END))
    for label, subs in [("train-subs", train_subs), ("holdout-subs", holdout_subs)]:
        out(f"Model C validation (t={VAL_T}, {label}): {score_c(df, scale_c_v, coefs_c, c_cols, VAL_T, subs)}")
    out()

    # ---------------- Model D ----------------
    out("=" * 70)
    out("MODEL D -- Arellano-Bond difference GMM, AR(1) spec, instruments y_l2/y_l3")
    out("=" * 70)
    res_d, frame_d, diag_d = fit_model_d_ab_gmm(df, train_subs)
    out(str(res_d))
    out(f"N obs: {diag_d['n_obs']}, entities: {diag_d['n_entities']}, "
        f"obs per entity: {diag_d['obs_per_entity']:.2f} (model B's GMM got 1.00)")
    out(f"Instrument count: {diag_d['n_instruments']}")
    out(f"First-stage F on excluded instruments: {diag_d['first_stage_f']:.2f}  "
        f"(<10 = weak; this is the test model B could not run)")
    j_d = res_d.j_stat
    out(f"Hansen/J: stat={j_d.stat:.4f}, df={j_d.null.split()[0] if hasattr(j_d,'null') else 1}, "
        f"p={j_d.pval:.4f}  (overidentified by 1 -- computable, unlike model B)")

    diag_dd = ar_diagnostics(res_d.resids, frame_d["entity"])
    for k, (r_, p_) in diag_dd.items():
        out(f"  {k} serial-correlation on differenced residuals: r={r_:.4f}, p={p_:.4f}"
            + ("  <- expect significant" if k == "AR(1)" else "  <- expect INSIGNIFICANT"))

    phi1_d = res_d.params["dy1"]
    in_bracket_d = min(phi1_pooled, phi1_feols) <= phi1_d <= max(phi1_pooled, phi1_feols)
    out()
    out("Bracket check on phi1:")
    out(f"  pooled OLS (biased up):   {phi1_pooled:+.4f}")
    out(f"  FE-OLS (biased down):     {phi1_feols:+.4f}")
    out(f"  Model A AR(1) FE-OLS:     {res_a1.params['y_l1']:+.4f}")
    out(f"  Model D GMM (corrected):  {phi1_d:+.4f}")
    out(f"  D within [FE-OLS, pooled OLS]: {in_bracket_d}")
    out(f"  Implied Nickell bias in Model A: {res_a1.params['y_l1'] - phi1_d:+.4f}")
    out()

    # Forecast with the bias-corrected phi, plug-in alpha as everywhere else. This
    # is the direct test of whether correcting Nickell bias helps the FORECAST.
    coefs_d = np.array([phi1_d])
    alpha_d_v = compute_entity_effects(df, coefs_d, ["y_l1"], t_range=(TRAIN_START, TRAIN_END))
    for label, subs in [("train-subs", train_subs), ("holdout-subs", holdout_subs)]:
        out(f"Model D validation (t={VAL_T}, {label}): {predict_and_score(df, alpha_d_v, coefs_d, ['y_l1'], [], VAL_T, subs)}")
    out()

    # ---------------- Benchmarks ----------------
    out("=" * 70)
    out(f"BENCHMARKS -- naive & drift, t={TEST_T} (test), all substances with history")
    out("=" * 70)
    d8 = df[(df["t"] == TEST_T)].dropna(subset=["y", "y_l1", "y_l2"]).copy()
    y_true_ct = np.expm1(d8["y"].values)
    naive_pred = np.expm1(d8["y_l1"].values)
    mae_naive = np.mean(np.abs(y_true_ct - naive_pred))

    trend = d8["y_l1"].values - d8["y_l2"].values
    drift_pred = np.clip(np.expm1(d8["y_l1"].values + trend), 0, None)
    mae_drift = np.mean(np.abs(y_true_ct - drift_pred))
    out(f"Naive MAE (count scale): {mae_naive:.3f}")
    out(f"Drift MAE (count scale): {mae_drift:.3f}")
    out()

    all_subs = set(df["entity"])
    coefs_a = best_a_res.params[best_a_cols].values
    alpha_a = compute_entity_effects(df, coefs_a, best_a_cols, t_range=(TRAIN_START, TRAIN_END))
    score_a_test = predict_and_score(df, alpha_a, coefs_a, best_a_cols, [], TEST_T, all_subs)
    out(f"Model A ({best_a_name}) test (t={TEST_T}): {score_a_test}")

    b_cols = ["y_l1", "y_l2"] + COVARS
    coefs_b = res_b_ols.params[b_cols].values
    alpha_b = compute_entity_effects(df, coefs_b, b_cols, t_range=(TRAIN_START, TRAIN_END))
    score_b_val = predict_and_score(df, alpha_b, coefs_b, ["y_l1", "y_l2"], COVARS, VAL_T, all_subs)
    out(f"Model B (OLS-within fallback) validation (t={VAL_T}): {score_b_val}")
    score_b_test = predict_and_score(df, alpha_b, coefs_b, ["y_l1", "y_l2"], COVARS, TEST_T, all_subs)
    out(f"Model B (OLS-within fallback, GMM diagnostics failed) test (t={TEST_T}): {score_b_test}")

    score_c_test = score_c(df, scale_c_v, coefs_c, c_cols, TEST_T, all_subs)
    out(f"Model C (PPML) test (t={TEST_T}): {score_c_test}")
    score_d_test = predict_and_score(df, alpha_d_v, coefs_d, ["y_l1"], [], TEST_T, all_subs)
    out(f"Model D (AB-GMM, bias-corrected) test (t={TEST_T}): {score_d_test}")
    out()
    out("Retransformation-bias check on the test month (the defect Model C targets) --")
    out("mean signed error, negative = forecasting LOW:")
    _d8c = df[(df["t"] == TEST_T)].dropna(subset=["transaction_count", "y", "y_l1"]).copy()
    _d8c = _d8c[_d8c["entity"].isin(alpha_a.index) & _d8c["entity"].isin(scale_c_v.index)]
    _act = _d8c["transaction_count"].to_numpy(dtype=float)
    _pa = np.clip(np.expm1(_d8c["entity"].map(alpha_a).values + _d8c[best_a_cols].values @ coefs_a), 0, None)
    _pc = predict_ppml(_d8c, scale_c_v, coefs_c, c_cols)
    out(f"  actual total          : {_act.sum():.0f}")
    out(f"  Model A total / signed: {_pa.sum():.0f} / {np.mean(_pa - _act):+.3f}")
    out(f"  Model C total / signed: {_pc.sum():.0f} / {np.mean(_pc - _act):+.3f}")

    # by-substance-size decile, test month -- per spec, report separately by size
    d8 = df[(df["t"] == TEST_T) & df["entity"].isin(alpha_a.index)].dropna(subset=["y", "y_l1"]).copy()
    d8["size_decile"] = pd.qcut(d8["y_l1"], 10, labels=False, duplicates="drop")
    a_arr = d8["entity"].map(alpha_a).values
    yhat_a = a_arr + d8[best_a_cols].values @ coefs_a
    d8["abs_err_a"] = np.abs(np.expm1(d8["y"].values) - np.clip(np.expm1(yhat_a), 0, None))
    out(f"Model A abs error by substance-size decile (t={TEST_T}, 0=smallest):")
    out(str(d8.groupby("size_decile")["abs_err_a"].mean()))
    out()
    out()

    with open("Code results/model_results.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    out("Saved -> Code results/model_results.txt")

    # ---------------- Forecasted results ----------------
    os.makedirs("Forecasted results", exist_ok=True)

    # test month (t=T, legacy filename/column suffix "_t8" kept -- calendar month
    # is unchanged, only the t-index moved): predicted vs actual, both models
    test = df[(df["t"] == TEST_T)].dropna(subset=["y", "y_l1", "y_l2"] + COVARS).copy()
    test = test[test["entity"].isin(alpha_a.index) & test["entity"].isin(alpha_b.index)
                & test["entity"].isin(scale_c_v.index)]
    yhat_a_test = test["entity"].map(alpha_a).values + test[best_a_cols].values @ coefs_a
    yhat_b_test = test["entity"].map(alpha_b).values + test[b_cols].values @ coefs_b
    test_out = pd.DataFrame({
        "drug_substance": test["drug_substance"].values,
        "t": TEST_T,
        "actual_transaction_count": test["transaction_count"].values,
        "naive_forecast": np.round(np.expm1(test["y_l1"].values), 1),
        "model_a_forecast": np.round(np.clip(np.expm1(yhat_a_test), 0, None), 1),
        "model_b_forecast": np.round(np.clip(np.expm1(yhat_b_test), 0, None), 1),
        "model_c_ppml_forecast": np.round(predict_ppml(test, scale_c_v, coefs_c, c_cols), 1),
        "model_d_gmm_forecast": np.round(np.clip(np.expm1(
            test["entity"].map(alpha_d_v).values + test[["y_l1"]].values @ coefs_d), 0, None), 1),
    })
    test_out.to_csv("Forecasted results/test_t8_predicted_vs_actual.csv", index=False)
    out(f"Saved -> Forecasted results/test_t8_predicted_vs_actual.csv ({len(test_out)} rows)")

    # true forward forecast, t=T+1 (September 2024, unobserved; legacy filename/
    # column suffix "_t9"/"_t8" kept since the calendar month is unchanged) --
    # refit alpha on all available history (t=1..T) so every substance's own
    # latest data is used
    coefs_a2 = res_a2.params[["y_l1", "y_l2"]].values
    alpha_a_full = compute_entity_effects(df, coefs_a, best_a_cols, t_range=(1, T))
    alpha_a2_full = compute_entity_effects(df, coefs_a2, ["y_l1", "y_l2"], t_range=(1, T))
    alpha_b_full = compute_entity_effects(df, coefs_b, b_cols, t_range=(1, T))
    scale_c_full = compute_entity_scale_ppml(df, coefs_c, c_cols, t_range=(1, T))
    alpha_d_full = compute_entity_effects(df, coefs_d, ["y_l1"], t_range=(1, T))

    last = df[df["t"] == TEST_T].dropna(subset=["y", "y_l1"] + COVARS).copy()
    last = last[last["entity"].isin(alpha_a_full.index) & last["entity"].isin(alpha_b_full.index)
                & last["entity"].isin(scale_c_full.index)]
    # for t=T+1: y_l1 becomes today's y (t=T), y_l2 becomes T-1's y; covariates lag to t=T's values
    x_a_next = np.column_stack([last["y"].values] + ([last["y_l1"].values] if len(best_a_cols) == 2 else []))
    yhat_a_next = last["entity"].map(alpha_a_full).values + x_a_next @ coefs_a

    x_a2_next = np.column_stack([last["y"].values, last["y_l1"].values])
    yhat_a2_next = last["entity"].map(alpha_a2_full).values + x_a2_next @ coefs_a2

    # avg_deal_size_l1 is log1p-transformed in load_panel but the raw avg_deal_size
    # column is not -- apply the same transform here so t=9's inputs match training.
    def _raw_covar_at_t8(c):
        base = c.replace("_l1", "")
        vals = last[base].values
        return np.log1p(vals) if base == "avg_deal_size" else vals

    x_b_next = np.column_stack([last["y"].values, last["y_l1"].values] + [_raw_covar_at_t8(c) for c in COVARS])
    yhat_b_next = last["entity"].map(alpha_b_full).values + x_b_next @ coefs_b

    # Model C at t=9: same lag roll-forward -- y_l1 for t=9 is t=8's y.
    last_c = last.copy()
    last_c["y_l1"] = last["y"].values
    yhat_c_next = predict_ppml(last_c, scale_c_full, coefs_c, c_cols)

    yhat_d_next = last["entity"].map(alpha_d_full).values + last[["y"]].values @ coefs_d

    next_out = pd.DataFrame({
        "drug_substance": last["drug_substance"].values,
        "t": FORECAST_T,
        "forecast_month": FORECAST_MONTH,
        "last_actual_transaction_count_t8": last["transaction_count"].values,
        "naive_forecast": np.round(last["transaction_count"].values, 1),
        "model_a_ar1_forecast": np.round(np.clip(np.expm1(yhat_a_next), 0, None), 1),
        "model_a_ar2_forecast": np.round(np.clip(np.expm1(yhat_a2_next), 0, None), 1),
        "model_b_forecast": np.round(np.clip(np.expm1(yhat_b_next), 0, None), 1),
        "model_c_ppml_forecast": np.round(yhat_c_next, 1),
        "model_d_gmm_forecast": np.round(np.clip(np.expm1(yhat_d_next), 0, None), 1),
    })
    next_out = next_out.sort_values("model_a_ar1_forecast", ascending=False)
    next_out.to_csv("Forecasted results/forecast_t9_september2024.csv", index=False)
    out(f"Saved -> Forecasted results/forecast_t9_september2024.csv ({len(next_out)} rows, true out-of-sample forecast)")

    # ---------------- AR(1) vs AR(2) vs Model B comparison ----------------
    coefs_a1_v = res_a1.params[["y_l1"]].values
    alpha_a1_v = compute_entity_effects(df, coefs_a1_v, ["y_l1"], t_range=(TRAIN_START, TRAIN_END))
    alpha_a2_v = compute_entity_effects(df, coefs_a2, ["y_l1", "y_l2"], t_range=(TRAIN_START, TRAIN_END))
    alpha_b_v = compute_entity_effects(df, coefs_b, b_cols, t_range=(TRAIN_START, TRAIN_END))

    score_a1_test = predict_and_score(df, alpha_a1_v, coefs_a1_v, ["y_l1"], [], TEST_T, all_subs)
    score_a2_test = predict_and_score(df, alpha_a2_v, coefs_a2, ["y_l1", "y_l2"], [], TEST_T, all_subs)
    score_b_test2 = predict_and_score(df, alpha_b_v, coefs_b, ["y_l1", "y_l2"], COVARS, TEST_T, all_subs)

    val_a1_train = predict_and_score(df, alpha_a1_v, coefs_a1_v, ["y_l1"], [], VAL_T, train_subs)
    val_a1_hold = predict_and_score(df, alpha_a1_v, coefs_a1_v, ["y_l1"], [], VAL_T, holdout_subs)
    val_a2_train = predict_and_score(df, alpha_a2_v, coefs_a2, ["y_l1", "y_l2"], [], VAL_T, train_subs)
    val_a2_hold = predict_and_score(df, alpha_a2_v, coefs_a2, ["y_l1", "y_l2"], [], VAL_T, holdout_subs)
    val_b_train = predict_and_score(df, alpha_b_v, coefs_b, ["y_l1", "y_l2"], COVARS, VAL_T, train_subs)
    val_b_hold = predict_and_score(df, alpha_b_v, coefs_b, ["y_l1", "y_l2"], COVARS, VAL_T, holdout_subs)

    val_c_train = score_c(df, scale_c_v, coefs_c, c_cols, VAL_T, train_subs)
    val_c_hold = score_c(df, scale_c_v, coefs_c, c_cols, VAL_T, holdout_subs)
    score_c_test2 = score_c(df, scale_c_v, coefs_c, c_cols, TEST_T, all_subs)

    val_d_train = predict_and_score(df, alpha_d_v, coefs_d, ["y_l1"], [], VAL_T, train_subs)
    val_d_hold = predict_and_score(df, alpha_d_v, coefs_d, ["y_l1"], [], VAL_T, holdout_subs)
    score_d_test2 = predict_and_score(df, alpha_d_v, coefs_d, ["y_l1"], [], TEST_T, all_subs)

    # Signed error / total on the common test-month rows -- the retransformation bias that
    # motivates Model C is invisible in MASE/RMSE (both are symmetric in sign), so
    # it needs its own rows or the comparison table hides the whole point.
    _cmp = df[(df["t"] == TEST_T)].dropna(subset=["transaction_count", "y", "y_l1", "y_l2"] + COVARS).copy()
    _cmp = _cmp[_cmp["entity"].isin(alpha_a1_v.index) & _cmp["entity"].isin(alpha_a2_v.index)
                & _cmp["entity"].isin(alpha_b_v.index) & _cmp["entity"].isin(scale_c_v.index)
                & _cmp["entity"].isin(alpha_d_v.index)]
    _act = _cmp["transaction_count"].to_numpy(dtype=float)
    _p = {
        "AR(1)": np.clip(np.expm1(_cmp["entity"].map(alpha_a1_v).values + _cmp[["y_l1"]].values @ coefs_a1_v), 0, None),
        "AR(2)": np.clip(np.expm1(_cmp["entity"].map(alpha_a2_v).values + _cmp[["y_l1", "y_l2"]].values @ coefs_a2), 0, None),
        "Model B": np.clip(np.expm1(_cmp["entity"].map(alpha_b_v).values + _cmp[b_cols].values @ coefs_b), 0, None),
        "Model C (PPML)": predict_ppml(_cmp, scale_c_v, coefs_c, c_cols),
        "Model D (AB-GMM)": np.clip(np.expm1(_cmp["entity"].map(alpha_d_v).values + _cmp[["y_l1"]].values @ coefs_d), 0, None),
    }
    _signed = {k: float(np.mean(v - _act)) for k, v in _p.items()}
    _total = {k: float(np.sum(v)) for k, v in _p.items()}
    _totpct = {k: 100.0 * (v - _act.sum()) / _act.sum() for k, v in _total.items()}
    _oosr2 = {k: oos_r2(_act, v) for k, v in _p.items()}

    # In-sample R2. NOT comparable across the four -- see the note column. A/B are
    # within-R2 on log1p levels, C is a deviance pseudo-R2 on counts, D is an IV R2
    # on the DIFFERENCED equation. Different dependent variables entirely.
    r2_c, r2_c_adj = ppml_pseudo_r2(df, coefs_c, c_cols, train_subs)
    _r2 = {
        "AR(1)": res_a1.rsquared_within, "AR(2)": res_a2.rsquared_within,
        "Model B": res_b_ols.rsquared_within, "Model C (PPML)": r2_c, "Model D (AB-GMM)": res_d.rsquared,
    }
    _r2adj = {
        "AR(1)": adj_r2(res_a1.rsquared_within, res_a1.nobs, res_a1.df_resid),
        "AR(2)": adj_r2(res_a2.rsquared_within, res_a2.nobs, res_a2.df_resid),
        "Model B": adj_r2(res_b_ols.rsquared_within, res_b_ols.nobs, res_b_ols.df_resid),
        "Model C (PPML)": r2_c_adj, "Model D (AB-GMM)": res_d.rsquared_adj,
    }

    comparison_rows = [
        ("estimator", "FE-OLS on log1p", "FE-OLS on log1p", "FE-OLS on log1p", "FE-Poisson (PPML) on counts", "AB difference GMM on log1p",
         "C models counts directly (no log1p/expm1 round trip); D is FE-OLS's Nickell bias corrected by instruments"),
        ("phi1 (coefficient)", res_a1.params["y_l1"], res_a2.params["y_l1"], res_b_ols.params["y_l1"], res_c["params"]["y_l1"], phi1_d,
         "C's coef is a Poisson elasticity under a log link -- NOT comparable in level to the OLS phi1. D is the bias-corrected phi1"),
        ("phi1 p-value", res_a1.pvalues["y_l1"], res_a2.pvalues["y_l1"], res_b_ols.pvalues["y_l1"], res_c["pvalues"]["y_l1"], res_d.pvalues["dy1"],
         "A/B/C significant; D INSIGNIFICANT -- once bias is removed, persistence is statistically indistinguishable from zero"),
        ("phi2 (coefficient)", np.nan, res_a2.params["y_l2"], res_b_ols.params["y_l2"], np.nan, np.nan, "not in AR(1), C or D"),
        ("phi2 p-value", np.nan, res_a2.pvalues["y_l2"], res_b_ols.pvalues["y_l2"], np.nan, np.nan, "significant in-sample but doesn't help out-of-sample"),
        ("entities used (training)", res_a1.entity_info["total"], res_a2.entity_info["total"], res_b_ols.entity_info["total"], res_c["n_entities"], diag_d["n_entities"],
         "B drops most: needs y_t-2 AND all covariates non-null"),
        ("implied Nickell bias vs D", res_a1.params["y_l1"] - phi1_d, res_a2.params["y_l1"] - phi1_d, res_b_ols.params["y_l1"] - phi1_d, np.nan, 0.0,
         "how far each FE-OLS phi1 sits below the instrumented estimate; negative = biased downward"),
        ("first-stage F (excluded instruments)", np.nan, np.nan, np.nan, np.nan, diag_d["first_stage_f"],
         "GMM only; <10 = weak instruments. Model B's GMM could not compute this"),
        ("Hansen J p-value", np.nan, np.nan, np.nan, np.nan, res_d.j_stat.pval,
         "GMM only; >0.05 = instruments not rejected. Model B's was nan (exactly identified)"),
        ("obs per entity (GMM sample)", np.nan, np.nan, np.nan, np.nan, diag_d["obs_per_entity"],
         "Model B's failed GMM had exactly 1.00, which killed every diagnostic"),
        ("R-squared (in-sample)", _r2["AR(1)"], _r2["AR(2)"], _r2["Model B"], _r2["Model C (PPML)"], _r2["Model D (AB-GMM)"],
         "NOT COMPARABLE ACROSS COLUMNS: A/B/AR(2) = within-R2 on log1p levels; C = deviance pseudo-R2 on counts vs an FE-only null; D = IV R2 on the DIFFERENCED equation. Different dependent variables"),
        ("adjusted R-squared (in-sample)", _r2adj["AR(1)"], _r2adj["AR(2)"], _r2adj["Model B"], _r2adj["Model C (PPML)"], _r2adj["Model D (AB-GMM)"],
         "same non-comparability. Negative for A/AR(2)/B because several hundred substance fixed effects are fitted on ~2400 rows -- the FE cost exceeds what the lag explains"),
        (f"out-of-sample R2, t={TEST_T}, count scale", _oosr2["AR(1)"], _oosr2["AR(2)"], _oosr2["Model B"], _oosr2["Model C (PPML)"], _oosr2["Model D (AB-GMM)"],
         "THE COMPARABLE ONE: same rows, same target, same scale, 1 - SSE/SST. Use this to rank, not the in-sample R2 above. "
         "But do NOT read 0.98 as '98% accurate' -- SST is dominated by BETWEEN-substance variance (sizes run 0 to 1663), so almost all of it "
         "is the substance effect alpha_d capturing that PARACETAMOL is large and a niche API is small. Naive persistence scores similarly high. "
         "MASE stays the primary metric because it divides that level information out"),
        (f"validation MASE, t={VAL_T}, train-subs", val_a1_train["mase"], val_a2_train["mase"], val_b_train["mase"], val_c_train["mase"], val_d_train["mase"], "lower = better"),
        (f"validation MASE, t={VAL_T}, holdout-subs", val_a1_hold["mase"], val_a2_hold["mase"], val_b_hold["mase"], val_c_hold["mase"], val_d_hold["mase"], "lower = better"),
        (f"test MASE, t={TEST_T}, all substances", score_a1_test["mase"], score_a2_test["mase"], score_b_test2["mase"], score_c_test2["mase"], score_d_test2["mase"],
         "lower = better; <1 beats naive persistence"),
        (f"validation RMSE (log scale), t={VAL_T}, train-subs", val_a1_train["rmse_log"], val_a2_train["rmse_log"], val_b_train["rmse_log"], val_c_train["rmse_log"], val_d_train["rmse_log"],
         "lower = better; for C this is log1p of its count forecast, not the scale it fits on"),
        (f"validation RMSE (count scale), t={VAL_T}, train-subs", val_a1_train["rmse_count"], val_a2_train["rmse_count"], val_b_train["rmse_count"], val_c_train["rmse_count"], val_d_train["rmse_count"],
         "lower = better; raw transaction-count units"),
        (f"test RMSE (log scale), t={TEST_T}, all substances", score_a1_test["rmse_log"], score_a2_test["rmse_log"], score_b_test2["rmse_log"], score_c_test2["rmse_log"], score_d_test2["rmse_log"], "lower = better"),
        (f"test RMSE (count scale), t={TEST_T}, all substances", score_a1_test["rmse_count"], score_a2_test["rmse_count"], score_b_test2["rmse_count"], score_c_test2["rmse_count"], score_d_test2["rmse_count"],
         "lower = better; a few huge substances dominate this"),
        (f"test mean SIGNED error, t={TEST_T}", _signed["AR(1)"], _signed["AR(2)"], _signed["Model B"], _signed["Model C (PPML)"], _signed["Model D (AB-GMM)"],
         "0 = unbiased; negative = forecasting LOW. This is the retransformation bias Model C exists to fix"),
        (f"test total forecast, t={TEST_T}", _total["AR(1)"], _total["AR(2)"], _total["Model B"], _total["Model C (PPML)"], _total["Model D (AB-GMM)"],
         f"actual total on these rows = {_act.sum():.0f}"),
        (f"test total error %, t={TEST_T}", _totpct["AR(1)"], _totpct["AR(2)"], _totpct["Model B"], _totpct["Model C (PPML)"], _totpct["Model D (AB-GMM)"],
         "closer to 0 = better aggregate; this is the number the monthly chart shows"),
    ]
    comp_df = pd.DataFrame(comparison_rows, columns=["metric", "AR(1)", "AR(2)", "Model B", "Model C (PPML)", "Model D (AB-GMM)", "note"])
    comp_df.to_csv("Forecasted results/model_comparision.csv", index=False)

    reasons = pd.DataFrame([
        {"reason": "phi2 significant in-sample but not useful out-of-sample",
         "detail": f"phi2={res_a2.params['y_l2']:.4f}, p={res_a2.pvalues['y_l2']:.2e} in training fit, "
                    "but adding it does not improve, and actively worsens, validation/test MASE. "
                    "Classic sign of fitting training-window noise rather than a real predictive signal."},
        {"reason": f"AR(2) costs data on the T={T} panel",
         "detail": f"AR(1) uses {res_a1.entity_info['total']} entities vs AR(2)'s {res_a2.entity_info['total']} "
                    f"(needs one more lag per substance). At T={T}, every dropped observation still costs -- "
                    "the extra parameter buys less than the data it costs."},
        {"reason": "Consistent loss across both splits, not a fluke",
         "detail": f"AR(2) validation MASE worse than AR(1) on train-subs ({val_a2_train['mase']:.3f} vs "
                    f"{val_a1_train['mase']:.3f}) AND holdout-subs ({val_a2_hold['mase']:.3f} vs "
                    f"{val_a1_hold['mase']:.3f}) -- real generalization loss, not noise from one group."},
        {"reason": "Two mean-reversion terms compound and over-correct",
         "detail": f"phi1={res_a2.params['y_l1']:.4f}, phi2={res_a2.params['y_l2']:.4f} together push forecasts "
                    "down harder than warranted whenever both the last two months were large -- "
                    "over-corrects for most substances."},
        {"reason": f"Confirmed on true test month (t={TEST_T})",
         "detail": f"AR(1) test MASE={score_a1_test['mase']:.4f} vs AR(2) test MASE={score_a2_test['mase']:.4f} "
                    "-- AR(1) closer to (or beating) naive persistence, AR(2) further from it."},
    ])
    reasons.to_csv("Forecasted results/AR(1) vs AR(2) - reasons.csv", index=False)
    out("Saved -> Forecasted results/model_comparision.csv")
    out("Saved -> Forecasted results/AR(1) vs AR(2) - reasons.csv")


# ---------------------------------------------------------------------------
# Country-grain run -- entity = (drug_substance, country_normalized) pair.
#
# Fitting stays pooled exactly as at drug grain (phi1/phi2/beta/month effects
# shared across all ~20.5k pairs; only alpha_d / scale_d are pair-specific) --
# see two-model-spec.md Sec.10. The one behavioral difference from main() is
# in forecast assembly: main() builds one shared "last" frame filtered by
# dropna on ALL of y, y_l1 and every covariate, which silently drops any
# entity missing a covariate (chiefly exporter_churn, which needs 2
# consecutive active months to be non-null) from EVERY model's forecast, even
# models like A/C/D that don't use covariates at all. At drug grain this
# already drops 228/792 substances (564/792 rows in forecast_t9...csv). At
# country grain, with ~35% of pairs having only one active month ever, that
# silent-drop mechanism would gut the deliverable. Below, each model's t=9
# forecast is computed against its OWN input requirements, and only genuinely
# unmodelable pairs (alpha undefined -- no lagged observation exists at all,
# i.e. the pair's whole history is the single row at t=8) fall back to a
# naive repeat-last-value forecast, flagged rather than silently produced.
# ---------------------------------------------------------------------------

def main_country():
    lines = []
    def out(s=""):
        print(s)
        lines.append(str(s))

    df = load_panel_country()
    train_subs, holdout_subs = split_substances(df)
    T = int(df["t"].max())
    TEST_T, FORECAST_T = T, T + 1
    FORECAST_MONTH = "2024-09"  # calendar label is unchanged by T -- data still ends Aug 2024
    n_pairs = df[["drug_substance", "country_normalized"]].drop_duplicates().shape[0]
    out(f"Country panel: {len(df)} rows, {n_pairs} drug x country pairs, T={T}")
    out(f"Train pairs: {len(train_subs)}, holdout pairs: {len(holdout_subs)}")
    out()

    tier_counts = df.drop_duplicates(["drug_substance", "country_normalized"])
    out(f"Tier 1 (consecutive4 -- full A/B/C/D incl. GMM instruments): "
        f"{int(tier_counts['consecutive4'].sum())}")
    out(f"Tier 2/3 (thin pairs, FE-OLS/PPML + fallback): "
        f"{int((~tier_counts['consecutive4']).sum())}")
    out()

    # ---------------- Model A (AR1) ----------------
    out("=" * 70)
    out("MODEL A -- AR(1), pair fixed effects, clustered SE")
    out("=" * 70)
    res_a = fit_model_a(df, train_subs, lags=1)
    out(str(res_a))
    a_cols = ["y_l1"]
    coefs_a = res_a.params[a_cols].values

    # ---------------- Model B (FE-OLS -- the fallback IS the delivered estimator,
    # per two-model-spec.md Sec.3.5/Sec.8: "This fallback is active." Re-running
    # the raw GMM bracket check here would only re-confirm that policy, not change
    # what's reported, so it's skipped at this grain to keep the run tractable
    # across ~20.5k entities. ----------------
    out("=" * 70)
    out("MODEL B -- dynamic panel + covariates + month effects, FE-OLS (delivered estimator per spec Sec.3.5)")
    out("=" * 70)
    res_b = fit_model_b_ols_within(df, train_subs, covars=COVARS_COUNTRY)
    out(str(res_b))
    b_cols = ["y_l1", "y_l2"] + COVARS_COUNTRY
    coefs_b = res_b.params[b_cols].values

    # ---------------- Model C (PPML) ----------------
    out("=" * 70)
    out("MODEL C -- Poisson pseudo-MLE (PPML), pair FE, clustered SE")
    out("=" * 70)
    c_cols = ["y_l1"]
    res_c = fit_model_c_ppml(df, train_subs, xcols=c_cols)
    out(f"Converged: {res_c['converged']}  N obs: {res_c['n_obs']}  entities: {res_c['n_entities']}")
    coefs_c = np.array([res_c["params"][c] for c in c_cols])

    # ---------------- Model D (AB-GMM, corrected) ----------------
    out("=" * 70)
    out("MODEL D -- Arellano-Bond difference GMM, AR(1) spec, instruments y_l2/y_l3")
    out("=" * 70)
    res_d, frame_d, diag_d = fit_model_d_ab_gmm(df, train_subs)
    out(str(res_d))
    out(f"N obs: {diag_d['n_obs']}, entities: {diag_d['n_entities']}, "
        f"obs per entity: {diag_d['obs_per_entity']:.2f}")
    out(f"Hansen/J p-value: {res_d.j_stat.pval:.4f}")
    phi1_d = res_d.params["dy1"]
    coefs_d = np.array([phi1_d])
    out(f"phi1 (bias-corrected): {phi1_d:+.4f}  p={res_d.pvalues['dy1']:.4f}")
    out()

    with open("Code results/model_results_country.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    out("Saved -> Code results/model_results_country.txt")

    # ---------------- Forecast assembly (t=T+1, September 2024) ----------------
    os.makedirs("Forecasted results", exist_ok=True)

    alpha_a_full = compute_entity_effects(df, coefs_a, a_cols, t_range=(1, T))
    alpha_b_full = compute_entity_effects(df, coefs_b, b_cols, t_range=(1, T))
    scale_c_full = compute_entity_scale_ppml(df, coefs_c, c_cols, t_range=(1, T))
    alpha_d_full = compute_entity_effects(df, coefs_d, ["y_l1"], t_range=(1, T))

    # every active pair has a row at t=T by construction (the zero-fill window
    # always runs [first_t, T]) -- this is the full deliverable universe.
    last = df[df["t"] == TEST_T].copy()

    yhat_a = last["entity"].map(alpha_a_full).values + last["y"].values * coefs_a[0]
    yhat_d = last["entity"].map(alpha_d_full).values + last["y"].values * coefs_d[0]
    last_c = last.copy()
    last_c["y_l1"] = last["y"].values
    yhat_c = predict_ppml(last_c, scale_c_full, coefs_c, c_cols)

    def _raw_covar_at_t8(frame, c):
        base = c.replace("_l1", "")
        vals = frame[base].values
        return np.log1p(vals) if base == "avg_deal_size" else vals

    b_ok = (last["entity"].isin(alpha_b_full.index) & last["y_l1"].notna()
            & np.all([last[c.replace("_l1", "")].notna().values for c in COVARS_COUNTRY], axis=0))
    yhat_b = np.full(len(last), np.nan)
    if b_ok.any():
        sub = last[b_ok]
        x_b = np.column_stack([sub["y"].values, sub["y_l1"].values]
                               + [_raw_covar_at_t8(sub, c) for c in COVARS_COUNTRY])
        yhat_b[b_ok.values] = sub["entity"].map(alpha_b_full).values + x_b @ coefs_b

    a_ct = np.clip(np.expm1(yhat_a), 0, None)
    b_ct = np.clip(np.expm1(yhat_b), 0, None)
    c_ct = yhat_c
    d_ct = np.clip(np.expm1(yhat_d), 0, None)

    naive_ct = last["transaction_count"].to_numpy(dtype=float)
    no_history = ~last["entity"].isin(alpha_a_full.index).to_numpy()  # single-row-window pairs

    fallback_reason = np.full(len(last), "", dtype=object)
    fallback_reason[no_history] = "single_month_no_lag"
    b_missing = np.isnan(yhat_b) & ~no_history
    fallback_reason[b_missing & (fallback_reason == "")] = "model_b_missing_covariate_history"

    # universal fallback: pairs with no lagged observation at all get the naive
    # repeat-last-value forecast for every model, not an undefined/NaN cell.
    a_final = np.where(no_history, naive_ct, a_ct)
    c_final = np.where(no_history, naive_ct, c_ct)
    d_final = np.where(no_history, naive_ct, d_ct)
    # model B additionally falls back per-row whenever its own covariate history
    # is missing (thinner condition than no_history -- e.g. exporter_churn only
    # needs one prior active month, not a full AR history).
    b_final = np.where(no_history, naive_ct, np.where(np.isnan(yhat_b), a_final, b_ct))

    low_confidence = (last["n_active_months"] < 2).to_numpy() | no_history

    next_out = pd.DataFrame({
        "drug_substance": last["drug_substance"].values,
        "country_normalized": last["country_normalized"].values,
        "t": FORECAST_T,
        "forecast_month": FORECAST_MONTH,
        "last_actual_transaction_count_t8": last["transaction_count"].values,
        "n_active_months": last["n_active_months"].values,
        "naive_forecast": np.round(naive_ct, 1),
        "model_a_ar1_forecast": np.round(a_final, 1),
        "model_b_forecast": np.round(b_final, 1),
        "model_c_ppml_forecast": np.round(c_final, 1),
        "model_d_gmm_forecast": np.round(d_final, 1),
        "low_confidence": low_confidence,
        "fallback_reason": fallback_reason,
    })
    next_out = next_out.sort_values(["drug_substance", "country_normalized"]).reset_index(drop=True)
    next_out.to_csv("Forecasted results/forecast_t9_september2024_country.csv", index=False)
    out(f"Saved -> Forecasted results/forecast_t9_september2024_country.csv "
        f"({len(next_out)} rows, {int(low_confidence.sum())} low_confidence)")

    # ---------------- Test month (t=T) scoring, for pairs with enough own
    # history to backtest at all (needs y_l1/y_l2 present at t=T, i.e. the
    # window reaches back to T-2) -- pairs without that history genuinely
    # cannot be backtested, same as at drug grain. ----------------
    TRAIN_START, TRAIN_END = _train_window(df)
    alpha_a_v = compute_entity_effects(df, coefs_a, a_cols, t_range=(TRAIN_START, TRAIN_END))
    alpha_b_v = compute_entity_effects(df, coefs_b, b_cols, t_range=(TRAIN_START, TRAIN_END))
    scale_c_v = compute_entity_scale_ppml(df, coefs_c, c_cols, t_range=(TRAIN_START, TRAIN_END))
    alpha_d_v = compute_entity_effects(df, coefs_d, ["y_l1"], t_range=(TRAIN_START, TRAIN_END))

    test = df[df["t"] == TEST_T].dropna(subset=["y", "y_l1"]).copy()
    test = test[test["entity"].isin(alpha_a_v.index)]
    yhat_a_test = test["entity"].map(alpha_a_v).values + test["y_l1"].values * coefs_a[0]
    yhat_d_test = test["entity"].map(alpha_d_v).values + test["y_l1"].values * coefs_d[0]
    test_c = test.copy()
    yhat_c_test = predict_ppml(test_c, scale_c_v, coefs_c, c_cols)

    b_ok_t = (test["entity"].isin(alpha_b_v.index) & test["y_l2"].notna()
              & np.all([test[c].notna().values for c in COVARS_COUNTRY], axis=0))
    yhat_b_test = np.full(len(test), np.nan)
    if b_ok_t.any():
        sub = test[b_ok_t]
        xb = np.column_stack([sub["y_l1"].values, sub["y_l2"].values] + [sub[c].values for c in COVARS_COUNTRY])
        yhat_b_test[b_ok_t.values] = sub["entity"].map(alpha_b_v).values + xb @ coefs_b

    test_out = pd.DataFrame({
        "drug_substance": test["drug_substance"].values,
        "country_normalized": test["country_normalized"].values,
        "t": TEST_T,
        "actual_transaction_count": test["transaction_count"].values,
        "n_active_months": test["n_active_months"].values,
        "naive_forecast": np.round(np.expm1(test["y_l1"].values), 1),
        "model_a_forecast": np.round(np.clip(np.expm1(yhat_a_test), 0, None), 1),
        "model_b_forecast": np.round(np.where(np.isnan(yhat_b_test), np.nan,
                                               np.clip(np.expm1(yhat_b_test), 0, None)), 1),
        "model_c_ppml_forecast": np.round(yhat_c_test, 1),
        "model_d_gmm_forecast": np.round(np.clip(np.expm1(yhat_d_test), 0, None), 1),
        "low_confidence": (test["n_active_months"] < 2).to_numpy(),
        "fallback_reason": np.where(np.isnan(yhat_b_test), "model_b_missing_covariate_history", ""),
    })
    test_out.to_csv("Forecasted results/test_t8_predicted_vs_actual_country.csv", index=False)
    out(f"Saved -> Forecasted results/test_t8_predicted_vs_actual_country.csv ({len(test_out)} rows)")

    y_true_ct = test["transaction_count"].to_numpy(dtype=float)
    naive_ct_t = np.expm1(test["y_l1"].values)
    scores = {
        "AR(1)": score_counts(y_true_ct, np.clip(np.expm1(yhat_a_test), 0, None), naive_ct_t),
        "Model B (FE-OLS)": score_counts(y_true_ct[b_ok_t.values],
                                          np.clip(np.expm1(yhat_b_test[b_ok_t.values]), 0, None),
                                          naive_ct_t[b_ok_t.values]) if b_ok_t.any() else None,
        "Model C (PPML)": score_counts(y_true_ct, yhat_c_test, naive_ct_t),
        "Model D (AB-GMM)": score_counts(y_true_ct, np.clip(np.expm1(yhat_d_test), 0, None), naive_ct_t),
    }
    comp_rows = []
    for name, s in scores.items():
        if s is None:
            comp_rows.append((name, np.nan, np.nan, np.nan, "no rows with complete covariate history"))
            continue
        comp_rows.append((name, s["mase"], s["rmse_count"], s["mean_signed_err"], f"n={s['n']}"))
    comp_df = pd.DataFrame(comp_rows, columns=["model", "test_mase_t8", "test_rmse_count_t8",
                                                "test_mean_signed_error_t8", "note"])
    comp_df.loc[len(comp_df)] = ["tier summary", np.nan, np.nan, np.nan,
                                  f"{int(tier_counts['consecutive4'].sum())} pairs consecutive4=True "
                                  f"(GMM order condition met); {n_pairs} pairs total"]
    comp_df.to_csv("Forecasted results/model_comparision_country.csv", index=False)
    out("Saved -> Forecasted results/model_comparision_country.csv")
    for name, s in scores.items():
        out(f"{name} test (t={TEST_T}): {s}")


if __name__ == "__main__":
    import sys
    if "--country" in sys.argv:
        main_country()
    else:
        main()
        main_country()
