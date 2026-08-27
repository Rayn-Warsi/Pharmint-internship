# Model Specification — Models A and B

**Panel:** 792 drug substances × 8 months (Jan–Aug 2024), 6,012 rows · **Updated:** 2026-07-28

Model A is a pooled autoregression with substance fixed effects. Model B adds
market-condition covariates and month effects, estimated by Arellano–Bond
difference GMM to correct the bias a lagged dependent variable induces in a short
panel. A is the benchmark; B must beat it out of sample or the complexity isn't
justified.

---

## 1. Common setup

`d` indexes substance (792), `t` indexes month (`1…8`), `T = 8`.

### 1.1 Dependent variable

```
y_dt = log(1 + transaction_count_dt)
```

Three reasons for the transform: counts are heavily right-skewed (median 6, max
1663); a linear AR model is unbounded below and would forecast negative for small
substances; and log(1+y) is defined at zero, which matters because 15.6% of rows
are dormant months. Forecasts are back-transformed and clipped at zero.

### 1.2 Covariates

All enter **lagged one month**. September's price is unknown when forecasting
September, and within a month price and volume are simultaneously determined, so
contemporaneous entry would be endogenous.

| Variable | Definition | Role |
|---|---|---|
| Price level | `log(p50_unit_price)` | median not mean — unit prices mix quantity units |
| Average deal size | `log(1 + quantity/count)` | separates "more deals" from "bigger deals" |
| Exporter churn | `\|top5_t ∩ top5_{t-1}\| / 5` | supplier-base stability; falling overlap = lane going quiet |
| Destination count | distinct destination countries | one market vs five behaves differently |
| Price spread | `(p75 − p25) / p50` | product-mix heterogeneity |

**Months-since-first is excluded by construction.** With entity *and* month effects
it equals `t − first_t − 1` exactly — a deterministic function of the two effects,
and a constant after differencing. Perfectly collinear and fully absorbed;
including it is a rank-deficiency error, not extra information.

### 1.3 Sampling

**Time split:** train `t = 3…6`, validate `t = 7`, test `t = 8`. Never random rows —
lag features would leak the target across the boundary.

**Substance split:** 80/20 → 634 train / 158 holdout. Costs no time periods; tests
generalization against memorization.

**Inference:** standard errors clustered by substance — arbitrary within-substance
serial correlation, heteroskedasticity across substances.

---

## 2. Model A — AR(p) with substance fixed effects

```
y_dt = α_d + φ₁·y_{d,t-1} + φ₂·y_{d,t-2} + ε_dt
```

AR(1) restricts `φ₂ = 0`. Both orders estimated and compared out of sample.
Estimator is within (fixed-effects) OLS with clustered errors, **no time effects** —
the structural difference from Model B.

**`α_d`** absorbs each substance's operating level: a generic at 200 shipments/month
and a niche API at 3/month need separating, not explaining. Since `α_d` is
unrestricted, all identification of `φ` comes from *within-substance* variation.
The within transform subtracts substance means:

```
(y_dt − ȳ_d) = φ₁·(y_{d,t-1} − ȳ_{d,-1}) + φ₂·(y_{d,t-2} − ȳ_{d,-2}) + (ε_dt − ε̄_d)
```

**`φ` is pooled.** The central design choice: no substance has enough history to
estimate its own autoregressive parameter from 4–6 rows, but several hundred pooled
do. `φ` is constrained equal across substances; only the level is substance-specific.

**`φ₂` is substantively informative** — negative means a large month pulls orders
forward from the next. Real behaviour in lumpy tender-driven demand, not a nuisance
parameter.

### 2.1 Why not deeper lags

| Lags | Usable target months | Rows per substance |
|---|---|---|
| 1 | 2–8 | 7 |
| 2 | 3–8 | 6 |
| 7 | 8 only | 1 |

AR(7) leaves one row per substance and no month to validate on. Lags 1–2 capture
momentum; lag 12 would capture seasonality but is unreachable at T=8; lag 7 is
noise from seven months ago.

### 2.2 Known bias

The within transform makes `y_{d,t-1}` mechanically correlated with `(ε_dt − ε̄_d)`,
because `ε̄_d` contains `ε_{d,t-1}`, which determines `y_{d,t-1}`. This is **Nickell
bias** — downward, order `−1/T`, so roughly 10–15% on `φ₁` at T=8. Model A does not
correct it. Tolerable for forecasting, not for reporting persistence as a finding —
which motivates Model B.

---

## 3. Model B — dynamic panel, covariates, month effects, GMM

```
y_dt = α_d + λ_t + φ₁·y_{d,t-1} + φ₂·y_{d,t-2} + β'X_{d,t-1} + ε_dt
```

**`λ_t` — month effects.** Absorbs anything hitting all substances simultaneously:
port disruption, holiday period, rupee move. Without them a market-wide dip is
misattributed to each substance as an idiosyncratic shock. Cost: 7 parameters.

**`β'X` — covariates** (§1.2). Churn is the one most likely to carry information the
lags cannot see — the top-5 exporter list looks purely descriptive, but differencing
it month to month turns it into a leading indicator of trade relationships forming
and dissolving.

### 3.1 The identification problem

With a lagged dependent variable and fixed effects, OLS is biased either way:

- **Pooled OLS** ignores `α_d`; the omitted effect correlates positively with
  `y_{d,t-1}`, biasing `φ₁` **upward**.
- **FE-OLS** eliminates `α_d` but induces the §2.2 correlation, biasing `φ₁`
  **downward**.

A consistent estimate must lie between the two — used as a diagnostic (§3.3).

### 3.2 Arellano–Bond difference GMM

**First-difference** to eliminate `α_d`:

```
Δy_dt = φ₁·Δy_{d,t-1} + φ₂·Δy_{d,t-2} + β'ΔX_{d,t-1} + Δλ_t + Δε_dt
```

This removes the fixed effect without inducing the within-transform correlation. It
does introduce MA(1) structure in `Δε_dt` by construction — which is why the AR(2)
residual test is the meaningful one, not AR(1).

**Instrument the differenced lags.** `Δy_{d,t-1}` correlates with `Δε_dt` (both
contain `ε_{d,t-1}`). Under serially uncorrelated `ε_dt`, levels dated `t−2` and
earlier are valid instruments:

```
E[ y_{d,t-s} · Δε_dt ] = 0     for s ≥ 2
```

Levels `y_{t-3}` and `y_{t-4}` are used, capping lag depth at 2. Two endogenous
regressors require at least two excluded instruments for the order condition.
Differenced covariates and month dummies are treated as exogenous.

**Instrument discipline, non-negotiable:** collapse the instrument matrix (the full
Arellano–Bond matrix grows quadratically in T); cap lag depth at 2–3; report
instrument count against N — proliferation overfits the endogenous regressors and
mechanically drives the Hansen p-value toward 1.

### 3.3 The T=8 constraint

The binding limitation. The order condition requires both `Δy_{t-1}` and `Δy_{t-2}`
instrumented, which requires `y_{t-3}` and `y_{t-4}` to exist. With differencing the
usable window collapses to `t = 4…6`; after the non-null requirement on all five
covariates, **one usable target period per substance** remains.

Consequences, structural rather than incidental:

- **Exactly identified** — instrument count equals endogenous parameter count, zero
  overidentifying restrictions
- **Hansen/J has zero degrees of freedom**, undefined
- **AR(1)/AR(2) residual tests need within-substance variation across periods**;
  with one row per substance there is none, so both are undefined
- Standard errors very large, point estimate unstable

### 3.4 Diagnostics

**Bracket check.** GMM `φ₁` must satisfy

```
min(φ₁_FE-OLS, φ₁_pooled) ≤ φ₁_GMM ≤ max(φ₁_FE-OLS, φ₁_pooled)
```

Outside that interval it is not a bias correction of anything. Both bracketing
models are fitted solely to construct this test — pooled OLS has no other role.

**Hansen / J.** A p-value near 1.00 is a **red flag, not a pass** — it indicates too
many instruments relative to N.

**Residual serial correlation.** On differenced residuals: AR(1) significant
(mechanical, from differencing), **AR(2) insignificant**. Significant AR(2) means
the error was already serially correlated in levels, invalidating the moment
conditions.

**Stationarity** (both models): `φ₁ + φ₂ < 1`, `φ₂ − φ₁ < 1`, `|φ₂| < 1`. Violation
implies forecasts diverge at horizon 2 and beyond.

### 3.5 Fallback rule

If the bracket check or stationarity fails, Model B is **reported via FE-OLS with
the ~10–15% downward Nickell bias flagged**, not via a fragile GMM point estimate. A
known, signed, bounded bias is more defensible than an unstable estimate from weak
instruments.

**This fallback is active.** Every reported Model B quantity — coefficients, scores,
t=8 predictions, t=9 forecast — comes from the FE-OLS fit. The GMM step produces
only the bracket-check and residual diagnostics.

> **Model B as delivered is an FE-OLS (within) estimator with entity and month
> effects. It is not a GMM estimate.**

---

## 4. Forecast construction

Shared by both models, and deliberately **not** the fitted fixed effects. The
substance level is a plug-in from each substance's own history:

```
α_d = mean over the substance's available months of ( y_dt − x_dt'β̂ )
```

This matters for validity: because `α_d` comes from the substance's own data rather
than the estimation sample, substances held out of the `φ` estimation still get a
fair forecast — only momentum is pooled. It also mirrors production forecasting of a
new substance: pooled momentum, own-history level.

For the **t = 9 forecast** (September 2024), `α_d` is refit over `t = 1…8` and inputs
roll forward one period — `y_{t-1}` takes `t=8`'s value, `y_{t-2}` takes `t=7`'s, and
covariates take their `t=8` values under the same transforms used in training.

---

## 5. Evaluation protocol

**Benchmarks — both must be beaten:** naïve (this month repeats) and drift (naïve
plus average trend). Naïve is genuinely hard to beat on count data; confirmed, not
assumed.

**Primary metric — MASE**, mean absolute error divided by naïve MAE. Below 1.0 beats
doing nothing. The denominator is built from ~7 numbers and is noisy: 0.85 vs 0.92
is not a real gap.

**Secondary — RMSE on both scales:** log (what the model fits) and count (what
decisions use). These can disagree, and the disagreement is informative about
retransformation bias.

**Reported separately:** by substance-size decile, and train-substances against
holdout-substances.

---

## 6. Assumptions and limitations

**Assumed.** `ε_dt` serially uncorrelated in levels (required for the GMM moment
conditions); covariates predetermined at `t−1`; `φ` homogeneous across substances;
symmetric forecast loss.

**Structural limits.** No seasonality is estimable at T=8 — the horizon ceiling is
two months. `φ` rests on 4–6 usable rows per substance, so intervals should be
reported rather than point estimates. A single large procurement award inside the
window reads as trend and gets extrapolated, so any strong-growth substance needs
checking for whether the growth is one month.

**More data switches features on rather than changing the architecture.** At 12–18
months seasonality becomes visible and GMM stable; at 24 months, real trend
components and longer horizons.

---

## 7. Model C — Poisson pseudo-MLE (PPML) with substance fixed effects

*Added 2026-07-29.*

```
E[ transaction_count_dt | · ] = exp( α_d + φ₁·y_{d,t-1} )
```

Same right-hand side as Model A AR(1). Only the estimator changes, so the
comparison isolates the estimator rather than the specification.

### 7.1 What it fixes

**Retransformation bias.** A and B fit `log1p(count)` and invert with `expm1`. That
round trip does not recover `E[y|x]` under heteroskedasticity — by Jensen it
recovers something nearer the conditional geometric mean, so forecasts are biased
**down**. Model A's t=8 mean signed error of **−2.09** per substance, and a total
**2.68% below actual**, is that bias in the shipped numbers, not sampling noise.

**The zeros.** The `+1` in `log1p` is a free tuning constant. It touches the 15.6%
of rows that are zero and the 35.5% that are ≤ 2 — change it to `+0.5` and the
coefficients move. PPML needs no offset.

**Scale.** Log-OLS fits the geometric mean; decisions use levels.

Consistent under the conditional-mean assumption alone even when the data is not
Poisson (Santos Silva & Tenreyro, *The Log of Gravity*).

### 7.2 Estimation

`pyfixest` is unavailable in this environment, so the 792 fixed effects are absorbed
analytically rather than as dummies. The FE score has a closed form:

```
exp(α_d) = Y_d / S_d(b),     Y_d = Σ_t y_dt,     S_d = Σ_t exp(x_dt'b)
```

Substituting back leaves a concentrated log-likelihood in `b` alone:

```
l(b) = Σ_dt y_dt·x_dt'b − Σ_d Y_d·log S_d(b)   (+ const)
```

One parameter to optimize, no dense dummy matrix, exact rather than approximate.
Substances with `Y_d = 0` contribute nothing and drop out on their own. Standard
errors use the Wooldridge FE-Poisson clustered sandwich: `μ`-weighted
within-transformed regressors in the bread, scores summed per substance in the meat.

**Forecast construction** is the multiplicative analogue of §4, and is the exact
FE-Poisson plug-in:

```
scale_d = Σ_t y_dt / Σ_t exp(x_dt'b)        forecast = scale_d · exp(x'b)
```

Always finite and non-negative — a dormant substance gets `scale_d = 0` and
forecasts 0, with no `log(0)` and no clipping.

### 7.3 Caveat

The result that FE Poisson has **no incidental-parameters problem at fixed T**
assumes *strictly exogenous* regressors. `y_{d,t-1}` is not strictly exogenous, so
**Model C carries the same Nickell bias as A and B** (`φ₁ = −0.2930`, close to A's
−0.2479). PPML fixes scale and zeros; it does not fix the dynamic-panel bias — that
is Model D's job. Proper count-panel treatments would be Wooldridge (1997)
quasi-differencing or the Blundell–Griffith–Windmeijer linear feedback model;
neither is implemented.

`φ₁` is a Poisson elasticity under a log link and is **not** comparable in level to
the OLS `φ₁` in A/B/D.

---

## 8. Model D — Arellano–Bond difference GMM, corrected

*Added 2026-07-29.*

```
Δy_dt = φ₁·Δy_{d,t-1} + Δε_dt
```

Model B's GMM (§3.2) with two implementation errors fixed. **This supersedes §3.3's
conclusion that T=8 is the binding constraint** — the constraint was lag depth, not
panel length.

### 8.1 The two corrections

**1. `Δy_{t-2}` is not endogenous.** The error is `ε_t − ε_{t-1}`; `Δy_{t-2}` carries
`ε_{t-2}, ε_{t-3}`. No overlap — it is predetermined. §3.2 treats it as endogenous,
which doubles the order condition for nothing.

**2. Lag depth is one level too deep.** The shallowest valid Arellano–Bond
instrument is `y_{t-2}`, not `y_{t-3}`. Each extra level costs a period of history
per substance — the direct cause of §3.3's one-row-per-substance collapse.

Depth also decides whether the instrument retains any **relevance** at `φ ≈ 0`,
where `y_t = α_d + ε_t`:

```
Cov( y_{t-2}, Δy_{t-1} ) = Cov( ε_{t-2}, ε_{t-1} − ε_{t-2} ) = −Var(ε)  ≠ 0
Cov( y_{t-3}, Δy_{t-1} ) = Cov( ε_{t-3}, ε_{t-1} − ε_{t-2} ) =  0
```

`y_{t-2}` stays relevant exactly where `y_{t-3}` dies. §3.3's `φ₁ = 16.25` is a
zero-first-stage blowup: IV is a ratio, and a near-zero denominator produces a large
finite number rather than an error. This is the *opposite* of the textbook
Blundell–Bond failure, which occurs as `φ → 1`. Both ends of the range break
difference GMM; this panel sits at the bottom end.

**Specification as run:** AR(1), one endogenous regressor, instrumented at `y_{t-2}`
**and** `y_{t-3}`. The second instrument exists solely to make the model
overidentified by one, so Hansen J has a degree of freedom and is computable — the
trap §3.3 fell into. Estimation window `t = 3…6`, unchanged, so `t=7` stays clean for
validation and `t=8` for test. Forecasts use the §4 plug-in `α_d`.

### 8.2 Results

```
φ₁ = −0.0766    SE 0.0653    p = 0.241      <- INSIGNIFICANT
First-stage F on excluded instruments  170.00
Hansen J                        p = 0.206
N obs 1069     entities 570    obs/entity 1.88   (§3.3 got 1.00)
Bracket [−0.2492, +0.4720]      PASS             (§3.3 got 16.25, fail)
```

Every diagnostic §3.3 could not compute now runs. Instruments are strong and not
rejected.

**`φ₁` is statistically indistinguishable from zero**, and this is corroborated
independently by window sensitivity — refitting Model A on longer windows drives
`φ₁` monotonically toward zero at the 1/T rate:

| window | T | `φ₁` (FE-OLS) |
|---|---|---|
| `t = 3…6` | 4 | −0.2479 |
| `t = 3…7` | 5 | −0.1801 |
| `t = 3…8` | 6 | −0.1332 |
| `t = 2…8` | 7 | −0.1078 |

**Central empirical finding: transaction counts have no exploitable month-to-month
momentum.** They wander around a substance-specific level. Model A's `−0.2479` is
roughly 69% Nickell bias. That explains why naïve persistence is hard to beat and
why the log-OLS models cluster near MASE 1.0.

### 8.3 Two corrections to §2.2

**Magnitude.** §2.2 states the bias is "roughly 10–15% on `φ₁` at T=8". Measured
against Model D it is **−0.171 on −0.2479, about 69%**. The estimation window is
`t = 3…6`, so the binding **T is 4, not 8** — which is why it is this large.

**Consequence.** §2.2 states the bias is "tolerable for forecasting". **This is
false.** `α_d` absorbs the level but not the slope, so an over-negative `φ₁`
over-mean-reverts on every deviation from a substance's own average. Correcting it
improves test MASE from 0.975 to 0.895 — the largest single accuracy gain in the
project. The bias damages the forecast, not only the reported persistence.

### 8.4 Limitation

The **AR(2) residual test still returns `nan`** — it needs 3+ observations per
entity and there are 1.88. That test is what validates `y_{t-2}` as an instrument
(§3.4), since it requires `ε` serially uncorrelated in levels. Hansen J passing at
p = 0.206 is supporting evidence but is **not the same check**. Report Model D as
"instrument validity partially verified", not as fully diagnosed Arellano–Bond.

---

## 9. Results — A, B, C, D

*Added 2026-07-29.* All figures `t = 8` test month unless stated. Full table:
`Forecasted results/model_comparision.csv`.

| metric | A: AR(1) | AR(2) | B | C: PPML | **D: AB-GMM** |
|---|---|---|---|---|---|
| `φ₁` | −0.2479 | −0.2990 | −0.2492 | −0.2930 | **−0.0766** |
| `φ₁` p-value | 0.000 | 0.000 | 0.000 | 0.000 | **0.241** |
| implied Nickell bias vs D | −0.171 | −0.222 | −0.173 | — | 0 |
| validation MASE (train) | 0.858 | 0.904 | 0.905 | 0.876 | **0.826** |
| validation MASE (holdout) | 0.911 | 0.940 | 0.917 | 0.930 | **0.881** |
| **test MASE** | 0.975 | 1.048 | 1.036 | 1.008 | **0.895** |
| test RMSE (count) | 20.53 | 22.68 | 27.08 | 21.15 | **18.62** |
| test mean signed error | −2.09 | −1.81 | −1.66 | **−1.20** | −1.63 |
| test total error % | −2.68 | −2.31 | −2.13 | **−1.54** | −2.09 |
| Sept 2024 total forecast | 41016 | 40788 | 40546 | 41613 | 41000 |

Naïve benchmark MAE 8.730; drift 19.553.

**Reading:**

- **D is the best forecaster.** Wins validation-train, validation-holdout and test —
  consistent across all three, not a one-month artefact. Only model materially below
  MASE 1.0.
- **C is the best aggregator.** Halves A's total bias. It *loses* on MASE because MAE
  rewards the conditional **median**; on right-skewed counts log-OLS's downward bias
  accidentally aims there, while PPML correctly targets the mean. The two optimise
  different targets — use C for totals, D for per-substance ranking.
- **B fails.** MASE 1.036, beaten by naïve persistence. The covariates and second lag
  add noise, not signal.
- **A, B and C all carry Nickell bias; only D corrects it.** C's nonlinearity does
  not exempt it (§7.3).
- The **naïve benchmark has no estimation bias at all** and scores MAE 8.730 against
  A's 8.513. A method that estimates nothing nearly ties three estimated models —
  the clearest single indication that signal, not estimator quality, is the ceiling.

**Signed error and total are reported because §5's metrics cannot see them.** MASE
and RMSE are symmetric in sign and therefore blind to the §7.1 bias; without these
two rows the entire motivation for Model C is invisible in the comparison.

### 9.1 Ruled out

**System GMM (Blundell–Bond)** is not a remedy here: it requires mean stationarity
of initial conditions, and **20.7% of substances enter after `t = 1`**, so entry
timing is correlated with the fixed effect by construction.

**Note on §6's "more data ... GMM stable".** GMM stability was **not** a sample-size
problem — it was fixed by correcting lag depth (§8.1) at unchanged T. Longer T would
additionally shrink Nickell bias on its own, at which point FE-OLS and GMM converge
and Model D's advantage narrows.

---

## 10. Drug × country grain

*Added 2026-08-10.* All four models (A/B/C/D) also run at
`drug_substance × country_normalized × month` grain — entity = the pair, not just
the substance. Implementation: `panel_build.build_panel_country()` →
`Code results/panel_dataset_country.csv`; `model_ab.main_country()` →
`Forecasted results/forecast_t9_september2024_country.csv` (the deliverable) and
`test_t8_predicted_vs_actual_country.csv`. Run via `python model_ab.py --country`
(or with no flag, which runs both grains).

**Fitting stays pooled**, unchanged in kind from §1–§9: `φ`/`β`/month effects are
estimated once across all pairs, not per pair. Only `α_d` (Model A/B/D's fixed
effect) and `scale_d` (Model C's plug-in) are pair-specific, computed from each
pair's own history exactly per §4's plug-in construction. Pooling now draws on
~67k active drug×country×month observations rather than ~6k drug×month ones — more
data for `φ`, not less; country-grain Model D's `φ₁ = +0.0084` (p = 0.65, Hansen J
p = 0.76) reinforces §8.2's central finding that transaction counts carry no
exploitable pooled month-to-month momentum, now on a wider sample.

**Coverage.** 793 drugs × 210 countries, but only 20,562 pairs are ever active (a
full cross product would be ~88% fabricated zeros and was not built — each pair's
panel window is its own `[first_t, 8]`, same trim rule as the drug-level panel).

**The real sparsity effect is per-row, not per-model.** The original concern going
in was that AR/GMM lag depth would make thin pairs unmodelable. In practice every
active pair's window always reaches `t = 8` by construction, so Models A/C/D's
forecast (`α_d` plug-in + pooled `φ` × own `t=8` value) is computable for all but
genuinely single-month pairs. The actual failure mode is narrower and was not
anticipated: **Model B's `exporter_churn` covariate is undefined whenever the
current month has zero transactions** (no exporter list that month → no overlap to
compute), independent of how much history the pair has otherwise. At country grain
this hits the large majority of pairs (most go quiet in some month); at drug grain
it already silently dropped 228/792 substances from the existing `forecast_t9_*.csv`
before this change — that pre-existing gap was invisible until building the
country-grain fallback surfaced it.

**Fallback, applied only where genuinely needed, always flagged:**

| condition | rows (of 20,562) | behavior |
|---|---|---|
| full history, all covariates defined at t=8 | 4,276 | A/B/C/D all independently estimated |
| `exporter_churn_l1` undefined at t=8 (dormant that month) | 15,281 | A/C/D estimated normally; Model B forecast = Model A's value, `fallback_reason="model_b_missing_covariate_history"` |
| single active month, no lagged observation exists at all | 1,005 | all four models forecast = naive repeat-last-value, `low_confidence=True`, `fallback_reason="single_month_no_lag"` |

`low_confidence` (`n_active_months < 2`) and `fallback_reason` are written on every
row of both country-grain output files — degraded rows are visible, never blended
silently with well-estimated ones.

**Not recomputed at this grain:** Model B's raw GMM bracket-check/Hansen
diagnostics (§3.2) — skipped because Model B's *delivered* estimate has been the
FE-OLS fallback since §3.5 was adopted ("This fallback is active"), so re-running
the raw GMM here would only re-confirm existing policy at the cost of fitting IVGMM
across ~20k more entities. Model D's GMM *is* re-run in full, since correcting the
Nickell bias is Model D's entire purpose.
