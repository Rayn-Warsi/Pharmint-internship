# Two-Model Framework — Working Spec

Companion to the one-page sheet.

---

## The design

**Model A** is univariate — a substance's count explained by its own past, nothing else.

**Model B** is multivariate and dynamic — same autoregressive core, plus market conditions, month effects, and a proper fix for the bias that a lagged dependent variable creates in a short panel.

A is the benchmark. B has to beat it out of sample or the extra complexity isn't justified.

---

## Model A

```
y_dt = α_d + φ₁·y_d,t-1 + φ₂·y_d,t-2 + ε_dt
```

**α_d** — substance fixed effect. Each substance's normal size. Paracetamol at 200/month and a niche API at 3/month don't need explaining, just separating.

**φ₁, φ₂** — momentum, pooled across all substances. This is the key move: no single substance has enough history to estimate momentum from 6 rows, but thousands of substances pooled together do. Expect φ₁ around 0.5–0.9.

**φ₂ is informative.** Negative means a big month pulls orders forward from the next — real ordering behaviour.

**Estimate:** OLS with substance fixed effects (within estimator), errors clustered by substance.

### Why not AR(7)

AR(7) needs seven prior months. Only month 8 has those, so each substance gives one row and no month is left to validate on.

| Lags | Usable target months | Rows per substance |
|---|---|---|
| 1 | 2–8 | 7 |
| 2 | 3–8 | 6 |
| 7 | 8 only | 1 |

Lag 7 captures nothing anyway. Lags 1–2 capture momentum; lag 12 would capture seasonality (unreachable at 8 months). Lag 7 is noise from seven months ago.

Test AR(1) against AR(2) and keep the winner.

---

## Model B

```
y_dt = α_d + λ_t + φ₁·y_d,t-1 + φ₂·y_d,t-2 + β'X_d,t-1 + ε_dt
```

Three additions over A, each doing separate work.

### 1. Month effects (λ_t)

Absorbs anything hitting all substances simultaneously — a port disruption, a holiday period, a rupee move. Without them, a market-wide dip gets misattributed to each substance individually.

Cost: 7 parameters. Cheap given your row count.

### 2. Covariates (X)

All lagged — you won't know September's price when forecasting September, and within a month price and volume are determined together.

| Variable | Why |
|---|---|
| `ln(p50_unit_price)` | price level, robust to your contaminated columns |
| `total_quantity / transaction_count` | average deal size — separates "more deals" from "bigger deals" |
| **Exporter churn** | overlap between consecutive top-5 exporter lists. Falling overlap = lane going quiet |
| **Destination count** | one market vs five behaves very differently |
| `(p75−p25)/p50` | price spread — proxy for product-mix heterogeneity |
| Months since first appearance | lifecycle position |

The churn measures are the ones most likely to add something A can't see. `top5_exporters` and `top5_destinations` look descriptive, but comparing them month to month turns them into leading indicators of relationships forming and dissolving.

### 3. GMM estimation

This is the real complexity step, and the reason B is a different animal rather than A with extras.

**The problem.** When you have a lagged dependent variable *and* substance fixed effects, demeaning within each substance makes the lagged term correlate with the demeaned error. This is **Nickell bias**, it's downward, and it's roughly −1/T. At T=8 that's 10–15% on φ₁.

For pure forecasting this is tolerable. If you want to report persistence as a finding — and for the academic side you probably do — it needs correcting.

**The fix.** Arellano–Bond (difference GMM) first-differences the equation to remove α_d, then uses deeper lags of y as instruments for the differenced lag. Blundell–Bond (system GMM) adds a levels equation instrumented by lagged differences — better when the series is persistent, which yours likely is.

**Discipline, non-negotiable:**
- Collapse the instrument matrix
- Cap lag depth at 2–3
- **Report instrument count against N.** Instrument proliferation overfits and makes the Hansen test meaningless

**Diagnostics:**
- **AR(1) significant, AR(2) insignificant** on differenced residuals. AR(2) significant means your lag structure is wrong.
- **Hansen test.** A p-value near 1.00 is a red flag, not a pass — it means too many instruments.
- **The bracket check.** Your GMM φ₁ should sit between the OLS-with-fixed-effects estimate (biased down) and pooled OLS (biased up). Outside that range, something is wrong.

**Honest caveat:** T=8 leaves 6–7 periods after differencing. Enough lag depth to work, not much margin. If instruments come out weak, fall back to OLS with fixed effects and simply note the bias — that's more defensible than a fragile GMM result.

---

## The comparison

| | A | B |
|---|---|---|
| Inputs | own past only | own past + market conditions |
| Month effects | no | yes |
| Estimation | OLS (within) | system GMM |
| φ₁ bias | ~10–15% low | corrected |
| Answers | what's next | what's next + what drives it |
| Failure mode | too simple | weak instruments |

**A wins if:** transaction counts are mostly momentum and the covariates are noise. Entirely possible — it happens often with count data.

**B wins if:** price, churn, or destination structure carry signal beyond what the lags already contain.

**Either result is worth reporting.** "The simple model wins" is a finding, not a failure — and a defensible one for the research write-up.

---

## Fixes that apply to both

**Negative forecasts.** AR is linear and unbounded below. A small substance with a low recent count can forecast below zero. Either floor at zero, or model `ln(1+y)` and convert back with `exp(ŷ)−1`. The log version is better behaved on skewed counts generally.

**Stationarity.** Check after estimating:
```
φ₁ + φ₂ < 1
φ₂ − φ₁ < 1
|φ₂| < 1
```
Violated → forecasts diverge at h=2 and beyond.

**Residual plot.** Residuals against fitted values should be a formless cloud. A funnel (spread growing with level) means variance scales with size — switch to the `ln(1+y)` version.

---

## Data prep

### 1. Zero rows

Does a dormant month appear as a row with count = 0, or not at all?

If missing, you're training only on active months — the model never sees a dormant month, so it can't predict one. Build the full substance × month grid.

Three cases:
- **True zero** — could have shipped, didn't → keep as 0
- **Not yet launched** — first appears month 5 → drop months 1–4
- **Missing data** → investigate

### 2. Price columns

Your odd numbers are almost certainly **mixed quantity units** — kg, grams, packs, bottles summed together. A price-per-kg and a price-per-bottle averaged gives nonsense.

Confirm: `min` near zero with huge `max`, `stddev` > `avg`, `avg` far from `p50`.

| Use | Skip |
|---|---|
| `p50_unit_price` | `avg`, `min`, `max` |
| `(p75−p25)/p50` for spread | `stddev` |
| `max/p50` as a quality flag | — |

Second cause: **API vs finished dosage** under one substance code — price scales differ by 100×. Check `top5_product_descriptions`.

Control for `fob_estimated_pct` throughout.

### 3. Zero share

Under 5% with median count above 20 → no special handling needed. If zeros are common, a zero month drags the next forecast down mechanically — that's when `ln(1+y)` helps.

---

## Validation

**Time splits only.** Train months 3–6, validate 7, test 8. Never split rows randomly — the lag features leak the answer and produce great scores that mean nothing.

**Also hold out substances.** Train on 80% of substances, test on the rest. Costs no time periods, and tells you whether the model generalises or memorises.

**Benchmarks — beat both or don't deploy:**
- Naïve (this month repeats)
- Drift (naïve + average trend)

Naïve is genuinely hard to beat. Confirm rather than assume.

**MASE example.** Substance runs 40, 44, 39, 47, 45, 50, 46, 52. Monthly changes: 4, 5, 8, 2, 5, 4, 6 → average 4.86. That's the denominator. Forecast 49, actual 54 → error 5 → MASE = 1.03. Slightly worse than doing nothing.

Denominator is 7 numbers, so it's noisy. 0.85 vs 0.92 is not a real gap.

**Report separately:** h=1 vs h=2, and by substance size decile.

---

## Known limits

**Two-month horizon, hard.** No seasonality possible at 8 months. Set this expectation before anyone asks for an annual number.

**Tender trap.** One large procurement award in your window gets read as trend and extrapolated forever. Any strong-growth substance — check whether it's a single month.

**φ is imprecise.** 6 rows per substance. Pooling helps, but report intervals.

**GMM is fragile at T=8.** If diagnostics fail, retreat to OLS and note the bias rather than reporting a shaky result.

**Symmetric loss assumed.** If a stockout costs more than excess stock, decide that before picking a model.

---

## What changes with more data

- **12–18 months:** seasonality becomes visible; GMM gets more stable
- **24 months:** real trend components, longer horizons

Architecture doesn't change. You'll switch things on, not rebuild.
