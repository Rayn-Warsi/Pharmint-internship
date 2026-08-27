# Two-Model Plan — One Page

**Target:** transaction_count per drug substance per month
**Data:** 8 months, many substances
**Horizon:** next month, maybe the one after

---

## Model A — AR(2)

Only uses the substance's own past.

```
y_dt = α_d + φ₁·y_d,t-1 + φ₂·y_d,t-2 + ε_dt
```

- `α_d` — substance's baseline level
- `φ₁, φ₂` — momentum, same for all substances
- Usable months: t = 3 to 8 → 6 rows per substance

**Estimate:** OLS with substance fixed effects, errors clustered by substance.

**Forecast:**
```
ŷ_d,9  = α_d + φ₁·y_d,8 + φ₂·y_d,7
ŷ_d,10 = α_d + φ₁·ŷ_d,9 + φ₂·y_d,8
```

---

## Model B — Dynamic panel with covariates

Adds what else was happening, plus a fix for the bias in A.

```
y_dt = α_d + λ_t + φ₁·y_d,t-1 + φ₂·y_d,t-2 + β'X_d,t-1 + ε_dt
```

**New pieces:**

| | What it adds |
|---|---|
| `λ_t` | month effects — absorbs shocks hitting everything at once |
| `X_d,t-1` | lagged covariates (below) |
| GMM estimation | corrects the downward bias in φ₁ |

**Covariates (all lagged):**
- `ln(p50_unit_price)`
- `total_quantity / transaction_count` — avg deal size
- **Exporter churn** — how many of last month's top-5 exporters were there the month before
- **Destination count** — one market vs five
- `(p75−p25)/p50` — price spread
- Months since first appearance

**Estimate:** Arellano–Bond / system GMM. Collapse instruments, cap lag depth at 2–3.

**Check:** AR(2) test on differenced residuals must be insignificant. Hansen p-value near 1.00 is a red flag, not a pass.

---

## The difference

| | A | B |
|---|---|---|
| Inputs | own past only | own past + market conditions |
| Estimation | OLS | GMM |
| Month effects | no | yes |
| Answers | what's next | what's next + what drives it |
| Bias in φ₁ | ~10–15% low | corrected |
| Risk | too simple | weak instruments |

**A is the benchmark B has to beat.** If B doesn't win out of sample, the covariates aren't earning their place.

---

## Fixes for both

**Negative forecasts.** AR is unbounded below. Either floor at zero, or model `ln(1+y)` and convert back with `exp(ŷ)−1`.

**Stationarity.** Check `φ₁+φ₂ < 1`, `φ₂−φ₁ < 1`, `|φ₂| < 1`. Violated → h=2 forecasts explode.

---

## Before modelling — 3 checks

1. **Zero rows exist?** If a substance skips a month, is there a row with count = 0 or no row? If missing, build the full grid.
2. **Price columns.** Use `p50_unit_price` only. Never `avg`, `min`, `max` — mixed unit codes (kg vs bottles) corrupted them.
3. **Zero share.** Under 5% with median count above 20 → no special handling.

---

## Scoring

Beat both or don't deploy:
- **Naïve** — next month = this month
- **Drift** — naïve + average trend

Metric: **MASE**. Below 1 beats naïve. Score h=1 and h=2 separately.

No MAPE — breaks at zero, biased toward under-forecasting.

---

## The one trap

**Tenders.** Annual procurement means 8 months can contain one big award. Trend models extrapolate it forever.

Strong growth? Check if it's one month. Usually it is.

---

## Order

1. Fix data
2. Model A
3. Build covariates
4. Model B
5. Compare on held-out months
