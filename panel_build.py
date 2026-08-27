"""
Build the substance x month panel used by the two-model spec (two-model-spec.md),
straight from the raw transaction file -- does not trust Code results/product_analytics.csv,
which was round-tripped through Excel at some point and now has corrupted numeric
columns (semicolon-delimited, comma-decimal, scientific notation, broken quantile
ordering: p50 > p99 in places).

Output: Code results/panel_dataset.csv, one row per (drug_substance, year, month),
full substance x month grid (dormant months kept as 0, pre-launch months dropped).
"""

import os
import time
import numpy as np
import pandas as pd

import Code as etl

OUT_PATH = os.path.join("Code results", "panel_dataset.csv")
OUT_PATH_COUNTRY = os.path.join("Code results", "panel_dataset_country.csv")

def _month_order(df: pd.DataFrame) -> list:
    """Distinct (year, month) pairs present in the enriched data, sorted
    chronologically. Was hardcoded to Jan-Aug 2024 (T=8); now derived so it
    naturally covers whatever span the source file spans (T=20 for the
    Jan 2023 - Aug 2024 merged dataset)."""
    pairs = df[["year", "month"]].drop_duplicates()
    return sorted(map(tuple, pairs.itertuples(index=False, name=None)))


def _top5_exporter_set(s: pd.Series) -> frozenset:
    return frozenset(s.value_counts().head(5).index)


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["drug_substance"] != "UNCLASSIFIED"].copy()
    month_order = _month_order(df)
    month_index = {ym: i + 1 for i, ym in enumerate(month_order)}
    T = len(month_order)
    df["t"] = df.apply(lambda r: month_index[(r["year"], r["month"])], axis=1)

    g = df.groupby(["drug_substance", "t"])
    base = g.agg(
        transaction_count=("drug_substance", "size"),
        total_quantity=("Quantity", "sum"),
        p25_unit_price=("Unit Price in Foreign Currency", lambda s: s.quantile(0.25)),
        p50_unit_price=("Unit Price in Foreign Currency", lambda s: s.quantile(0.50)),
        p75_unit_price=("Unit Price in Foreign Currency", lambda s: s.quantile(0.75)),
        destination_count=("country_normalized", "nunique"),
        fob_estimated_pct=("fob_estimated", "mean"),
    ).reset_index()

    top5 = g["Exporter Name"].apply(_top5_exporter_set).reset_index(name="top5_exporters")
    base = base.merge(top5, on=["drug_substance", "t"])

    # full substance x month grid: keep only months from first appearance onward
    substances = base["drug_substance"].unique()
    full_index = pd.MultiIndex.from_product([substances, range(1, T + 1)], names=["drug_substance", "t"])
    full = pd.DataFrame(index=full_index).reset_index()
    panel = full.merge(base, on=["drug_substance", "t"], how="left")

    first_t = panel.dropna(subset=["transaction_count"]).groupby("drug_substance")["t"].min()
    panel["first_t"] = panel["drug_substance"].map(first_t)
    panel = panel[panel["t"] >= panel["first_t"]].copy()

    panel["transaction_count"] = panel["transaction_count"].fillna(0)
    panel["total_quantity"] = panel["total_quantity"].fillna(0)
    panel["destination_count"] = panel["destination_count"].fillna(0)
    panel["fob_estimated_pct"] = panel["fob_estimated_pct"].fillna(0)
    # dormant months (0 transactions) have no price observations; carry forward last known price
    panel = panel.sort_values(["drug_substance", "t"])
    for c in ["p25_unit_price", "p50_unit_price", "p75_unit_price"]:
        panel[c] = panel.groupby("drug_substance")[c].ffill()
    panel["top5_exporters"] = panel["top5_exporters"].apply(lambda x: x if isinstance(x, frozenset) else frozenset())

    panel["months_since_first"] = panel["t"] - panel["first_t"]
    panel["avg_deal_size"] = np.where(
        panel["transaction_count"] > 0, panel["total_quantity"] / panel["transaction_count"], 0.0
    )
    panel["price_spread"] = (panel["p75_unit_price"] - panel["p25_unit_price"]) / panel["p50_unit_price"]

    panel = panel.sort_values(["drug_substance", "t"]).reset_index(drop=True)
    prev_exp = panel.groupby("drug_substance")["top5_exporters"].shift(1)
    def _overlap(cur, prev):
        if not isinstance(prev, frozenset) or len(prev) == 0 or len(cur) == 0:
            return np.nan
        return len(cur & prev) / 5.0
    panel["exporter_churn"] = [
        _overlap(cur, prev) for cur, prev in zip(panel["top5_exporters"], prev_exp)
    ]

    panel = panel.drop(columns=["top5_exporters", "first_t"])
    return panel


def build_panel_country(df: pd.DataFrame) -> pd.DataFrame:
    """Same construction as build_panel, but keyed on (drug_substance,
    country_normalized, t) -- mirrors Code.py's job_price_trends_monthly groupby.
    The zero-fill grid is built only over PAIRS that are ever active (not the
    full drug x country cross product, which is ~88% fabricated zeros: 793
    drugs x 210 countries = 166,530 possible pairs vs ~20,772 actually active).
    destination_count is dropped here (always 1 at this grain, degenerate).
    """
    df = df[df["drug_substance"] != "UNCLASSIFIED"].copy()
    month_order = _month_order(df)
    month_index = {ym: i + 1 for i, ym in enumerate(month_order)}
    T = len(month_order)
    df["t"] = df.apply(lambda r: month_index[(r["year"], r["month"])], axis=1)

    key = ["drug_substance", "country_normalized", "t"]
    g = df.groupby(key)
    base = g.agg(
        transaction_count=("drug_substance", "size"),
        total_quantity=("Quantity", "sum"),
        p25_unit_price=("Unit Price in Foreign Currency", lambda s: s.quantile(0.25)),
        p50_unit_price=("Unit Price in Foreign Currency", lambda s: s.quantile(0.50)),
        p75_unit_price=("Unit Price in Foreign Currency", lambda s: s.quantile(0.75)),
        fob_estimated_pct=("fob_estimated", "mean"),
    ).reset_index()

    top5 = g["Exporter Name"].apply(_top5_exporter_set).reset_index(name="top5_exporters")
    base = base.merge(top5, on=key)

    # full grid: only over pairs that are ever active, each cross its own
    # [first_t, 8] window -- not the full drug x country cross product.
    pairs = base[["drug_substance", "country_normalized"]].drop_duplicates()
    months = pd.DataFrame({"t": range(1, T + 1)})
    full = pairs.merge(months, how="cross")
    panel = full.merge(base, on=key, how="left")

    first_t = panel.dropna(subset=["transaction_count"]).groupby(
        ["drug_substance", "country_normalized"])["t"].min()
    panel["first_t"] = panel.set_index(["drug_substance", "country_normalized"]).index.map(first_t)
    panel = panel[panel["t"] >= panel["first_t"]].copy()

    panel["transaction_count"] = panel["transaction_count"].fillna(0)
    panel["total_quantity"] = panel["total_quantity"].fillna(0)
    panel["fob_estimated_pct"] = panel["fob_estimated_pct"].fillna(0)
    panel = panel.sort_values(["drug_substance", "country_normalized", "t"])
    for c in ["p25_unit_price", "p50_unit_price", "p75_unit_price"]:
        panel[c] = panel.groupby(["drug_substance", "country_normalized"])[c].ffill()
    panel["top5_exporters"] = panel["top5_exporters"].apply(
        lambda x: x if isinstance(x, frozenset) else frozenset())

    panel["months_since_first"] = panel["t"] - panel["first_t"]
    panel["avg_deal_size"] = np.where(
        panel["transaction_count"] > 0, panel["total_quantity"] / panel["transaction_count"], 0.0
    )
    panel["price_spread"] = (panel["p75_unit_price"] - panel["p25_unit_price"]) / panel["p50_unit_price"]

    panel = panel.sort_values(["drug_substance", "country_normalized", "t"]).reset_index(drop=True)
    ent_g = panel.groupby(["drug_substance", "country_normalized"])
    prev_exp = ent_g["top5_exporters"].shift(1)

    def _overlap(cur, prev):
        if not isinstance(prev, frozenset) or len(prev) == 0 or len(cur) == 0:
            return np.nan
        return len(cur & prev) / 5.0

    panel["exporter_churn"] = [
        _overlap(cur, prev) for cur, prev in zip(panel["top5_exporters"], prev_exp)
    ]

    # tier flags for the model-fitting fallback cascade (two-model-spec.md §10):
    # n_active_months = active (nonzero) months in this pair's own window;
    # consecutive4 = any 4-month-in-a-row run of active months present, the
    # minimum depth the Arellano-Bond order condition needs (t, t-1, t-2, t-3).
    active = panel["transaction_count"] > 0
    n_active = active.groupby([panel["drug_substance"], panel["country_normalized"]]).transform("sum")
    panel["n_active_months"] = n_active.astype(int)

    def _has_run4(sub):
        s = sub.to_numpy()
        if len(s) < 4:
            return False
        run = 0
        for v in s:
            run = run + 1 if v else 0
            if run >= 4:
                return True
        return False

    consecutive4 = ent_g["transaction_count"].apply(lambda s: _has_run4(s > 0))
    panel["consecutive4"] = panel.set_index(["drug_substance", "country_normalized"]).index.map(consecutive4)

    panel = panel.drop(columns=["top5_exporters", "first_t"])
    return panel


def main():
    t0 = time.time()
    print(f"Loading {etl.SOURCE_FILE} ...")
    raw = etl.load_data(etl.SOURCE_FILE)
    clean, stage_counts = etl.apply_quality_filters(raw)
    enriched, stage_counts = etl.enrich(clean, stage_counts)
    print(f"Enriched rows: {len(enriched):,} ({time.time()-t0:.1f}s)")

    panel = build_panel(enriched)
    os.makedirs("Code results", exist_ok=True)
    panel.to_csv(OUT_PATH, index=False)
    print(f"-> {OUT_PATH} ({len(panel):,} rows, {panel['drug_substance'].nunique():,} substances, {time.time()-t0:.1f}s)")

    panel_c = build_panel_country(enriched)
    panel_c.to_csv(OUT_PATH_COUNTRY, index=False)
    n_pairs = panel_c[["drug_substance", "country_normalized"]].drop_duplicates().shape[0]
    print(f"-> {OUT_PATH_COUNTRY} ({len(panel_c):,} rows, {n_pairs:,} drug x country pairs, {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
