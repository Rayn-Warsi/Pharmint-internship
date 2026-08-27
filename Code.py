"""
PharmInt ETL Pipeline (Python / pandas port)

Mirrors etl-pipeline.html: raw India pharma export shipment records ->
quality filters -> enrichment -> six pre-aggregated analytics tables.

Source: 30_JAN24-AUG24 Data Set.xlsx (~1M rows, Jan-Aug 2024 export shipments)
Outputs: written to "Code results/" as CSV files + a run log.
"""

import os
import re
import time
import numpy as np
import pandas as pd

SOURCE_FILE = "Data/combined_export_dataset.parquet"
OUT_DIR = "Code results"
DRUG_NAME_DIR = os.path.join("Seperated dataset", "pharma_split")
if not os.path.isdir(DRUG_NAME_DIR):
    DRUG_NAME_DIR = os.path.join("Code results", "drug_substance_split")


def _load_drug_name_pattern():
    """Compile a regex matching any known active-ingredient/drug name against
    Product text, from the reference drug list in Seperated dataset/pharma_split."""
    names = [os.path.splitext(f)[0].upper() for f in os.listdir(DRUG_NAME_DIR) if f.lower().endswith(".xlsx")]
    names = sorted(set(names), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b")


DRUG_NAME_PATTERN = _load_drug_name_pattern()

# ---------------------------------------------------------------------------
# Country normalization: raw label (as seen in source) -> canonical name
# ---------------------------------------------------------------------------
COUNTRY_NORMALIZE = {
    "ANTIGUA": "ANTIGUA AND BARBUDA",
    "AZARBAIJAN": "AZERBAIJAN",
    "BOSNIA & HERZEGOVINA": "BOSNIA AND HERZEGOVINA",
    "BRIT VIRGIN IS": "BRITISH VIRGIN ISLANDS",
    "COMORO ISLANDS": "COMOROS",
    "CONGO D REP": "DR CONGO",
    "CONGO D REPULIC": "DR CONGO",
    "CONGO, THE DEMOCRATIC REPUBLIC OF THE": "DR CONGO",
    "COTE D IVOIRE": "IVORY COAST",
    "CZECHOSLOVAKIA": "CZECH REPUBLIC",
    "DOMINICAN REP": "DOMINICAN REPUBLIC",
    "DOMINICAN REPULIC": "DOMINICAN REPUBLIC",
    "EQUATORAL GUINEA": "EQUATORIAL GUINEA",
    "GAUTEMALA": "GUATEMALA",
    "GYANA": "GUYANA",
    "HONGKONG": "HONG KONG",
    "KAZAKISTAN": "KAZAKHSTAN",
    "KOREA DEMOCRATIC PEOPLES REPUBLIC OF": "NORTH KOREA",
    "KOREA,DEMOCRATIC PEOPLE'S REPUBLIC OF": "NORTH KOREA",
    "KOREA, REPULIC": "SOUTH KOREA",
    "KOREA,REPUBLIC OF": "SOUTH KOREA",
    "REPUBLIC OF KOREA": "SOUTH KOREA",
    "KYRGHYSTAN": "KYRGYZSTAN",
    "LAO": "LAOS",
    "LAO PEOPLE'S DEMOCRATIC REPUBLIC": "LAOS",
    "LIBYAN ARAB REP": "LIBYA",
    "LIBYAN ARAB REPUBLIC": "LIBYA",
    "MACAO ISLANDS": "MACAU",
    "MACAU ISLANDS": "MACAU",
    "MACEDONIA": "NORTH MACEDONIA",
    "MACEDONIA,THE FORMER YUGOSLAV REPUBLIC OF": "NORTH MACEDONIA",
    "MOLDOVA, REP. OF": "MOLDOVA",
    "MOLDOVA,REPUBLIC OF": "MOLDOVA",
    "NAMBIA": "NAMIBIA",
    "NETHLANDS ANTL": "NETHERLANDS ANTILLES",
    "PANAMA REPUBLIC": "PANAMA",
    "SIERRA LEONA": "SIERRA LEONE",
    "SLOVAK REPUBLIC": "SLOVAKIA",
    "SOMAALIA": "SOMALIA",
    "ST KIT-N-ANGUILA": "SAINT KITTS AND NEVIS",
    "ST KITTS-NEVIS-ANGUILLA": "SAINT KITTS AND NEVIS",
    "ST LUCIA": "SAINT LUCIA",
    "ST VINCENT": "SAINT VINCENT AND THE GRENADINES",
    "SAINT VINCENT AND GRENADINES": "SAINT VINCENT AND THE GRENADINES",
    "SURINAM": "SURINAME",
    "TEMOR LESTE": "TIMOR LESTE",
    "TRINIDAD & TOBAGO": "TRINIDAD AND TOBAGO",
    "TURKS & CAICOS ISLANDS": "TURKS AND CAICOS ISLANDS",
    "UAE": "UNITED ARAB EMIRATES",
    "URUGAY": "URUGUAY",
    "VIETNAM DEM REP": "VIETNAM",
    "VIETNAM, DEMOCRATIC REP. OF": "VIETNAM",
    "WEST SAMOA": "SAMOA",
    "YEMEN, DEMO": "YEMEN",
    "YEMEN, DEMOCRATIC": "YEMEN",
    "PALESTINE STATE": "PALESTINE",
}

# canonical country -> world region
REGION_MAP = {}
_REGIONS = {
    "ASIA": [
        "AFGHANISTAN", "ARMENIA", "AZERBAIJAN", "BAHRAIN", "BANGLADESH", "BHUTAN",
        "BRUNEI", "CAMBODIA", "CHINA", "GEORGIA", "HONG KONG", "INDONESIA", "IRAN",
        "IRAQ", "ISRAEL", "JAPAN", "JORDAN", "KAZAKHSTAN", "NORTH KOREA", "SOUTH KOREA",
        "KUWAIT", "KYRGYZSTAN", "LAOS", "LEBANON", "MACAU", "MALAYSIA", "MALDIVES",
        "MONGOLIA", "MYANMAR", "NEPAL", "OMAN", "PAKISTAN", "PALESTINE", "PHILIPPINES",
        "QATAR", "SAUDI ARABIA", "SINGAPORE", "SRI LANKA", "SYRIA", "TAIWAN",
        "TAJIKISTAN", "THAILAND", "TIMOR LESTE", "TURKEY", "TURKMENISTAN",
        "UNITED ARAB EMIRATES", "UZBEKISTAN", "VIETNAM", "YEMEN",
    ],
    "EUROPE": [
        "ALBANIA", "AUSTRIA", "BELARUS", "BELGIUM", "BOSNIA AND HERZEGOVINA",
        "BULGARIA", "CROATIA", "CYPRUS", "CZECH REPUBLIC", "DENMARK", "ESTONIA",
        "FINLAND", "FRANCE", "GERMANY", "GIBRALTAR", "GREECE", "HUNGARY", "ICELAND",
        "IRELAND", "ITALY", "KOSOVO", "LATVIA", "LIECHTENSTEIN", "LITHUANIA",
        "LUXEMBOURG", "MALTA", "MOLDOVA", "MONACO", "MONTENEGRO", "NETHERLANDS",
        "NORTH MACEDONIA", "NORWAY", "POLAND", "PORTUGAL", "ROMANIA", "RUSSIA",
        "SERBIA", "SLOVAKIA", "SLOVENIA", "SPAIN", "SWEDEN", "SWITZERLAND",
        "UKRAINE", "UNITED KINGDOM",
    ],
    "AFRICA": [
        "ALGERIA", "ANGOLA", "BENIN", "BOTSWANA", "BURKINA FASO", "BURUNDI",
        "CAMEROON", "CAPE VERDE ISLANDS", "CENTRAL AFRICAN REPUBLIC", "CHAD",
        "COMOROS", "CONGO", "DR CONGO", "DJIBOUTI", "EGYPT", "EQUATORIAL GUINEA",
        "ERITREA", "ETHIOPIA", "GABON", "GAMBIA", "GHANA", "GUINEA",
        "GUINEA BISSAU", "IVORY COAST", "KENYA", "LESOTHO", "LIBERIA", "LIBYA",
        "MADAGASCAR", "MALAWI", "MALI", "MAURITANIA", "MAURITIUS", "MOROCCO",
        "MOZAMBIQUE", "NAMIBIA", "NIGER", "NIGERIA", "REUNION", "RWANDA",
        "SAO TOME AND PRINCIPE", "SENEGAL", "SEYCHELLES", "SIERRA LEONE",
        "SOMALIA", "SOUTH AFRICA", "SOUTH SUDAN", "SUDAN", "SWAZILAND",
        "TANZANIA", "TOGO", "TUNISIA", "UGANDA", "WESTERN SAHARA", "ZAMBIA",
        "ZIMBABWE",
    ],
    "NORTH AMERICA": [
        "ANGUILLA", "ANTIGUA AND BARBUDA", "ARUBA", "BAHAMAS", "BARBADOS",
        "BELIZE", "BERMUDA", "CANADA", "CAYMAN ISLANDS", "COSTA RICA", "CUBA",
        "CURACAO", "DOMINICA", "DOMINICAN REPUBLIC", "EL SALVADOR", "GRENADA",
        "GUADELOUPE", "GUATEMALA", "HAITI", "HONDURAS", "JAMAICA", "MARTINIQUE",
        "MEXICO", "MONTSERRAT", "NETHERLANDS ANTILLES", "NICARAGUA", "PANAMA",
        "PUERTO RICO", "SAINT KITTS AND NEVIS", "SAINT LUCIA",
        "SAINT VINCENT AND THE GRENADINES", "TRINIDAD AND TOBAGO",
        "TURKS AND CAICOS ISLANDS", "UNITED STATES",
    ],
    "SOUTH AMERICA": [
        "ARGENTINA", "BOLIVIA", "BRAZIL", "CHILE", "COLOMBIA", "ECUADOR",
        "GUYANA", "PARAGUAY", "PERU", "SURINAME", "URUGUAY", "VENEZUELA",
    ],
    "OCEANIA": [
        "AUSTRALIA", "FIJI", "GUAM", "KIRIBATI", "MARSHALL ISLANDS", "MICRONESIA",
        "NEW ZEALAND", "NORTHERN MARIANA ISLANDS", "PALAU", "PAPUA NEW GUINEA",
        "SAMOA", "SOLOMON ISLANDS", "TONGA", "UNITED STATES MINOR OUTLYING ISLANDS",
        "VANUATU",
    ],
}
for _region, _countries in _REGIONS.items():
    for _c in _countries:
        REGION_MAP[_c] = _region

# Product-description junk patterns (biological samples, veterinary, not-for-sale)
JUNK_PRODUCT_PATTERNS = re.compile(
    r"\b(?:BLOOD SAMPLE|BIOLOGICAL SAMPLE|VETERINARY USE|ANIMAL USE|NOT FOR SALE|"
    r"FREE SAMPLE|SAMPLE ONLY|GIFT PARCEL|PERSONAL USE ONLY)\b",
    re.IGNORECASE,
)


def load_data(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def apply_quality_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Stage 1: drop rows that fail basic quality checks. Returns (clean_df, stage_counts)."""
    stage_counts = {"raw": len(df)}

    price = pd.to_numeric(df["Unit Price in Foreign Currency"], errors="coerce")
    qty = pd.to_numeric(df["Quantity"], errors="coerce")
    product = df["Product"].astype("string").str.strip()
    country = df["Foreign Country"].astype("string").str.strip().str.upper()
    currency = df["Currency"].astype("string").str.strip()

    mask = (
        currency.notna() & (currency != "")
        & price.notna() & (price > 0)
        & qty.notna() & (qty > 0)
        & country.notna() & (country != "") & (country != "INDIA")
        & product.notna() & (product.str.len() >= 3)
        & ~product.str.contains(JUNK_PRODUCT_PATTERNS, na=False)
    )
    clean = df.loc[mask].copy()
    stage_counts["after_basic_filters"] = len(clean)

    return clean, stage_counts


def enrich(df: pd.DataFrame, stage_counts: dict) -> tuple[pd.DataFrame, dict]:
    """Stage 2: FOB fallback estimation, country normalization, region, date parts."""
    df = df.copy()
    df["Unit Price in Foreign Currency"] = pd.to_numeric(df["Unit Price in Foreign Currency"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Total FOB in INR"] = pd.to_numeric(df["Total FOB in INR"], errors="coerce")
    df["Total FOB in Foreign Currency"] = pd.to_numeric(df["Total FOB in Foreign Currency"], errors="coerce")

    valid_inr = df["Total FOB in INR"] > 0
    valid_fgn = df["Total FOB in Foreign Currency"] > 0

    # implied FX rate per currency, from rows where both values are trustworthy
    fx_basis = df.loc[valid_inr & valid_fgn]
    implied_fx = (fx_basis["Total FOB in INR"] / fx_basis["Total FOB in Foreign Currency"]).groupby(
        fx_basis["Currency"]
    ).median()

    fallback_fx = df["Currency"].map(implied_fx)
    fallback_value = df["Quantity"] * df["Unit Price in Foreign Currency"] * fallback_fx

    df["fob_value_inr"] = np.where(valid_inr, df["Total FOB in INR"], fallback_value)
    df["fob_estimated"] = (~valid_inr).astype(int)

    before = len(df)
    df = df[df["fob_value_inr"].notna() & (df["fob_value_inr"] > 0)]
    stage_counts["after_fob_recovery"] = len(df)
    stage_counts["fob_unrecoverable_dropped"] = before - len(df)

    country_upper = df["Foreign Country"].astype("string").str.strip().str.upper()
    df["country_normalized"] = country_upper.map(lambda c: COUNTRY_NORMALIZE.get(c, c))
    df["region"] = df["country_normalized"].map(lambda c: REGION_MAP.get(c, "OTHER"))

    df["Date"] = pd.to_datetime(df["Date"])
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["year_month"] = df["Date"].dt.to_period("M").astype(str)

    df["product_key"] = clean_product_key(df["Product"])
    df["drug_substance"] = (
        df["product_key"].str.extract(DRUG_NAME_PATTERN, expand=False).fillna("UNCLASSIFIED")
    )

    return df, stage_counts


# strips batch/invoice noise so shipments of the same drug group together instead
# of exploding into one group per batch number (e.g. "... - BATCH NO -TT110003")
_BATCH_NOISE_RE = re.compile(
    r"-?\s*BATCH\s*NO\.?\s*[-:]?\s*\S*|MFG\.?\s*\d{1,2}/\d{2,4}|EXP\.?\s*\d{1,2}/\d{2,4}",
    re.IGNORECASE,
)


def clean_product_key(product: pd.Series) -> pd.Series:
    key = product.astype("string").str.upper()
    key = key.str.replace(_BATCH_NOISE_RE, "", regex=True)
    key = key.str.replace(r"[-,]\s*$", "", regex=True)
    key = key.str.replace(r"\s+", " ", regex=True).str.strip()
    return key


def _top_n_counts(series: pd.Series, n: int = 5) -> str:
    return "; ".join(f"{k} ({v})" for k, v in series.value_counts().head(n).items())


def job_product_analytics(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["drug_substance", "year", "month"])
    out = g.agg(
        transaction_count=("drug_substance", "size"),
        total_quantity=("Quantity", "sum"),
        total_fob_inr=("fob_value_inr", "sum"),
        avg_unit_price=("Unit Price in Foreign Currency", "mean"),
        min_unit_price=("Unit Price in Foreign Currency", "min"),
        max_unit_price=("Unit Price in Foreign Currency", "max"),
        stddev_unit_price=("Unit Price in Foreign Currency", "std"),
        p25_unit_price=("Unit Price in Foreign Currency", lambda s: s.quantile(0.25)),
        p50_unit_price=("Unit Price in Foreign Currency", lambda s: s.quantile(0.50)),
        p75_unit_price=("Unit Price in Foreign Currency", lambda s: s.quantile(0.75)),
        p99_unit_price=("Unit Price in Foreign Currency", lambda s: s.quantile(0.99)),
        fob_estimated_pct=("fob_estimated", lambda s: round(100 * s.mean(), 2)),
    ).reset_index()

    top_exporters = g["Exporter Name"].apply(_top_n_counts).reset_index(name="top5_exporters")
    top_destinations = g["country_normalized"].apply(_top_n_counts).reset_index(name="top5_destinations")
    top_brands = g["product_key"].apply(lambda s: _top_n_counts(s, 5)).reset_index(name="top5_product_descriptions")
    out = out.merge(top_exporters, on=["drug_substance", "year", "month"]).merge(
        top_destinations, on=["drug_substance", "year", "month"]
    ).merge(top_brands, on=["drug_substance", "year", "month"])
    return out.sort_values(["drug_substance", "year", "month"])


def job_country_market_intel(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["country_normalized", "year", "month"])
    out = g.agg(
        transaction_count=("country_normalized", "size"),
        total_import_value_inr=("fob_value_inr", "sum"),
        total_quantity=("Quantity", "sum"),
        fob_estimated_pct=("fob_estimated", lambda s: round(100 * s.mean(), 2)),
    ).reset_index()

    top_products = g["drug_substance"].apply(lambda s: _top_n_counts(s, 10)).reset_index(name="top10_products")
    top_exporters = g["Exporter Name"].apply(_top_n_counts).reset_index(name="top_exporters")
    top_ports = g["Indian Port"].apply(_top_n_counts).reset_index(name="top_indian_ports")

    out = out.merge(top_products[["country_normalized", "year", "month", "top10_products"]],
                     on=["country_normalized", "year", "month"])
    out = out.merge(top_exporters, on=["country_normalized", "year", "month"])
    out = out.merge(top_ports, on=["country_normalized", "year", "month"])
    return out.sort_values(["country_normalized", "year", "month"])


def job_exporter_directory(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Exporter Name")
    out = g.agg(
        transaction_count=("Exporter Name", "size"),
        years_active=("year", "nunique"),
        months_active=("year_month", "nunique"),
        countries_served=("country_normalized", "nunique"),
        drug_substances_shipped=("drug_substance", "nunique"),
        sku_variants_shipped=("product_key", "nunique"),
        total_fob_inr=("fob_value_inr", "sum"),
        city=("City", lambda s: s.dropna().iloc[0] if s.notna().any() else pd.NA),
        exporter_address=("Exporter Address", lambda s: s.dropna().iloc[0] if s.notna().any() else pd.NA),
    ).reset_index()

    out["avg_monthly_shipments"] = (out["transaction_count"] / out["months_active"]).round(2)

    # reliability score: inverse coefficient-of-variation of monthly shipment volume, 0-100
    monthly_counts = df.groupby(["Exporter Name", "year_month"]).size().reset_index(name="n")
    stats = monthly_counts.groupby("Exporter Name")["n"].agg(["mean", "std"]).fillna(0)
    cv = (stats["std"] / stats["mean"]).replace([np.inf, -np.inf], np.nan).fillna(0)
    reliability = (100 / (1 + cv)).round(1)
    out = out.merge(reliability.rename("reliability_score"), left_on="Exporter Name", right_index=True, how="left")

    return out.sort_values("total_fob_inr", ascending=False)


def job_price_trends_monthly(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["drug_substance", "country_normalized", "year_month"])
    out = g.agg(
        transaction_count=("drug_substance", "size"),
        total_quantity=("Quantity", "sum"),
        avg_unit_price=("Unit Price in Foreign Currency", "mean"),
        min_unit_price=("Unit Price in Foreign Currency", "min"),
        max_unit_price=("Unit Price in Foreign Currency", "max"),
    ).reset_index()
    return out.sort_values(["drug_substance", "country_normalized", "year_month"])


def job_market_opportunities(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["drug_substance", "country_normalized"])
    out = g.agg(
        transaction_count=("drug_substance", "size"),
        supplier_count=("Exporter Name", "nunique"),
        size_estimate_inr=("fob_value_inr", "sum"),
    ).reset_index()

    underserved = out[
        (out["drug_substance"] != "UNCLASSIFIED")
        & (out["transaction_count"] >= 5)
        & (out["supplier_count"] < 5)
    ]
    underserved = underserved.sort_values("size_estimate_inr", ascending=False).head(1000)
    return underserved


def split_by_drug_substance(df: pd.DataFrame, out_dir: str) -> list[tuple[str, int]]:
    """Write one .xlsx per drug_substance with the full enriched transaction-level rows.

    UNCLASSIFIED (rows whose product text didn't match a known drug name) is written
    as .csv instead -- it's ~2/3 of all rows and openpyxl is too slow at that size.
    """
    split_dir = os.path.join(out_dir, "drug_substance_split")
    os.makedirs(split_dir, exist_ok=True)
    counts = []
    for substance, group in df.groupby("drug_substance"):
        safe_name = re.sub(r'[\\/*?:"<>|]', "_", str(substance)).strip()[:150] or "UNKNOWN"
        if substance == "UNCLASSIFIED":
            path = os.path.join(split_dir, f"{safe_name}.csv")
            group.to_csv(path, index=False)
        else:
            path = os.path.join(split_dir, f"{safe_name}.xlsx")
            group.to_excel(path, index=False)
        counts.append((substance, len(group)))
    return counts


def job_regional_analytics(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["region", "year"])
    out = g.agg(
        transaction_count=("region", "size"),
        country_count=("country_normalized", "nunique"),
        total_fob_inr=("fob_value_inr", "sum"),
    ).reset_index()
    return out.sort_values(["region", "year"])


def run_etl():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log(f"Loading {SOURCE_FILE} ...")
    raw = load_data(SOURCE_FILE)

    log("Stage 1: quality filters ...")
    clean, stage_counts = apply_quality_filters(raw)

    log("Stage 2: enrichment (FOB fallback, country normalization, region, dates) ...")
    enriched, stage_counts = enrich(clean, stage_counts)

    log("")
    log("Stage counts:")
    for k, v in stage_counts.items():
        log(f"  {k}: {v:,}")
    log(f"  final clean rows: {len(enriched):,} ({100 * len(enriched) / stage_counts['raw']:.1f}% of raw)")
    log("")

    jobs = [
        ("product_analytics", job_product_analytics),
        ("country_market_intel", job_country_market_intel),
        ("exporter_directory", job_exporter_directory),
        ("price_trends_monthly", job_price_trends_monthly),
        ("market_opportunities", job_market_opportunities),
        ("regional_analytics", job_regional_analytics),
    ]

    for name, fn in jobs:
        jt0 = time.time()
        log(f"Running job: {name} ...")
        result = fn(enriched)
        out_path = os.path.join(OUT_DIR, f"{name}.csv")
        result.to_csv(out_path, index=False)
        log(f"  -> {out_path} ({len(result):,} rows, {time.time() - jt0:.1f}s)")

    log("")
    log(f"Total runtime: {time.time() - t0:.1f}s")
    log("(run split_by_drug_substance.py separately to split full transaction rows by drug)")

    with open(os.path.join(OUT_DIR, "pipeline_run_log.txt"), "w") as f:
        f.write("\n".join(log_lines) + "\n")

if __name__ == "__main__":
    run_etl()