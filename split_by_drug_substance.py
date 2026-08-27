"""
Split product_analytics.csv (drug_substance x year x month aggregates,
built by Code.py) into one .xlsx file per drug_substance.

Note: this operates on pre-aggregated data -- each output file has one row
per year/month the drug shipped in (up to 8, for Jan-Aug 2024), not
per-shipment transaction detail. For full transaction-level rows, rerun
Code.py's enrichment stage directly instead.
"""

import os
import re
import time

import pandas as pd

OUT_DIR = "Code results"
SOURCE_CSV = os.path.join(OUT_DIR, "product_analytics.csv")
SPLIT_DIR = os.path.join(OUT_DIR, "drug_substance_split")


def main():
    t0 = time.time()

    print(f"Loading {SOURCE_CSV} ...")
    df = pd.read_csv(SOURCE_CSV)

    os.makedirs(SPLIT_DIR, exist_ok=True)
    print("Splitting by drug_substance ...")
    n = 0
    for substance, group in df.groupby("drug_substance"):
        safe_name = re.sub(r'[\\/*?:"<>|]', "_", str(substance)).strip()[:150] or "UNKNOWN"
        path = os.path.join(SPLIT_DIR, f"{safe_name}.xlsx")
        group.to_excel(path, index=False)
        n += 1

    print(f"  -> {SPLIT_DIR}\\*.xlsx ({n} files, {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()