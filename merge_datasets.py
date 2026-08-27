"""
Merge the three raw export-shipment source files into one combined dataset
for the ETL pipeline (Code.py):

  Data/New/30_JAN23-JUN23EXP.xlsx    (H1 2023)
  Data/New/30_JUL23-DEC23EXP.xlsx    (H2 2023)
  Data/OLD/30_JAN24-AUG24 Data Set.xlsx (Jan-Aug 2024)

Combined row count (~2.49M) exceeds Excel's 1,048,576-row cap, so the merged
output is written as Parquet instead of .xlsx.
"""

import time
import pandas as pd

SOURCES = [
    "Data/New/30_JAN23-JUN23EXP.xlsx",
    "Data/New/30_JUL23-DEC23EXP.xlsx",
    "Data/OLD/30_JAN24-AUG24 Data Set.xlsx",
]
OUT_PATH = "Data/combined_export_dataset.parquet"


def main():
    frames = []
    for path in SOURCES:
        t0 = time.time()
        print(f"Reading {path} ...")
        df = pd.read_excel(path, engine="openpyxl")
        df.columns = [c.strip() for c in df.columns]
        print(f"  -> {len(df):,} rows, {time.time() - t0:.1f}s")
        frames.append(df)

    print("Concatenating ...")
    combined = pd.concat(frames, ignore_index=True)
    print(f"Combined: {len(combined):,} rows, {len(combined.columns)} columns")

    t0 = time.time()
    combined.to_parquet(OUT_PATH, engine="pyarrow", index=False)
    print(f"Wrote {OUT_PATH} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
