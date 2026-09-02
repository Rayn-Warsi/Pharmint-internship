"""Writes backtest_log.xlsx from the cached raw scores left by run_backtest.py,
without redoing any model fits. Use after closing a locked backtest_log.xlsx."""

from run_backtest import LOG_PATH, RAW_CACHE_PATH, write_workbook
import pandas as pd

if __name__ == "__main__":
    if not RAW_CACHE_PATH.exists():
        raise SystemExit(f"No cache at {RAW_CACHE_PATH} -- run run_backtest.py first.")
    result = pd.read_csv(RAW_CACHE_PATH)
    write_workbook(result, LOG_PATH)
    RAW_CACHE_PATH.unlink(missing_ok=True)
    print(f"-> {LOG_PATH} ({len(result)} rows)")
