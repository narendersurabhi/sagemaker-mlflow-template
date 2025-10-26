#!/usr/bin/env python3
"""
Population Stability Index (PSI) for tabular columns.
Computes PSI between a reference dataset and a current dataset.

Usage:
  python src/monitor/psi.py --ref data/ref.parquet --cur data/cur.parquet --cols billed paid units --out reports/psi.json
"""
import argparse, json, pathlib
import numpy as np, pandas as pd

def psi_for_series(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref = ref.replace([np.inf, -np.inf], np.nan).dropna()
    cur = cur.replace([np.inf, -np.inf], np.nan).dropna()
    if ref.empty or cur.empty:
        return float("nan")
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, bins=edges)[0].astype(float)
    c = np.histogram(cur, bins=edges)[0].astype(float)
    r = np.where(r == 0, 0.5, r)
    c = np.where(c == 0, 0.5, c)
    r, c = r / r.sum(), c / c.sum()
    return float(np.sum((r - c) * np.log(r / c)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference parquet file")
    ap.add_argument("--cur", required=True, help="current parquet file")
    ap.add_argument("--cols", nargs="+", required=False, help="columns to evaluate; default = numeric cols")
    ap.add_argument("--out", default="reports/psi.json")
    args = ap.parse_args()

    ref = pd.read_parquet(args.ref)
    cur = pd.read_parquet(args.cur)

    if args.cols:
        cols = args.cols
    else:
        num = ref.select_dtypes(include=[np.number]).columns
        cols = [c for c in num if c in cur.columns]

    results = {}
    for c in cols:
        try:
            results[c] = {"psi": psi_for_series(ref[c], cur[c])}
        except Exception as e:
            results[c] = {"psi": None, "error": str(e)}

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"columns": results}, indent=2))
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
