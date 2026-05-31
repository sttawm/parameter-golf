#!/usr/bin/env python3
"""Merge lama_results_a.csv and lama_results_b.csv into lama_results_cosine_multi.csv."""
import pandas as pd

main = pd.read_csv("lama_results_cosine_multi.csv")
a    = pd.read_csv("lama_results_a.csv")
b    = pd.read_csv("lama_results_b.csv")

new_a = a[a["seed"].astype(str).apply(lambda s: int(float(s)) if s not in ("nan",) else -1).isin(range(9000, 21000))]
new_b = b[b["seed"].astype(str).apply(lambda s: int(float(s)) if s not in ("nan",) else -1).isin(range(21000, 35000))]

merged = pd.concat([main, new_a, new_b], ignore_index=True)
merged.to_csv("lama_results_cosine_multi.csv", index=False)
print(f"Merged: {len(main)} original + {len(new_a)} from A + {len(new_b)} from B = {len(merged)} total rows")

finals = merged[merged["epoch"] == "final"]
print("\nFinal rows per lambda:")
print(finals.groupby("lambda")["seed"].count())
