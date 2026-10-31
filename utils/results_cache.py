"""
Reuse of the draws already computed by the paper_study notebooks (02-05). The results of a notebook are saved in a CSV
whose name contains the number of draws, e.g. results/04_..._20draws.csv; draw d is the same whatever the number of
draws (its seeds depend only on the dataset, the KL and d), so a run with another number of draws can start from it.

- load_draws: the rows of RESULTS_CSV; if it does not exist yet, the rows with draw < draws of the CSV files of the same
              experiment with another number of draws (saved as RESULTS_CSV), so only the missing draws are computed
"""

import glob
import os
import re

import pandas as pd

KEYS = ['dataset', 'target_KL', 'draw', 'method']        # a result row (target_KL only in the label-shift notebooks)


def load_draws(results_csv, draws, keys=KEYS):
    """Rows of `results_csv` if it exists. Otherwise the rows with draw < draws of every CSV with the same name and another
    '<n>draws' (more draws: their first `draws` draws, nothing left to compute; fewer: all of them, the rest is computed),
    written to `results_csv`. keys: the columns that identify a result row (duplicates across files are dropped).
    Returns a DataFrame (empty if nothing was found)."""
    if os.path.exists(results_csv):
        records = pd.read_csv(results_csv)
        print(f'{len(records)} rows loaded from {results_csv}')
        return records
    folder, name = os.path.split(results_csv)
    if not re.search(r'\d+draws', name):
        return pd.DataFrame()
    pattern = re.compile('^' + re.escape(name).replace(re.escape(re.search(r'\d+draws', name).group()), r'(\d+)draws') + '$')
    found = sorted((int(m.group(1)), f) for f in glob.glob(os.path.join(folder or '.', '*draws*.csv'))
                   if (m := pattern.match(os.path.basename(f))))
    if not found:
        print(f'0 rows: {results_csv} is new')
        return pd.DataFrame()
    records = pd.concat([pd.read_csv(f) for _, f in found], ignore_index=True)
    records = records[records['draw'] < draws]
    records = records.drop_duplicates(subset=[k for k in keys if k in records.columns], keep='last').reset_index(drop=True)
    records.to_csv(results_csv, index=False)
    print(f"{len(records)} rows (draws 0-{records['draw'].max() if len(records) else '-'}) taken from "
          f"{', '.join(os.path.basename(f) for _, f in found)} -> {results_csv}")
    return records
