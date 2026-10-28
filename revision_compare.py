#!/usr/bin/env python3
"""Paired comparisons between revision runs (same folds, same recipe, CPU).

    python revision_compare.py

Reads results/revision/summary_<tag>.json (written by revision_analysis.py) and
compares the full model against each controlled variant with a two-sided
paired Wilcoxon signed-rank test over the 10 outer folds, for the five MIML
metrics under the paper's deployment policy ('selected').  Writes
results/revision/comparisons.json.
"""
import json
import os

import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.join('results', 'revision')
DATASETS = ['scene', 'reuters', 'letter_carroll', 'letter_frost', 'MSCV2']
METRICS = ['HammingLoss', 'OneError', 'Coverage', 'RankingLoss', 'AveragePrecision']
LOWER_BETTER = {'HammingLoss', 'OneError', 'Coverage', 'RankingLoss'}
VARIANTS = ['protomiml_noproto', 'protomiml_nosam', 'abmil_per_label', 'abmil']


def load(tag):
    path = os.path.join(ROOT, f'summary_{tag}.json')
    return json.load(open(path))['policies'] if os.path.exists(path) else None


def main():
    out = {}
    for ds in DATASETS:
        full = load(f'protomiml_standard_{ds}')
        if full is None:
            continue
        for var in VARIANTS:
            other = load(f'{var}_standard_{ds}')
            if other is None:
                continue
            rows = {}
            for m in METRICS:
                x = np.array(full['selected'][m]['per_fold'])
                y = np.array(other['selected'][m]['per_fold'])
                p = float(wilcoxon(x, y).pvalue) if np.any(x != y) else 1.0
                full_better = x.mean() < y.mean() if m in LOWER_BETTER else x.mean() > y.mean()
                rows[m] = {'full': float(x.mean()), 'other': float(y.mean()), 'p': p,
                           'full_better': bool(full_better)}
            out.setdefault(ds, {})[var] = rows
            cells = ' '.join(f"{m[:4]} {r['full']:.4f}/{r['other']:.4f}"
                             f"{'*' if r['p'] < .05 else ' '}{'+' if r['full_better'] else '-'}"
                             for m, r in rows.items())
            print(f'{ds:15s} vs {var:18s} {cells}')
    json.dump(out, open(os.path.join(ROOT, 'comparisons.json'), 'w'), indent=1)
    print("\n(full/other; * p<.05 paired Wilcoxon over 10 folds; + full better)")


if __name__ == '__main__':
    main()
