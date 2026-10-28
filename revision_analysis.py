#!/usr/bin/env python3
"""Score the archived predictions of run_revision.py under every deployment policy.

Because run_revision.py stores the per-seed validation and test probabilities of
the parametric, prototype-only and fused pathways, the questions below are
answered from the SAME trained models, with no re-training:

  selected      the paper's rule: per seed, deploy fused if its validation AP
                >= the parametric validation AP, else parametric; average the
                5 seeds; per-label Hamming thresholds on validation
  parametric    always deploy the parametric pathway (prototype bank ignored)
  fused         always deploy the fused pathway (no selection step at all)
  proto         prototype scores only (non-parametric pathway alone)
  *_t05         the same, with every threshold fixed at 0.5 (no threshold
                optimisation at all; only Hamming loss / F1 can change)

For cvens runs the pathway and the thresholds are chosen on the pooled
out-of-fold predictions of the whole outer training partition.

    python revision_analysis.py                      # every run in results/revision
    python revision_analysis.py protomiml_standard_letter_frost

Writes results/revision/summary_<tag>.json.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics.miml_metrics import (AllFive, AllFiveWithThresholds, optimize_thresholds,
                                  AveragePrecision as AP_fn)

ROOT = os.path.join('results', 'revision')
METRICS = ['HammingLoss', 'OneError', 'Coverage', 'RankingLoss', 'AveragePrecision',
           'MacroF1', 'MicroF1']


def score(val_y, val_p, test_y, test_p, tune=True):
    if not tune:
        return AllFive(test_y, test_p)
    thr = optimize_thresholds(val_y, val_p, metric='hamming')
    return AllFiveWithThresholds(test_y, test_p, thr)


def standard_policies(r):
    vy, ty = r['val_labels'], r['test_labels']
    n_seeds = r['val_raw'].shape[0]
    pick_fused = np.array([AP_fn(vy, r['val_fused'][s]) >= AP_fn(vy, r['val_raw'][s])
                           for s in range(n_seeds)])
    sel_v = np.mean([r['val_fused'][s] if pick_fused[s] else r['val_raw'][s]
                     for s in range(n_seeds)], axis=0)
    sel_t = np.mean([r['test_fused'][s] if pick_fused[s] else r['test_raw'][s]
                     for s in range(n_seeds)], axis=0)
    pv = {'selected': (sel_v, sel_t)}
    for name in ['raw', 'fused', 'proto']:
        pv[{'raw': 'parametric'}.get(name, name)] = (r[f'val_{name}'].mean(0),
                                                    r[f'test_{name}'].mean(0))
    out = {}
    for name, (v, t) in pv.items():
        out[name] = score(vy, v, ty, t)
        out[name + '_t05'] = score(vy, v, ty, t, tune=False)
    extra = {'n_fused_selected': int(pick_fused.sum()), 'n_seeds': int(n_seeds),
             'val_size': int(len(vy)), 'gate': r['gate'].tolist()}
    return out, extra


def cvens_policies(r):
    oy, ty = r['oof_labels'], r['test_labels']
    ap_f, ap_r = AP_fn(oy, r['oof_fused']), AP_fn(oy, r['oof_raw'])
    use = 'fused' if ap_f >= ap_r else 'raw'
    pv = {'selected': (r[f'oof_{use}'], r[f'test_{use}'].mean(0)),
          'parametric': (r['oof_raw'], r['test_raw'].mean(0)),
          'fused': (r['oof_fused'], r['test_fused'].mean(0)),
          'proto': (r['oof_proto'], r['test_proto'].mean(0))}
    out = {}
    for name, (v, t) in pv.items():
        out[name] = score(oy, v, ty, t)
        out[name + '_t05'] = score(oy, v, ty, t, tune=False)
    extra = {'pathway': use, 'val_size': int(len(oy)), 'gate': r['gate'].tolist()}
    return out, extra


def summarise(tag):
    d = os.path.join(ROOT, tag)
    files = sorted(glob.glob(os.path.join(d, 'fold_*.npz')),
                   key=lambda p: int(p.rsplit('_', 1)[1].split('.')[0]))
    if not files:
        return None
    cfg = json.load(open(os.path.join(d, 'run_config.json')))
    per_fold, extras = [], []
    for f in files:
        r = dict(np.load(f))
        res, ex = (cvens_policies if 'oof_raw' in r else standard_policies)(r)
        per_fold.append(res)
        extras.append(ex)
    policies = per_fold[0].keys()
    summary = {
        policy: {m: {'mean': float(np.mean([pf[policy][m] for pf in per_fold])),
                     'std': float(np.std([pf[policy][m] for pf in per_fold])),
                     'per_fold': [float(pf[policy][m]) for pf in per_fold]}
                 for m in METRICS}
        for policy in policies}
    out = {'tag': tag, 'model': cfg['model'], 'protocol': cfg['protocol'],
           'dataset': cfg['dataset'], 'val_seed': cfg.get('val_seed', 0),
           'n_folds': len(files), 'policies': summary, 'fold_info': extras}
    with open(os.path.join(ROOT, f'summary_{tag}.json'), 'w') as f:
        json.dump(out, f, indent=1)
    return out


def main():
    tags = sys.argv[1:] or sorted(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, '*'))
                                  if os.path.isdir(p) and p.rsplit('/', 1)[-1] != 'logs')
    for tag in tags:
        s = summarise(tag)
        if s is None:
            continue
        sel = s['policies']['selected']
        line = '  '.join(f"{m[:4]}={sel[m]['mean']:.4f}" for m in METRICS[:5])
        print(f"{tag:55s} [{s['n_folds']:2d} folds] selected: {line}")
        for pol in ['parametric', 'fused', 'proto']:
            if pol in s['policies']:
                p = s['policies'][pol]
                print(f"{'':55s}   {pol:10s}: AP={p['AveragePrecision']['mean']:.4f} "
                      f"HL={p['HammingLoss']['mean']:.4f} RL={p['RankingLoss']['mean']:.4f}")


if __name__ == '__main__':
    main()
