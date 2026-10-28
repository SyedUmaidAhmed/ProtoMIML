#!/usr/bin/env python3
"""Classical MIML baselines re-run on ProtoMIML's exact folds (KAIS revision).

The published numbers in Table 4 come from other partitions (Carroll, Frost,
MSCV2) or from a smaller training set (Scene, Reuters: 1500/500).  To remove
both differences we re-run the classical methods ourselves on the SAME folds,
the SAME train-fold feature standardisation and the SAME 90/10 fitting /
validation carve-out (seeded by the fold index, exactly as train_cv.run_fold),
with hyperparameters chosen on the validation split only and per-label
Hamming thresholds tuned on it -- the treatment ProtoMIML receives.

    mimlknn     MIML-kNN (Zhang, 2010): average-Hausdorff bag distance,
                references + citers neighbourhood, least-squares map from
                neighbourhood label counts to label scores.  k on validation.
    mlknn       ML-kNN (Zhang & Zhou, 2007) Bayesian posteriors over the same
                average-Hausdorff bag distances.  k on validation.
    mimlsvm     MIMLSVM (Zhou & Zhang, 2007): k-medoids over training bags
                with the (maximal) Hausdorff distance, k = 20% of the training
                bags; each bag becomes its vector of distances to the medoids;
                one RBF-SVM per label.  (C, gamma) on validation.
    br_mean     binary-relevance logistic regression on bag-mean features.

    python run_classical_baselines.py --dataset all
    python run_classical_baselines.py --dataset yeast,yeast_si

Writes results/revision/classical_<dataset>.json (per-fold metrics and the
chosen hyperparameters).
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import MIMLDatasetCV
from data.flat_knn_dataset import FlatKNNBagDataset
from metrics.miml_metrics import AllFiveWithThresholds, optimize_thresholds
from metrics.miml_metrics import AveragePrecision as AP_fn
from run_experiment import CONFIGS
from run_revision import FLAT_K, make_splits

K_GRID = [3, 5, 7, 9, 11, 15, 21]
SVM_C = [0.1, 1.0, 10.0]
SVM_GAMMA_SCALE = [0.25, 1.0, 4.0]   # multiples of 1 / (n_medoids * var)
SMOOTH = 1.0


# ── bag distances ────────────────────────────────────────────────────────────
def bag_distances(bags_a, bags_b, kind='avg', bags_per_chunk=128):
    """Average ('avg') or maximal ('max') Hausdorff distance for every bag pair.

    Instances of B are stacked once; A is processed in chunks of WHOLE bags so
    every per-bag reduction sees all of that bag's rows.
    """
    Xb = np.concatenate(bags_b).astype(np.float64)
    nb = np.array([len(b) for b in bags_b])
    sb = np.concatenate([[0], np.cumsum(nb)[:-1]])
    sq_b = (Xb ** 2).sum(1)
    D = np.zeros((len(bags_a), len(bags_b)))
    for lo in range(0, len(bags_a), bags_per_chunk):
        chunk = bags_a[lo:lo + bags_per_chunk]
        Xa = np.concatenate(chunk).astype(np.float64)
        na = np.array([len(b) for b in chunk])
        sa = np.concatenate([[0], np.cumsum(na)[:-1]])
        d = (Xa ** 2).sum(1)[:, None] + sq_b[None, :] - 2.0 * Xa @ Xb.T
        np.maximum(d, 0.0, out=d)
        np.sqrt(d, out=d)
        a_to_b = np.minimum.reduceat(d, sb, axis=1)          # (inst_a, bags_b): min over b's instances
        b_to_a = np.minimum.reduceat(d, sa, axis=0)          # (bags_a, inst_b): min over a's instances
        if kind == 'avg':
            s_ab = np.add.reduceat(a_to_b, sa, axis=0)       # (bags_a, bags_b)
            s_ba = np.add.reduceat(b_to_a, sb, axis=1)
            D[lo:lo + len(chunk)] = (s_ab + s_ba) / (na[:, None] + nb[None, :])
        else:
            m_ab = np.maximum.reduceat(a_to_b, sa, axis=0)
            m_ba = np.maximum.reduceat(b_to_a, sb, axis=1)
            D[lo:lo + len(chunk)] = np.maximum(m_ab, m_ba)
    return D


# ── MIML-kNN ────────────────────────────────────────────────────────────────
def _neighbourhood(D_q_ref, D_ref_ref, k):
    n_q, n_ref = D_q_ref.shape
    ind = np.zeros((n_q, n_ref))
    ref = np.argpartition(D_q_ref, min(k, n_ref - 1), axis=1)[:, :k]
    np.add.at(ind, (np.repeat(np.arange(n_q), ref.shape[1]), ref.ravel()), 1.0)
    Dm = D_ref_ref.copy()
    np.fill_diagonal(Dm, np.inf)
    radius = np.partition(Dm, min(k, n_ref - 1) - 1, axis=1)[:, min(k, n_ref - 1) - 1]
    ind += (D_q_ref <= radius[None, :]).astype(float)        # citers
    return ind


def mimlknn(D_fit, Y_fit, D_query, k, ridge=1e-6):
    ind = _neighbourhood(D_fit, D_fit, k)
    np.fill_diagonal(ind, 0.0)                                # leave-one-out
    Phi = ind @ Y_fit
    W = np.linalg.solve(Phi.T @ Phi + ridge * np.eye(Phi.shape[1]), Phi.T @ Y_fit)
    return (_neighbourhood(D_query, D_fit, k) @ Y_fit) @ W


# ── ML-kNN ──────────────────────────────────────────────────────────────────
def mlknn(D_fit, Y_fit, D_query, k):
    n, L = Y_fit.shape
    prior = (SMOOTH + Y_fit.sum(0)) / (2 * SMOOTH + n)
    Dm = D_fit.copy()
    np.fill_diagonal(Dm, np.inf)
    cnt = Y_fit[np.argsort(Dm, axis=1)[:, :k]].sum(1).astype(int)
    c1, c0 = np.zeros((L, k + 1)), np.zeros((L, k + 1))
    for l in range(L):
        np.add.at(c1[l], cnt[Y_fit[:, l] == 1, l], 1)
        np.add.at(c0[l], cnt[Y_fit[:, l] == 0, l], 1)
    lik1 = (SMOOTH + c1) / (SMOOTH * (k + 1) + c1.sum(1, keepdims=True))
    lik0 = (SMOOTH + c0) / (SMOOTH * (k + 1) + c0.sum(1, keepdims=True))
    cq = Y_fit[np.argsort(D_query, axis=1)[:, :k]].sum(1).astype(int)
    out = np.empty((len(D_query), L))
    for l in range(L):
        p1 = prior[l] * lik1[l, cq[:, l]]
        p0 = (1 - prior[l]) * lik0[l, cq[:, l]]
        out[:, l] = p1 / np.maximum(p1 + p0, 1e-12)
    return out


# ── MIMLSVM ─────────────────────────────────────────────────────────────────
def k_medoids(D, k, seed=0, iters=50):
    rng = np.random.RandomState(seed)
    med = rng.choice(len(D), k, replace=False)
    for _ in range(iters):
        assign = np.argmin(D[:, med], axis=1)
        new = med.copy()
        for c in range(k):
            members = np.where(assign == c)[0]
            if len(members):
                new[c] = members[np.argmin(D[np.ix_(members, members)].sum(1))]
        if np.array_equal(np.sort(new), np.sort(med)):
            break
        med = new
    return med


def mimlsvm_scores(Z_fit, Y_fit, Z_query, C, gamma):
    out = np.zeros((len(Z_query), Y_fit.shape[1]))
    for l in range(Y_fit.shape[1]):
        y = Y_fit[:, l]
        if y.min() == y.max():
            out[:, l] = 1.0 if y[0] else -1.0
            continue
        clf = SVC(C=C, kernel='rbf', gamma=gamma).fit(Z_fit, y)
        out[:, l] = clf.decision_function(Z_query)
    return 1.0 / (1.0 + np.exp(-out))                       # monotone; ranks unchanged


def br_mean(B_fit, Y_fit, B_query):
    Xf = np.stack([b.mean(0) for b in B_fit])
    Xq = np.stack([b.mean(0) for b in B_query])
    out = np.zeros((len(Xq), Y_fit.shape[1]))
    for l in range(Y_fit.shape[1]):
        y = Y_fit[:, l]
        if y.min() == y.max():
            out[:, l] = float(y[0])
            continue
        out[:, l] = LogisticRegression(max_iter=2000).fit(Xf, y).predict_proba(Xq)[:, 1]
    return out


# ── protocol ────────────────────────────────────────────────────────────────
def load(name):
    config = CONFIGS['yeast' if name in FLAT_K else name].copy()
    if name in FLAT_K:
        ds = FlatKNNBagDataset(config['data_path'], k_neighbors=FLAT_K[name])
    else:
        ds = MIMLDatasetCV(mat_path=config['data_path'],
                           bag_key=config.get('bag_key', 'bags'),
                           label_key=config.get('label_key', 'labels'),
                           dd_format=config.get('dd_format', False))
    return ds, config


def evaluate(Y_val, s_val, Y_te, s_te):
    th = optimize_thresholds(Y_val, s_val, metric='hamming')
    return {k: float(v) for k, v in AllFiveWithThresholds(Y_te, s_te, th).items()}


def run_dataset(name, protocol='standard', repeats=10):
    ds0, config = load(name)
    splits = make_splits(config, ds0, protocol, repeats)
    per_fold = {m: [] for m in ['mimlknn', 'mlknn', 'mimlsvm', 'br_mean']}
    chosen = {m: [] for m in ['mimlknn', 'mlknn', 'mimlsvm']}
    print(f"\n=== {name} ({protocol}): {len(ds0)} bags, {len(splits)} splits ===", flush=True)

    for fi, (train_idx, test_idx) in enumerate(splits):
        t0 = time.time()
        ds = copy.deepcopy(ds0)
        ds.normalize_features(train_idx)
        tr = np.array(train_idx)
        np.random.seed(fi)
        np.random.shuffle(tr)
        n_val = max(len(tr) // 10, 1)
        val_idx, fit_idx = tr[:n_val].tolist(), tr[n_val:].tolist()
        if hasattr(ds, 'prepare_fold'):
            ds.prepare_fold(fit_idx)

        bags = [b.numpy().astype(np.float64) for b in ds.bags]
        Y = (np.stack([l.numpy() for l in ds.labels]) > 0).astype(int)
        B_fit = [bags[i] for i in fit_idx]
        B_val = [bags[i] for i in val_idx]
        B_te = [bags[i] for i in test_idx]
        Y_fit, Y_val, Y_te = Y[fit_idx], Y[val_idx], Y[test_idx]

        D_ff = bag_distances(B_fit, B_fit)
        D_vf = bag_distances(B_val, B_fit)
        D_tf = bag_distances(B_te, B_fit)
        grid = [k for k in K_GRID if k < len(B_fit)]

        for mname, fn in [('mimlknn', mimlknn), ('mlknn', mlknn)]:
            k = max(grid, key=lambda kk: AP_fn(Y_val, fn(D_ff, Y_fit, D_vf, kk)))
            chosen[mname].append(int(k))
            per_fold[mname].append(evaluate(Y_val, fn(D_ff, Y_fit, D_vf, k),
                                            Y_te, fn(D_ff, Y_fit, D_tf, k)))

        H_ff = bag_distances(B_fit, B_fit, kind='max')
        n_med = max(2, int(round(0.2 * len(B_fit))))
        med = k_medoids(H_ff, n_med, seed=fi)
        Z_fit = H_ff[:, med]
        Z_val = bag_distances(B_val, [B_fit[m] for m in med], kind='max')
        Z_te = bag_distances(B_te, [B_fit[m] for m in med], kind='max')
        base_gamma = 1.0 / (Z_fit.shape[1] * Z_fit.var())
        best = None
        for C in SVM_C:
            for g in SVM_GAMMA_SCALE:
                ap = AP_fn(Y_val, mimlsvm_scores(Z_fit, Y_fit, Z_val, C, g * base_gamma))
                if best is None or ap > best[0]:
                    best = (ap, C, g)
        _, C, g = best
        chosen['mimlsvm'].append({'C': C, 'gamma_scale': g, 'n_medoids': n_med})
        per_fold['mimlsvm'].append(evaluate(
            Y_val, mimlsvm_scores(Z_fit, Y_fit, Z_val, C, g * base_gamma),
            Y_te, mimlsvm_scores(Z_fit, Y_fit, Z_te, C, g * base_gamma)))

        per_fold['br_mean'].append(evaluate(Y_val, br_mean(B_fit, Y_fit, B_val),
                                            Y_te, br_mean(B_fit, Y_fit, B_te)))
        print(f"  split {fi + 1}: " + "  ".join(
            f"{m}={per_fold[m][-1]['AveragePrecision']:.4f}" for m in per_fold)
            + f"  ({time.time() - t0:.0f}s)", flush=True)

    out = {m: {'per_fold': folds,
               'summary': {k: {'mean': float(np.mean([f[k] for f in folds])),
                               'std': float(np.std([f[k] for f in folds]))}
                           for k in folds[0]},
               'chosen': chosen.get(m)}
           for m, folds in per_fold.items()}
    os.makedirs(os.path.join('results', 'revision'), exist_ok=True)
    suffix = '' if protocol == 'standard' else f'_{protocol}'
    path = os.path.join('results', 'revision', f'classical_{name}{suffix}.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"  -> {path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='all')
    p.add_argument('--protocol', default='standard', choices=['standard', 'holdout'])
    p.add_argument('--repeats', type=int, default=10)
    args = p.parse_args()
    names = (['scene', 'reuters', 'letter_carroll', 'letter_frost', 'MSCV2', 'yeast', 'yeast_si']
             if args.dataset == 'all' else args.dataset.split(','))
    for n in names:
        run_dataset(n, args.protocol, args.repeats)


if __name__ == '__main__':
    main()
