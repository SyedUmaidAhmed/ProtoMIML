#!/usr/bin/env python3
"""EXP-T: clean wall-clock timing for the revised Table 5.

    python revision_timing.py [--threads 2] [--datasets scene reuters ...]

Trains outer fold 1 of the standard protocol (5 ensemble members) for each
dataset, one dataset at a time in a single process, on CPU, and records per
member: training wall time, epochs actually run, and test inference time per
bag.  Uses the same code path as run_revision.py (train_member/predict_all),
so the timings describe exactly the models that produced the revision results.
Nothing is archived except results/revision/timing.json.
"""
import argparse
import copy
import json
import os
import platform
import time

import numpy as np
import torch

import run_revision as rr
from data.dataset import MIMLDatasetCV
from run_experiment import CONFIGS


def time_dataset(name, device):
    config = CONFIGS[name].copy()
    config['dataset_name'] = name
    if not config.get('ensemble_seeds'):
        config['ensemble_seeds'] = [42, 123, 456, 789, 1024]
    dataset = MIMLDatasetCV(mat_path=config['data_path'],
                            bag_key=config.get('bag_key', 'bags'),
                            label_key=config.get('label_key', 'labels'),
                            dd_format=config.get('dd_format', False))
    config['input_dim'] = dataset.input_dim
    config['num_labels'] = dataset.num_labels
    torch.manual_seed(42)
    np.random.seed(42)
    train_idx, test_idx = rr.make_splits(config, dataset, 'standard', 10)[0]

    ds = copy.deepcopy(dataset)
    ds.normalize_features(train_idx)
    np.random.seed(0)
    arr = np.array(train_idx)
    np.random.shuffle(arr)
    n_val = max(len(arr) // 10, 1)
    val_idx, tr_idx = arr[:n_val].tolist(), arr[n_val:].tolist()
    train_loader = rr.loader(ds, tr_idx, config, ds.get_balanced_sampler(tr_idx))
    val_loader = rr.loader(ds, val_idx, config)
    test_loader = rr.loader(ds, test_idx, config)

    members = []
    for seed in config['ensemble_seeds']:
        t0 = time.perf_counter()
        model = rr.train_member('protomiml', config, ds, tr_idx, val_loader,
                                train_loader, seed, device)
        t_train = time.perf_counter() - t0
        t0 = time.perf_counter()
        rr.predict_all(model, test_loader, device)
        t_inf = time.perf_counter() - t0
        n_params = sum(p.numel() for p in model.parameters())
        members.append({'seed': seed, 'train_s': t_train,
                        'infer_ms_per_bag': 1000 * t_inf / len(test_idx),
                        'params': n_params})
        print(f"{name} seed {seed}: train {t_train:.1f}s, "
              f"infer {1000 * t_inf / len(test_idx):.2f} ms/bag", flush=True)
    tr = np.array([m['train_s'] for m in members])
    return {'n_train': len(tr_idx), 'n_val': len(val_idx), 'n_test': len(test_idx),
            'params': members[0]['params'], 'members': members,
            'train_s_per_member_mean': float(tr.mean()),
            'train_s_per_member_std': float(tr.std(ddof=1)),
            'train_s_per_fold_ensemble': float(tr.sum()),
            'est_train_h_10fold': float(10 * tr.sum() / 3600),
            'infer_ms_per_bag_mean': float(np.mean([m['infer_ms_per_bag'] for m in members]))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--datasets', nargs='+',
                   default=['scene', 'reuters', 'letter_carroll', 'letter_frost', 'MSCV2'])
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device('cpu')
    out_path = os.path.join('results', 'revision', 'timing.json')
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    out['_hardware'] = {'machine': platform.machine(), 'processor': platform.processor(),
                        'platform': platform.platform(), 'torch': torch.__version__,
                        'device': 'cpu', 'threads': args.threads,
                        'cpu_count': os.cpu_count(),
                        'cpu': os.popen('sysctl -n machdep.cpu.brand_string').read().strip()}
    for name in args.datasets:
        out[name] = time_dataset(name, device)
        json.dump(out, open(out_path, 'w'), indent=1)
        print(f"{name}: {out[name]['train_s_per_member_mean']:.1f}s per member, "
              f"~{out[name]['est_train_h_10fold']:.2f} h for 10 folds x 5 members", flush=True)


if __name__ == '__main__':
    main()
