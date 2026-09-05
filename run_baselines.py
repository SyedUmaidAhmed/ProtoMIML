#!/usr/bin/env python3
"""Neural MIL baselines for ProtoMIML, run under the identical CV protocol.

Trains ABMIL (Ilse et al., 2018) in two multi-label variants -- shared
attention and per-label attention -- on the same folds, with the same feature
normalisation, the same 90/10 train/val carve-out, the same per-label
threshold optimisation and the same five metrics as ProtoMIML.

The point of comparison is deliberately like-for-like: the only thing that
changes between these runs and the ProtoMIML runs is the model.

    python run_baselines.py --model abmil            # shared attention
    python run_baselines.py --model abmil_per_label  # per-label attention
    python run_baselines.py --model all --dataset scene

Saves results/baseline_{model}_{dataset}_{folds}folds.json
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import MIMLDatasetCV, collate_fn, get_cv_splits
from metrics.miml_metrics import (
    AllFiveWithThresholds, optimize_thresholds, AveragePrecision as AP_fn
)
from models.abmil import ABMIL
from run_experiment import CONFIGS

DATASETS = ['scene', 'reuters', 'letter_carroll', 'letter_frost', 'MSCV2']
MODELS = ['abmil', 'abmil_per_label']

# Total epoch budget matched to ProtoMIML (phase1 80 + phase2 up to 200).
MAX_EPOCHS = 280


def build_model(model_name, config, device):
    return ABMIL(
        input_dim=config['input_dim'],
        num_labels=config['num_labels'],
        embed_dim=config.get('embed_dim', 64),
        attn_dim=config.get('embed_dim', 64),
        dropout=config.get('dropout', 0.15),
        per_label_attention=(model_name == 'abmil_per_label'),
    ).to(device)


@torch.no_grad()
def predict(model, loader, device):
    """Return (labels, probs) over a loader."""
    model.eval()
    all_labels, all_probs = [], []
    for bags, labels, mask in loader:
        bags, mask = bags.to(device), mask.to(device)
        logits = model(bags, mask)['logits']
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def run_fold(fold_idx, dataset_orig, train_idx, test_idx, config, model_name, device):
    """One CV fold, mirroring training/train_cv.run_fold's protocol."""
    print(f"\n{'='*60}\nFold {fold_idx + 1}\n{'='*60}", flush=True)

    dataset = copy.deepcopy(dataset_orig)
    dataset.normalize_features(train_idx)

    # 90/10 train/val carve-out, seeded by fold index (identical to train_cv)
    np.random.seed(fold_idx)
    train_arr = np.array(train_idx)
    np.random.shuffle(train_arr)
    val_size = max(len(train_arr) // 10, 1)
    val_idx = train_arr[:val_size].tolist()
    actual_train_idx = train_arr[val_size:].tolist()

    sampler = dataset.get_balanced_sampler(actual_train_idx)
    train_loader = DataLoader(
        Subset(dataset, actual_train_idx), batch_size=config['batch_size'],
        sampler=sampler, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=config['batch_size'],
        shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(
        Subset(dataset, test_idx), batch_size=config['batch_size'],
        shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = build_model(model_name, config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.get('phase1_lr', 1e-3),
        weight_decay=config.get('weight_decay', 5e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    best_ap, best_state, patience = -1.0, None, 0
    max_patience = config.get('early_stop_patience', 35)

    for epoch in range(MAX_EPOCHS):
        model.train()
        for bags, labels, mask in train_loader:
            bags, labels, mask = bags.to(device), labels.to(device), mask.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bags, mask)['logits'], labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.get('gradient_clip', 1.0))
            optimizer.step()
        scheduler.step()

        val_labels, val_probs = predict(model, val_loader, device)
        val_ap = AP_fn(val_labels, val_probs)
        if val_ap > best_ap:
            best_ap, patience = val_ap, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= max_patience:
                print(f"  Early stop at epoch {epoch + 1} (best val AP {best_ap:.4f})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_labels, val_probs = predict(model, val_loader, device)
    test_labels, test_probs = predict(model, test_loader, device)

    thresholds = optimize_thresholds(val_labels, val_probs, metric='hamming')
    test_metrics = AllFiveWithThresholds(test_labels, test_probs, thresholds)

    print(f"Fold {fold_idx + 1} results:", flush=True)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}", flush=True)
    return test_metrics


def run_cv_baseline(dataset_name, model_name, num_folds, device):
    config = CONFIGS[dataset_name].copy()
    config.pop('ensemble_seeds', None)
    config['dataset_name'] = dataset_name

    dataset = MIMLDatasetCV(
        mat_path=config['data_path'],
        bag_key=config.get('bag_key', 'bags'),
        label_key=config.get('label_key', 'labels'),
        dd_format=config.get('dd_format', False),
    )
    config['input_dim'] = dataset.input_dim
    config['num_labels'] = dataset.num_labels
    print(f"Dataset: {len(dataset)} samples, {dataset.num_labels} labels, "
          f"{dataset.input_dim}-dim features", flush=True)

    # Split construction copied verbatim from training/train_cv.run_cv so that
    # baselines and ProtoMIML are evaluated on exactly the same folds.
    if config.get('cv_mat_path'):
        splits = get_cv_splits(config['cv_mat_path'], len(dataset), 10)
    else:
        k = num_folds
        indices = np.arange(len(dataset))
        np.random.seed(42)
        np.random.shuffle(indices)
        fold_size = len(dataset) // k
        splits = []
        for i in range(k):
            test_i = indices[i * fold_size:(i + 1) * fold_size].tolist()
            train_i = np.concatenate([indices[:i * fold_size],
                                      indices[(i + 1) * fold_size:]]).tolist()
            splits.append((train_i, test_i))
    splits = splits[:num_folds]

    torch.manual_seed(42)
    np.random.seed(42)

    all_metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        all_metrics.append(
            run_fold(fold_idx, dataset, train_idx, test_idx, config, model_name, device))

    results = {
        name: (float(np.mean([m[name] for m in all_metrics])),
               float(np.std([m[name] for m in all_metrics])))
        for name in all_metrics[0]
    }

    os.makedirs('results', exist_ok=True)
    path = f'results/baseline_{model_name}_{dataset_name}_{len(splits)}folds.json'
    with open(path, 'w') as f:
        json.dump({k: {'mean': v[0], 'std': v[1]} for k, v in results.items()}, f, indent=2)
    print(f"\nSaved: {path}")

    print(f"\n{len(splits)}-Fold CV ({model_name} on {dataset_name}):")
    for metric in ['HammingLoss', 'OneError', 'Coverage', 'RankingLoss', 'AveragePrecision']:
        if metric in results:
            print(f"  {metric}: {results[metric][0]:.4f} +/- {results[metric][1]:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='all', choices=DATASETS + ['all'])
    parser.add_argument('--model', default='all', choices=MODELS + ['all'])
    parser.add_argument('--folds', type=int, default=10)
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    models = MODELS if args.model == 'all' else [args.model]

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    summary = {}
    for model_name in models:
        summary[model_name] = {}
        for dataset in datasets:
            print(f"\n{'#'*60}\n# {model_name} on {dataset}\n{'#'*60}")
            try:
                summary[model_name][dataset] = run_cv_baseline(
                    dataset, model_name, args.folds, device)
            except Exception as e:
                print(f"ERROR on {model_name}/{dataset}: {e}")
                import traceback; traceback.print_exc()

    print("\n\nBASELINE RUNS COMPLETE\n" + "=" * 60)
    for model_name, per_ds in summary.items():
        print(f"\n{model_name}:")
        for d, res in per_ds.items():
            ap = res.get('AveragePrecision', (0, 0))
            hl = res.get('HammingLoss', (0, 0))
            print(f"  {d}: AP={ap[0]:.4f}, HL={hl[0]:.4f}")


if __name__ == '__main__':
    main()
