"""Cross-Validation runner for ProtoMIML.

Supports multi-seed ensemble, per-label threshold optimization,
and balanced sampling.
"""

import copy
import json
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from data.dataset import MIMLDatasetCV, collate_fn, get_cv_splits
from models.proto_miml import ProtoMIML
from metrics.miml_metrics import (
    AllFive, AllFiveWithThresholds, optimize_thresholds, AveragePrecision as AP_fn
)
from training.trainer import ProtoMIMLTrainer


def create_model(config, device='cpu'):
    """Create ProtoMIML model from config dict."""
    model = ProtoMIML(
        input_dim=config['input_dim'],
        num_labels=config['num_labels'],
        embed_dim=config.get('embed_dim', 64),
        num_heads=config.get('num_heads', 4),
        ff_dim=config.get('ff_dim', 128),
        num_inducing=config.get('num_inducing', 4),
        dropout=config.get('dropout', 0.15),
        proto_temperature=config.get('proto_temperature', 0.1),
        proto_ema_decay=config.get('proto_ema_decay', 0.999),
        use_isab=config.get('use_isab', True),
        use_gcn=config.get('use_gcn', True),
        use_prototypes=config.get('use_prototypes', True),
        embedding_layers=config.get('embedding_layers', 2),
        learnable_gate=config.get('learnable_gate', True),
    )
    return model.to(device)


def _init_model(model, config, dataset, train_idx, device):
    """Initialize label GCN from data statistics."""
    if config.get('use_gcn', True):
        co_matrix = dataset.get_label_cooccurrence(train_idx)
        model.init_label_gcn(co_matrix.to(device))


def run_fold(fold_idx, dataset_orig, train_idx, test_idx, config, device='cpu'):
    """Run a single fold of cross-validation."""
    print(f"\n{'='*60}", flush=True)
    print(f"Fold {fold_idx + 1}", flush=True)
    print(f"{'='*60}", flush=True)

    dataset = copy.deepcopy(dataset_orig)
    dataset.normalize_features(train_idx)

    # Split train into train/val (90/10)
    np.random.seed(fold_idx)
    train_arr = np.array(train_idx)
    np.random.shuffle(train_arr)
    val_size = max(len(train_arr) // 10, 1)
    val_idx = train_arr[:val_size].tolist()
    actual_train_idx = train_arr[val_size:].tolist()

    sampler = dataset.get_balanced_sampler(actual_train_idx)
    train_loader = DataLoader(
        Subset(dataset, actual_train_idx), batch_size=config['batch_size'],
        sampler=sampler, collate_fn=collate_fn, num_workers=0, drop_last=False)
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=config['batch_size'],
        shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(
        Subset(dataset, test_idx), batch_size=config['batch_size'],
        shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Multi-seed ensemble
    ensemble_seeds = config.get('ensemble_seeds', None)
    if ensemble_seeds and len(ensemble_seeds) > 1:
        return _run_fold_ensemble(
            fold_idx, dataset, actual_train_idx, val_idx, test_idx,
            train_loader, val_loader, test_loader, config, device, ensemble_seeds)

    # Single model training
    model = create_model(config, device)
    _init_model(model, config, dataset, actual_train_idx, device)

    trainer = ProtoMIMLTrainer(model, config, device)

    mixup_str = " + Mixup" if config.get('use_mixup') else ""
    print(f"Phase 1: Foundation training (SAM + BCE{mixup_str})...", flush=True)
    trainer.train_phase1(
        train_loader, val_loader,
        num_epochs=config.get('phase1_epochs', 80),
        lr=config.get('phase1_lr', 1e-3))

    print(f"Phase 2: Full training (SAM + BCE + Consistency + SWA{mixup_str})...", flush=True)
    trainer.train_phase2(
        train_loader, val_loader,
        max_epochs=config.get('phase2_max_epochs', 120),
        lr=config.get('phase2_lr', 3e-4),
        patience=config.get('early_stop_patience', 25))

    use_threshold_opt = config.get('use_threshold_opt', True)

    if use_threshold_opt:
        val_labels, val_raw, val_fused = trainer.get_predictions(val_loader)
        test_labels, test_raw, test_fused = trainer.get_predictions(test_loader)

        val_ap_raw = AP_fn(val_labels, val_raw)
        val_ap_fused = AP_fn(val_labels, val_fused)

        if val_ap_fused >= val_ap_raw:
            val_probs, test_probs = val_fused, test_fused
            source = "fused"
        else:
            val_probs, test_probs = val_raw, test_raw
            source = "parametric"

        thresholds = optimize_thresholds(val_labels, val_probs, metric='hamming')
        print(f"  Optimized thresholds ({source}): {thresholds.round(3)}", flush=True)
        test_metrics = AllFiveWithThresholds(test_labels, test_probs, thresholds)
    else:
        test_metrics = trainer.evaluate(test_loader)

    print(f"\nFold {fold_idx + 1} Results:", flush=True)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}", flush=True)
    return test_metrics


def _run_fold_ensemble(fold_idx, dataset, actual_train_idx, val_idx, test_idx,
                       train_loader, val_loader, test_loader, config, device, seeds):
    """Run multiple seeds and average predictions for ensemble."""
    print(f"  Running {len(seeds)}-seed ensemble...", flush=True)

    all_test_probs = []
    all_val_probs = []
    test_labels = None
    val_labels = None

    mixup_str = " + Mixup" if config.get('use_mixup') else ""

    for seed_idx, seed in enumerate(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = create_model(config, device)
        _init_model(model, config, dataset, actual_train_idx, device)
        trainer = ProtoMIMLTrainer(model, config, device)

        print(f"  Seed {seed} ({seed_idx+1}/{len(seeds)}) [SAM + BCE{mixup_str}]:", flush=True)
        trainer.train_phase1(
            train_loader, val_loader,
            num_epochs=config.get('phase1_epochs', 80),
            lr=config.get('phase1_lr', 1e-3))
        trainer.train_phase2(
            train_loader, val_loader,
            max_epochs=config.get('phase2_max_epochs', 120),
            lr=config.get('phase2_lr', 3e-4),
            patience=config.get('early_stop_patience', 25))

        vl, vr, vf = trainer.get_predictions(val_loader)
        tl, tr, tf = trainer.get_predictions(test_loader)

        if AP_fn(vl, vf) >= AP_fn(vl, vr):
            all_val_probs.append(vf)
            all_test_probs.append(tf)
            print(f"  Seed {seed}: selected FUSED pathway (val AP fused >= parametric)", flush=True)
        else:
            all_val_probs.append(vr)
            all_test_probs.append(tr)
            print(f"  Seed {seed}: selected PARAMETRIC pathway (prototype bank bypassed at test time)", flush=True)

        val_labels = vl
        test_labels = tl

    # Average probabilities across seeds
    val_probs = np.mean(all_val_probs, axis=0)
    test_probs = np.mean(all_test_probs, axis=0)

    use_threshold_opt = config.get('use_threshold_opt', True)
    if use_threshold_opt:
        thresholds = optimize_thresholds(val_labels, val_probs, metric='hamming')
        print(f"  Ensemble thresholds: {thresholds.round(3)}", flush=True)
        test_metrics = AllFiveWithThresholds(test_labels, test_probs, thresholds)
    else:
        test_metrics = AllFive(test_labels, test_probs)

    print(f"\nFold {fold_idx + 1} Ensemble Results:", flush=True)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}", flush=True)
    return test_metrics


def _save_incremental(all_metrics, dataset_name):
    """Save intermediate results after each fold to prevent data loss."""
    os.makedirs('results', exist_ok=True)
    n = len(all_metrics)
    results = {}
    for name in all_metrics[0].keys():
        values = [m[name] for m in all_metrics]
        results[name] = [float(np.mean(values)), float(np.std(values))]
    path = f'results/incremental_{dataset_name}_{n}folds.json'
    with open(path, 'w') as f:
        json.dump({dataset_name: results, 'per_fold': [
            {k: float(v) for k, v in m.items()} for m in all_metrics
        ]}, f, indent=2)
    print(f"  [Saved incremental results to {path}]", flush=True)


def run_cv(config, device='cpu', num_folds=None):
    """Run cross-validation."""
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

    # CV splits
    if 'cv_mat_path' in config and config['cv_mat_path']:
        splits = get_cv_splits(config['cv_mat_path'], len(dataset), 10)
    else:
        k = num_folds or 5
        indices = np.arange(len(dataset))
        np.random.seed(42)
        np.random.shuffle(indices)
        fold_size = len(dataset) // k
        splits = []
        for i in range(k):
            test_idx = indices[i * fold_size:(i + 1) * fold_size].tolist()
            train_idx = np.concatenate([indices[:i * fold_size],
                                        indices[(i + 1) * fold_size:]]).tolist()
            splits.append((train_idx, test_idx))

    if num_folds is not None:
        splits = splits[:num_folds]

    all_metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        fold_metrics = run_fold(fold_idx, dataset, train_idx, test_idx, config, device)
        all_metrics.append(fold_metrics)

        # Save incremental results after each fold
        _save_incremental(all_metrics, config.get('dataset_name', 'unknown'))

    # Aggregate results
    n = len(splits)
    print(f"\n{'='*60}", flush=True)
    print(f"{n}-Fold CV Results (mean +/- std)", flush=True)
    print(f"{'='*60}", flush=True)

    results = {}
    for name in all_metrics[0].keys():
        values = [m[name] for m in all_metrics]
        mean = np.mean(values)
        std = np.std(values)
        results[name] = (mean, std)
        print(f"  {name}: {mean:.4f} +/- {std:.4f}", flush=True)

    return results, all_metrics
