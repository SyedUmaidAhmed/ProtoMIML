#!/usr/bin/env python3
"""ProtoMIML: Prototypical Multi-Instance Multi-Label Learning.

A novel semi-parametric method combining:
  1. SAM Optimizer — flatter minima, better generalization
  2. Prototypical Memory Bank — EMA-updated per-label prototypes + gated fusion
  3. Cross-Attention Label Pooling — labels as queries, instances as KV
  4. Label GCN — 2-layer GCN over co-occurrence graph
  5. Multi-View Consistency Training — dual dropout views

Usage:
    python run_experiment.py scene              # Scene dataset (10-fold CV)
    python run_experiment.py reuters            # Reuters dataset (10-fold CV)
    python run_experiment.py all                # Run both datasets
    python run_experiment.py scene --folds 3    # Quick test with 3 folds
    python run_experiment.py scene --no-ensemble # Single model
"""

import torch
import numpy as np
import sys
import os
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training.train_cv import run_cv

# Path to datasets
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'MIML_CLAM_Release', '..', 'VIT_MIL', 'Code', 'MIML-SELFVIT')


# ============================================================
# Dataset Configurations
# ============================================================

CONFIGS = {
    'scene': {
        'data_path': os.path.join(DATA_DIR, 'scene.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, '10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels', 'dataset': 'scene',
        # Architecture
        'embed_dim': 64, 'num_heads': 4, 'ff_dim': 128,
        'num_inducing': 4, 'embedding_layers': 2,
        'use_isab': True, 'use_gcn': True, 'use_prototypes': True,
        # Prototype config
        'proto_temperature': 1.0, 'proto_ema_decay': 0.999,
        # Training
        'batch_size': 32, 'weight_decay': 5e-4, 'gradient_clip': 1.0,
        'phase1_epochs': 80, 'phase1_lr': 1e-3, 'phase1_sam_rho': 0.05,
        'phase2_max_epochs': 200, 'phase2_lr': 3e-4, 'phase2_sam_rho': 0.03,
        'early_stop_patience': 35, 'swa_start': 60,
        'label_smoothing': 0.05, 'dropout': 0.15,
        'consistency_weight': 0.1,
        'use_threshold_opt': True,
        # Augmentation
        'fixed_bag_size': True, 'use_mixup': True, 'mixup_alpha': 0.2,
        # Ensemble
        'ensemble_seeds': [42, 123, 456, 789, 1024],
        'default_folds': 10,
    },
    'reuters': {
        'data_path': os.path.join(DATA_DIR, 'reuters_MIML.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, '10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels', 'dataset': 'reuters',
        # Architecture
        'embed_dim': 64, 'num_heads': 4, 'ff_dim': 128,
        'num_inducing': 6, 'embedding_layers': 2,
        'use_isab': True, 'use_gcn': True, 'use_prototypes': True,
        # Prototype config
        'proto_temperature': 0.5, 'proto_ema_decay': 0.9995,
        # Training
        'batch_size': 32, 'weight_decay': 5e-4, 'gradient_clip': 1.0,
        'phase1_epochs': 80, 'phase1_lr': 1e-3, 'phase1_sam_rho': 0.03,
        'phase2_max_epochs': 200, 'phase2_lr': 3e-4, 'phase2_sam_rho': 0.02,
        'early_stop_patience': 35, 'swa_start': 60,
        'label_smoothing': 0.05, 'dropout': 0.15,
        'consistency_weight': 0.1,
        'use_threshold_opt': True,
        # Augmentation
        'fixed_bag_size': False, 'use_mixup': False,
        # Ensemble
        'ensemble_seeds': [42, 123, 456, 789, 1024],
        'default_folds': 10,
    },
    'birdsong': {
        'data_path': os.path.join(DATA_DIR, 'birdsong.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, 'birdsong_10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels', 'dataset': 'birdsong',
        # Architecture (38-dim features, 19 labels, 257 samples)
        'embed_dim': 64, 'num_heads': 4, 'ff_dim': 128,
        'num_inducing': 4, 'embedding_layers': 2,
        'use_isab': True, 'use_gcn': True, 'use_prototypes': True,
        # Prototype config
        'proto_temperature': 1.0, 'proto_ema_decay': 0.999,
        # Training (smaller dataset → more regularization)
        'batch_size': 8, 'weight_decay': 1e-3, 'gradient_clip': 1.0,
        'phase1_epochs': 80, 'phase1_lr': 1e-3, 'phase1_sam_rho': 0.05,
        'phase2_max_epochs': 200, 'phase2_lr': 3e-4, 'phase2_sam_rho': 0.03,
        'early_stop_patience': 35, 'swa_start': 60,
        'label_smoothing': 0.05, 'dropout': 0.2,
        'consistency_weight': 0.1,
        'use_threshold_opt': True,
        # Augmentation
        'fixed_bag_size': False, 'use_mixup': False,
        # Ensemble
        'ensemble_seeds': [42, 123, 456, 789, 1024],
        'default_folds': 10,
    },
    'protein_azotobacter': {
        'data_path': os.path.join(DATA_DIR, 'protein_azotobacter.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, 'protein_azotobacter_10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels', 'dataset': 'protein_azotobacter',
        # Architecture (216-dim features, 46 labels, 407 samples)
        'embed_dim': 64, 'num_heads': 4, 'ff_dim': 128,
        'num_inducing': 4, 'embedding_layers': 2,
        'use_isab': True, 'use_gcn': True, 'use_prototypes': True,
        # Prototype config
        'proto_temperature': 0.5, 'proto_ema_decay': 0.999,
        # Training
        'batch_size': 8, 'weight_decay': 1e-3, 'gradient_clip': 1.0,
        'phase1_epochs': 80, 'phase1_lr': 1e-3, 'phase1_sam_rho': 0.05,
        'phase2_max_epochs': 200, 'phase2_lr': 3e-4, 'phase2_sam_rho': 0.03,
        'early_stop_patience': 35, 'swa_start': 60,
        'label_smoothing': 0.05, 'dropout': 0.2,
        'consistency_weight': 0.1,
        'use_threshold_opt': True,
        # Augmentation
        'fixed_bag_size': False, 'use_mixup': False,
        # Ensemble
        'ensemble_seeds': [42, 123, 456, 789, 1024],
        'default_folds': 10,
    },
    'protein_geobacter': {
        'data_path': os.path.join(DATA_DIR, 'protein_geobacter.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, 'protein_geobacter_10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels', 'dataset': 'protein_geobacter',
        # Architecture (216-dim features, 28 labels, 379 samples)
        'embed_dim': 64, 'num_heads': 4, 'ff_dim': 128,
        'num_inducing': 4, 'embedding_layers': 2,
        'use_isab': True, 'use_gcn': True, 'use_prototypes': True,
        # Prototype config
        'proto_temperature': 0.5, 'proto_ema_decay': 0.999,
        # Training
        'batch_size': 8, 'weight_decay': 1e-3, 'gradient_clip': 1.0,
        'phase1_epochs': 80, 'phase1_lr': 1e-3, 'phase1_sam_rho': 0.05,
        'phase2_max_epochs': 200, 'phase2_lr': 3e-4, 'phase2_sam_rho': 0.03,
        'early_stop_patience': 35, 'swa_start': 60,
        'label_smoothing': 0.05, 'dropout': 0.2,
        'consistency_weight': 0.1,
        'use_threshold_opt': True,
        # Augmentation
        'fixed_bag_size': False, 'use_mixup': False,
        # Ensemble
        'ensemble_seeds': [42, 123, 456, 789, 1024],
        'default_folds': 10,
    },
    'protein_haloarcula': {
        'data_path': os.path.join(DATA_DIR, 'protein_haloarcula.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, 'protein_haloarcula_10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels', 'dataset': 'protein_haloarcula',
        # Architecture (216-dim features, 21 labels, 304 samples)
        'embed_dim': 64, 'num_heads': 4, 'ff_dim': 128,
        'num_inducing': 4, 'embedding_layers': 2,
        'use_isab': True, 'use_gcn': True, 'use_prototypes': True,
        # Prototype config
        'proto_temperature': 0.5, 'proto_ema_decay': 0.999,
        # Training
        'batch_size': 8, 'weight_decay': 1e-3, 'gradient_clip': 1.0,
        'phase1_epochs': 80, 'phase1_lr': 1e-3, 'phase1_sam_rho': 0.05,
        'phase2_max_epochs': 200, 'phase2_lr': 3e-4, 'phase2_sam_rho': 0.03,
        'early_stop_patience': 35, 'swa_start': 60,
        'label_smoothing': 0.05, 'dropout': 0.2,
        'consistency_weight': 0.1,
        'use_threshold_opt': True,
        # Augmentation
        'fixed_bag_size': False, 'use_mixup': False,
        # Ensemble
        'ensemble_seeds': [42, 123, 456, 789, 1024],
        'default_folds': 10,
    },
    'protein_pyrococcus': {
        'data_path': os.path.join(DATA_DIR, 'protein_pyrococcus.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, 'protein_pyrococcus_10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels', 'dataset': 'protein_pyrococcus',
        # Architecture (216-dim features, 52 labels, 425 samples)
        'embed_dim': 64, 'num_heads': 4, 'ff_dim': 128,
        'num_inducing': 4, 'embedding_layers': 2,
        'use_isab': True, 'use_gcn': True, 'use_prototypes': True,
        # Prototype config
        'proto_temperature': 0.5, 'proto_ema_decay': 0.999,
        # Training
        'batch_size': 8, 'weight_decay': 1e-3, 'gradient_clip': 1.0,
        'phase1_epochs': 80, 'phase1_lr': 1e-3, 'phase1_sam_rho': 0.05,
        'phase2_max_epochs': 200, 'phase2_lr': 3e-4, 'phase2_sam_rho': 0.03,
        'early_stop_patience': 35, 'swa_start': 60,
        'label_smoothing': 0.05, 'dropout': 0.2,
        'consistency_weight': 0.1,
        'use_threshold_opt': True,
        # Augmentation
        'fixed_bag_size': False, 'use_mixup': False,
        # Ensemble
        'ensemble_seeds': [42, 123, 456, 789, 1024],
        'default_folds': 10,
    },
    'yeast': {
        'data_path': os.path.join(DATA_DIR, 'yeast.mat'),
        'cv_mat_path': os.path.join(DATA_DIR, 'yeast_10CV.mat'),
        'bag_key': 'bags', 'label_key': 'labels', 'dataset': 'yeast',
        # Architecture (103-dim features, 14 labels, 2417 samples, fixed 5 instances)
        'embed_dim': 64, 'num_heads': 4, 'ff_dim': 128,
        'num_inducing': 4, 'embedding_layers': 2,
        'use_isab': True, 'use_gcn': True, 'use_prototypes': True,
        'proto_temperature': 1.0, 'proto_ema_decay': 0.999,
        'batch_size': 32, 'weight_decay': 5e-4, 'gradient_clip': 1.0,
        'phase1_epochs': 80, 'phase1_lr': 1e-3, 'phase1_sam_rho': 0.05,
        'phase2_max_epochs': 200, 'phase2_lr': 3e-4, 'phase2_sam_rho': 0.03,
        'early_stop_patience': 35, 'swa_start': 60,
        'label_smoothing': 0.05, 'dropout': 0.15,
        'consistency_weight': 0.1,
        'use_threshold_opt': True,
        'fixed_bag_size': True, 'use_mixup': True, 'mixup_alpha': 0.2,
        'ensemble_seeds': [42, 123, 456, 789, 1024],
        'default_folds': 10,
    },
    'letter_carroll': {
        'data_path': os.path.join(DATA_DIR, 'letter_carroll.mat'),
        'cv_mat_path': None,  # No CV file, use random 10-fold splits
        'dd_format': True, 'dataset': 'letter_carroll',
        # Architecture (16-dim features, 26 labels, 166 bags)
        'embed_dim': 64, 'num_heads': 4, 'ff_dim': 128,
        'num_inducing': 4, 'embedding_layers': 2,
        'use_isab': True, 'use_gcn': True, 'use_prototypes': True,
        'proto_temperature': 1.0, 'proto_ema_decay': 0.999,
        'batch_size': 8, 'weight_decay': 1e-3, 'gradient_clip': 1.0,
        'phase1_epochs': 80, 'phase1_lr': 1e-3, 'phase1_sam_rho': 0.05,
        'phase2_max_epochs': 200, 'phase2_lr': 3e-4, 'phase2_sam_rho': 0.03,
        'early_stop_patience': 35, 'swa_start': 60,
        'label_smoothing': 0.05, 'dropout': 0.2,
        'consistency_weight': 0.1,
        'use_threshold_opt': True,
        'fixed_bag_size': False, 'use_mixup': False,
        'ensemble_seeds': [42, 123, 456, 789, 1024],
        'default_folds': 10,
    },
    'letter_frost': {
        'data_path': os.path.join(DATA_DIR, 'letter_frost.mat'),
        'cv_mat_path': None,  # No CV file, use random 10-fold splits
        'dd_format': True, 'dataset': 'letter_frost',
        # Architecture (16-dim features, 26 labels, 144 bags)
        'embed_dim': 64, 'num_heads': 4, 'ff_dim': 128,
        'num_inducing': 4, 'embedding_layers': 2,
        'use_isab': True, 'use_gcn': True, 'use_prototypes': True,
        'proto_temperature': 1.0, 'proto_ema_decay': 0.999,
        'batch_size': 8, 'weight_decay': 1e-3, 'gradient_clip': 1.0,
        'phase1_epochs': 80, 'phase1_lr': 1e-3, 'phase1_sam_rho': 0.05,
        'phase2_max_epochs': 200, 'phase2_lr': 3e-4, 'phase2_sam_rho': 0.03,
        'early_stop_patience': 35, 'swa_start': 60,
        'label_smoothing': 0.05, 'dropout': 0.2,
        'consistency_weight': 0.1,
        'use_threshold_opt': True,
        'fixed_bag_size': False, 'use_mixup': False,
        'ensemble_seeds': [42, 123, 456, 789, 1024],
        'default_folds': 10,
    },
    'MSCV2': {
        'data_path': os.path.join(DATA_DIR, 'MSCV2.mat'),
        'cv_mat_path': None,  # No CV file, use random 10-fold splits
        'dd_format': True, 'dataset': 'MSCV2',
        # Architecture (48-dim features, 23 labels, 591 bags)
        'embed_dim': 64, 'num_heads': 4, 'ff_dim': 128,
        'num_inducing': 4, 'embedding_layers': 2,
        'use_isab': True, 'use_gcn': True, 'use_prototypes': True,
        'proto_temperature': 0.5, 'proto_ema_decay': 0.999,
        'batch_size': 16, 'weight_decay': 5e-4, 'gradient_clip': 1.0,
        'phase1_epochs': 80, 'phase1_lr': 1e-3, 'phase1_sam_rho': 0.05,
        'phase2_max_epochs': 200, 'phase2_lr': 3e-4, 'phase2_sam_rho': 0.03,
        'early_stop_patience': 35, 'swa_start': 60,
        'label_smoothing': 0.05, 'dropout': 0.15,
        'consistency_weight': 0.1,
        'use_threshold_opt': True,
        'fixed_bag_size': False, 'use_mixup': False,
        'ensemble_seeds': [42, 123, 456, 789, 1024],
        'default_folds': 10,
    },
}


# Published baselines for comparison
# Scene/Reuters: Zhou & Zhang (AIJ 2012), same 10-fold CV protocol
# Birdsong/Proteins: CLAM++ (same data files, same protocol)
# Letter/MSCV2: MIMLfast (Huang et al. 2019), Coverage converted from normalized to raw
BASELINES = {
    'scene': {
        'HammingLoss': (0.165, 'lower', 'MIML-kNN'),
        'OneError': (0.310, 'lower', 'MIML-kNN'),
        'Coverage': (0.896, 'lower', 'MIML-kNN'),
        'RankingLoss': (0.159, 'lower', 'MIML-kNN'),
        'AveragePrecision': (0.801, 'higher', 'MIML-kNN'),
    },
    'reuters': {
        'HammingLoss': (0.033, 'lower', 'MimlSvm'),
        'OneError': (0.055, 'lower', 'MimlSvmmi'),
        'Coverage': (0.284, 'lower', 'MimlSvmmi'),
        'RankingLoss': (0.020, 'lower', 'MimlSvmmi'),
        'AveragePrecision': (0.965, 'higher', 'MimlSvmmi'),
    },
    'birdsong': {
        'HammingLoss': (0.105, 'lower', 'CLAM++'),
        'OneError': (0.272, 'lower', 'CLAM++'),
        'Coverage': (3.984, 'lower', 'CLAM++'),
        'RankingLoss': (0.104, 'lower', 'CLAM++'),
        'AveragePrecision': (0.723, 'higher', 'CLAM++'),
    },
    'yeast': {
        'HammingLoss': (0.189, 'lower', 'MIML-kNN'),
        'OneError': (0.214, 'lower', 'MIML-kNN'),
        'Coverage': (6.275, 'lower', 'MIML-kNN'),
        'RankingLoss': (0.163, 'lower', 'MIML-kNN'),
        'AveragePrecision': (0.774, 'higher', 'MIML-kNN'),
    },
    'letter_carroll': {
        'HammingLoss': (0.132, 'lower', 'RankLossSIM'),
        'OneError': (0.119, 'lower', 'MIMLfast'),
        'Coverage': (9.880, 'lower', 'MIMLfast'),  # 0.380*26
        'RankingLoss': (0.130, 'lower', 'MIMLfast'),
        'AveragePrecision': (0.715, 'higher', 'MIMLfast'),
    },
    'letter_frost': {
        'HammingLoss': (0.136, 'lower', 'MIMLfast'),
        'OneError': (0.151, 'lower', 'MIMLfast'),
        'Coverage': (9.672, 'lower', 'RankLossSIM'),  # 0.372*26
        'RankingLoss': (0.134, 'lower', 'MIMLfast'),
        'AveragePrecision': (0.704, 'higher', 'MIMLfast'),
    },
    'MSCV2': {
        'HammingLoss': (0.084, 'lower', 'MIMLSVM'),
        'OneError': (0.295, 'lower', 'MIMLfast'),
        'Coverage': (5.474, 'lower', 'MIMLfast'),  # 0.238*23
        'RankingLoss': (0.107, 'lower', 'RankLossSIM'),
        'AveragePrecision': (0.688, 'higher', 'MIMLfast'),
    },
    'protein_azotobacter': {
        'HammingLoss': (0.073, 'lower', 'CLAM++'),
        'OneError': (0.557, 'lower', 'CLAM++'),
        'Coverage': (13.570, 'lower', 'CLAM++'),
        'RankingLoss': (0.130, 'lower', 'CLAM++'),
        'AveragePrecision': (0.510, 'higher', 'CLAM++'),
    },
    'protein_geobacter': {
        'HammingLoss': (0.086, 'lower', 'CLAM++'),
        'OneError': (0.468, 'lower', 'CLAM++'),
        'Coverage': (8.183, 'lower', 'CLAM++'),
        'RankingLoss': (0.127, 'lower', 'CLAM++'),
        'AveragePrecision': (0.584, 'higher', 'CLAM++'),
    },
    'protein_haloarcula': {
        'HammingLoss': (0.110, 'lower', 'CLAM++'),
        'OneError': (0.560, 'lower', 'CLAM++'),
        'Coverage': (5.963, 'lower', 'CLAM++'),
        'RankingLoss': (0.136, 'lower', 'CLAM++'),
        'AveragePrecision': (0.558, 'higher', 'CLAM++'),
    },
    'protein_pyrococcus': {
        'HammingLoss': (0.075, 'lower', 'CLAM++'),
        'OneError': (0.532, 'lower', 'CLAM++'),
        'Coverage': (14.360, 'lower', 'CLAM++'),
        'RankingLoss': (0.128, 'lower', 'CLAM++'),
        'AveragePrecision': (0.544, 'higher', 'CLAM++'),
    },
}


def compare_to_baselines(results, dataset_name):
    """Compare results to published baselines."""
    if dataset_name not in BASELINES:
        print(f"\n  No published baselines for {dataset_name}.")
        return 0

    baselines = BASELINES[dataset_name]
    beats = 0
    print(f"\n  Comparison to Published Baselines:")
    print(f"  {'Metric':<20} {'Ours':>10} {'Baseline':>10} {'Method':<15} {'Status':<10}")
    print(f"  {'-'*65}")

    for metric, (baseline, direction, method) in baselines.items():
        if metric in results:
            ours = results[metric][0]
            if direction == 'lower':
                won = ours < baseline
            else:
                won = ours > baseline
            status = "BEAT" if won else "miss"
            if won:
                beats += 1
            print(f"  {metric:<20} {ours:>10.4f} {baseline:>10.3f} {method:<15} {status:<10}")

    print(f"\n  Result: {beats}/5 beats")
    return beats


def main():
    parser = argparse.ArgumentParser(description='ProtoMIML Experiments')
    parser.add_argument('dataset', type=str, nargs='?', default='scene',
                        choices=list(CONFIGS.keys()) + ['all'],
                        help='Dataset to run (default: scene)')
    parser.add_argument('--folds', type=int, default=None,
                        help='Number of CV folds (default: dataset-specific)')
    parser.add_argument('--no-ensemble', action='store_true',
                        help='Disable ensemble (single model)')
    parser.add_argument('--no-mixup', action='store_true',
                        help='Disable bag mixup')
    parser.add_argument('--no-isab', action='store_true',
                        help='Disable ISAB set attention')
    parser.add_argument('--no-gcn', action='store_true',
                        help='Disable label GCN')
    parser.add_argument('--no-prototypes', action='store_true',
                        help='Disable prototypical memory bank')
    parser.add_argument('--no-consistency', action='store_true',
                        help='Disable consistency loss')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override data directory')
    args = parser.parse_args()

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    datasets_to_run = list(CONFIGS.keys()) if args.dataset == 'all' else [args.dataset]

    all_results = {}
    for dataset_name in datasets_to_run:
        config = CONFIGS[dataset_name].copy()

        # Override data directory if specified
        if args.data_dir:
            for key in ['data_path', 'cv_mat_path']:
                if config.get(key):
                    filename = os.path.basename(config[key])
                    config[key] = os.path.join(args.data_dir, filename)

        # Override folds
        num_folds = args.folds or config.get('default_folds', 10)

        # Disable options
        if args.no_ensemble:
            config.pop('ensemble_seeds', None)
        if args.no_mixup:
            config['use_mixup'] = False
        if args.no_isab:
            config['use_isab'] = False
        if args.no_gcn:
            config['use_gcn'] = False
        if args.no_prototypes:
            config['use_prototypes'] = False
        if args.no_consistency:
            config['consistency_weight'] = 0.0

        components = []
        if config.get('use_isab', True):
            components.append('ISAB')
        if config.get('use_gcn', True):
            components.append('GCN')
        if config.get('use_prototypes', True):
            components.append('Proto')
        if config.get('consistency_weight', 0) > 0:
            components.append('Consistency')
        if config.get('use_mixup'):
            components.append('Mixup')
        seeds = config.get('ensemble_seeds', [])
        if seeds:
            components.append(f'{len(seeds)}-seed')

        print(f"\n{'#'*70}")
        print(f"# ProtoMIML: {dataset_name} ({num_folds}-fold CV)")
        print(f"# Components: SAM + {', '.join(components) if components else 'base'}")
        print(f"{'#'*70}")

        torch.manual_seed(42)
        np.random.seed(42)

        config['dataset_name'] = dataset_name
        results, per_fold = run_cv(config, device, num_folds=num_folds)
        all_results[dataset_name] = results
        compare_to_baselines(results, dataset_name)

    # Save results
    os.makedirs('results', exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'results/proto_miml_{ts}.json'
    with open(output_path, 'w') as f:
        json.dump({
            name: {k: [float(v[0]), float(v[1])] for k, v in res.items()}
            for name, res in all_results.items()
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    print(f"\n{'='*70}")
    print("PROTO-MIML SUMMARY")
    print(f"{'='*70}")
    for name, res in all_results.items():
        print(f"\n  {name}:")
        for metric in ['HammingLoss', 'OneError', 'Coverage', 'RankingLoss', 'AveragePrecision']:
            if metric in res:
                print(f"    {metric}: {res[metric][0]:.4f} +/- {res[metric][1]:.4f}")


if __name__ == '__main__':
    main()
