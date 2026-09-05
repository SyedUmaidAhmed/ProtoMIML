#!/usr/bin/env python3
"""Run all ablation experiments for ProtoMIML EAAI paper.

Saves results to results/ablation_{variant}_{dataset}_5folds.json
"""
import torch
import numpy as np
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from training.train_cv import run_cv
from run_experiment import CONFIGS

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'MIML_CLAM_Release', '..', 'VIT_MIL', 'Code', 'MIML-SELFVIT')

ABLATION_VARIANTS = {
    # Matched control: the FULL model run under the identical protocol used for
    # every ablation row (5-fold CV, single seed, no ensemble).  Without this the
    # ablation deltas conflate component removal with the loss of the 10-fold
    # 5-seed ensemble used for the headline results.
    'full':           {},
    # Freezes the fusion gate at sigmoid(0) = 0.5, i.e. a fixed equal-weight
    # average of the parametric and prototype pathways.  Isolates whether the
    # LEARNED per-label gate contributes anything over a constant 0.5.
    'fixed_gate':     {'learnable_gate': False},
    # SAM ablation: rho=0 zeroes the adversarial perturbation, reducing SAM to
    # plain AdamW (with the double forward's cost but not its effect).
    'no_sam':         {'phase1_sam_rho': 0.0, 'phase2_sam_rho': 0.0},
    'no_isab':        {'use_isab': False},
    'no_gcn':         {'use_gcn': False},
    'no_prototypes':  {'use_prototypes': False},
    'no_consistency': {'consistency_weight': 0.0},
    'no_mixup':       {'use_mixup': False},
}

DATASETS = ['scene', 'reuters', 'letter_carroll', 'letter_frost', 'MSCV2']
FOLDS = 5


def run_single_ablation(dataset_name, variant_name, overrides):
    config = CONFIGS[dataset_name].copy()
    # Single seed, no ensemble for speed
    config.pop('ensemble_seeds', None)
    # Apply ablation overrides
    config.update(overrides)
    config['dataset_name'] = dataset_name

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"\n{'='*60}")
    print(f"ABLATION: {variant_name} on {dataset_name} ({FOLDS}-fold)")
    print(f"{'='*60}")
    torch.manual_seed(42)
    np.random.seed(42)

    results, per_fold = run_cv(config, device, num_folds=FOLDS)

    os.makedirs('results', exist_ok=True)
    save_path = f'results/ablation_{variant_name}_{dataset_name}_{FOLDS}folds.json'
    with open(save_path, 'w') as f:
        json.dump({
            k: {'mean': float(v[0]), 'std': float(v[1])}
            for k, v in results.items()
        }, f, indent=2)
    print(f"Saved: {save_path}")

    print("\n  Results:")
    for metric in ['HammingLoss', 'OneError', 'Coverage', 'RankingLoss', 'AveragePrecision']:
        if metric in results:
            print(f"    {metric}: {results[metric][0]:.4f} ± {results[metric][1]:.4f}")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='all', choices=DATASETS + ['all'])
    parser.add_argument('--variant', default='all', choices=list(ABLATION_VARIANTS.keys()) + ['all'])
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    variants = ABLATION_VARIANTS if args.variant == 'all' else {args.variant: ABLATION_VARIANTS[args.variant]}

    # Skip no_mixup for reuters/letter/MSCV2 (mixup already disabled)
    all_results = {}
    for dataset in datasets:
        all_results[dataset] = {}
        for vname, overrides in variants.items():
            # no_mixup is only meaningful for scene (where mixup is enabled)
            if vname == 'no_mixup' and dataset != 'scene':
                print(f"Skipping {vname} for {dataset} (mixup not used)")
                continue
            try:
                res = run_single_ablation(dataset, vname, overrides)
                all_results[dataset][vname] = res
            except Exception as e:
                print(f"ERROR on {dataset}/{vname}: {e}")
                import traceback; traceback.print_exc()

    print("\n\nALL ABLATIONS COMPLETE")
    print("="*60)
    for d, variants_res in all_results.items():
        print(f"\n{d}:")
        for v, res in variants_res.items():
            ap = res.get('AveragePrecision', (0, 0))
            hl = res.get('HammingLoss', (0, 0))
            print(f"  {v}: AP={ap[0]:.4f}, HL={hl[0]:.4f}")


if __name__ == '__main__':
    main()
