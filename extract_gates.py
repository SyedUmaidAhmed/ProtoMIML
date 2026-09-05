#!/usr/bin/env python3
"""Extract learned fusion gate values sigma(g_l) from ProtoMIML.

Runs 3-fold x 1-seed training per dataset and records the converged
per-label gate values sigma(g_l) = sigmoid(model.proto_bank.gate).

Usage:
    python extract_gates.py scene
    python extract_gates.py all          # all 5 main datasets
    python extract_gates.py scene reuters letter_carroll letter_frost MSCV2
"""

import torch
import numpy as np
import sys
import os
import copy
import json
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import MIMLDatasetCV, collate_fn, get_cv_splits
from models.proto_miml import ProtoMIML
from training.trainer import ProtoMIMLTrainer
from training.train_cv import create_model, _init_model
from run_experiment import CONFIGS, DATA_DIR

LABEL_NAMES = {
    'scene':    ['desert', 'mountains', 'sea', 'sunset', 'trees'],
    'reuters':  ['C', 'E', 'G', 'M', 'T'],   # top-5 Reuters categories
    'letter_carroll': [chr(ord('a') + i) for i in range(26)],
    'letter_frost':   [chr(ord('a') + i) for i in range(26)],
    'MSCV2': [
        'building', 'car', 'person', 'sky', 'tree',
        'water', 'road', 'mountain', 'animal', 'boat',
        'flower', 'grass', 'sign', 'bird', 'window',
        'rock', 'snow', 'door', 'bus', 'train',
        'airplane', 'bicycle', 'bench'
    ],
}

# Full Reuters label list (18 labels)
REUTERS_LABELS = ['C15', 'C151', 'C152', 'C181', 'CCAT', 'E11', 'ECAT',
                  'G15', 'GCAT', 'M11', 'M12', 'M13', 'M14', 'MCAT',
                  'T1', 'T14', 'T148', 'TCAT']


def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def _run_gate_convergence(model, train_loader, config, device, n_epochs=80):
    """Short training run using only fused BCE to let gates converge.

    After train_phase2 restores best_state (gates reset to 0), we run this
    continuation with full fused loss and no early stopping so gates reach
    their equilibrium values.
    """
    import torch.optim as optim
    from training.sam_optimizer import SAM

    lr = config.get('phase2_lr', 3e-4) * 0.5  # half LR for fine-tuning
    rho = config.get('phase2_sam_rho', 0.03)
    label_smoothing = config.get('label_smoothing', 0.05)

    optimizer = SAM(
        model.parameters(), optim.AdamW, rho=rho,
        lr=lr, weight_decay=config.get('weight_decay', 5e-4)
    )

    def smooth(lbl):
        return lbl * (1 - label_smoothing) + 0.5 * label_smoothing

    for epoch in range(n_epochs):
        model.train()
        for bags, labels, mask in train_loader:
            bags, labels, mask = bags.to(device), labels.to(device), mask.to(device)
            sl = smooth(labels)

            result = model(bags, mask=mask, bag_labels=labels)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                result['fused_logits'], sl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.first_step(zero_grad=True)

            model.train()
            result2 = model(bags, mask=mask, bag_labels=labels)
            loss2 = torch.nn.functional.binary_cross_entropy_with_logits(
                result2['fused_logits'], sl)
            loss2.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.second_step(zero_grad=True)

        if (epoch + 1) % 20 == 0:
            g = torch.sigmoid(model.proto_bank.gate).detach().cpu().numpy()
            print(f"    Phase2B ep {epoch+1}/{n_epochs}: gates={np.round(g, 3).tolist()}",
                  flush=True)


def extract_gates_for_dataset(dataset_name, config, n_folds=3, seed=42):
    """Train on n_folds with one seed and collect gate values at convergence."""
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}  ({n_folds} folds, seed={seed})")
    print(f"{'='*60}")

    # Load data (matches run_cv convention)
    dataset = MIMLDatasetCV(
        mat_path=config['data_path'],
        bag_key=config.get('bag_key', 'bags'),
        label_key=config.get('label_key', 'labels'),
        dd_format=config.get('dd_format', False),
    )
    config['input_dim'] = dataset.input_dim
    config['num_labels'] = dataset.num_labels

    # CV splits — same logic as run_cv
    if config.get('cv_mat_path'):
        all_splits = get_cv_splits(config['cv_mat_path'], len(dataset), 10)
    else:
        indices = np.arange(len(dataset))
        np.random.seed(seed)
        np.random.shuffle(indices)
        k = 10
        fold_size = len(dataset) // k
        all_splits = []
        for i in range(k):
            t_idx = indices[i * fold_size:(i + 1) * fold_size].tolist()
            tr_idx = np.concatenate([indices[:i * fold_size],
                                     indices[(i + 1) * fold_size:]]).tolist()
            all_splits.append((tr_idx, t_idx))

    num_labels = dataset.num_labels

    all_gates = []  # List of (num_labels,) arrays

    for fold_idx in range(n_folds):
        train_idx, test_idx = all_splits[fold_idx]
        ds = copy.deepcopy(dataset)
        ds.normalize_features(train_idx)

        np.random.seed(fold_idx)
        train_arr = np.array(train_idx)
        np.random.shuffle(train_arr)
        val_size = max(len(train_arr) // 10, 1)
        val_idx = train_arr[:val_size].tolist()
        actual_train_idx = train_arr[val_size:].tolist()

        train_loader = DataLoader(
            Subset(ds, actual_train_idx), batch_size=config['batch_size'],
            shuffle=True, collate_fn=collate_fn, num_workers=0, drop_last=False)
        val_loader = DataLoader(
            Subset(ds, val_idx), batch_size=config['batch_size'],
            shuffle=False, collate_fn=collate_fn, num_workers=0)

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = create_model(config, device)
        _init_model(model, config, ds, actual_train_idx, device)
        trainer = ProtoMIMLTrainer(model, config, device)

        print(f"\n  Fold {fold_idx+1}/{n_folds} — Phase 1 ...", flush=True)
        trainer.train_phase1(
            train_loader, val_loader,
            num_epochs=config.get('phase1_epochs', 80),
            lr=config.get('phase1_lr', 1e-3))

        print(f"  Fold {fold_idx+1}/{n_folds} — Phase 2 ...", flush=True)
        trainer.train_phase2(
            train_loader, val_loader,
            max_epochs=config.get('phase2_max_epochs', 200),
            lr=config.get('phase2_lr', 3e-4),
            patience=config.get('early_stop_patience', 35))

        # --- Phase 2B: gate convergence run (80 epochs, full fused loss, no early stop) ---
        # After train_phase2 restores best_state, gates are at 0 (sigma=0.5).
        # Run a short continuation with full fused loss so gates can find their equilibrium.
        print(f"  Fold {fold_idx+1}/{n_folds} — Phase 2B (gate convergence, 80 ep) ...",
              flush=True)
        _run_gate_convergence(model, train_loader, config, device, n_epochs=80)

        # Extract gate values
        gate_raw = model.proto_bank.gate.detach().cpu()
        gate_sigmoid = torch.sigmoid(gate_raw).numpy()
        all_gates.append(gate_sigmoid)
        print(f"  Gates σ(g_l): mean={gate_sigmoid.mean():.3f}  "
              f"min={gate_sigmoid.min():.3f}  max={gate_sigmoid.max():.3f}",
              flush=True)
        print(f"  Per-label: {np.round(gate_sigmoid, 3).tolist()}", flush=True)

    gates_array = np.stack(all_gates)  # (n_folds, num_labels)
    mean_gates = gates_array.mean(axis=0)
    std_gates = gates_array.std(axis=0)

    print(f"\n--- {dataset_name} Gate Summary ({n_folds} folds) ---")
    print(f"Mean σ(g_l): {mean_gates.mean():.3f} ± {mean_gates.std():.3f}")
    print(f"Proto-dominant labels (σ < 0.45): "
          f"{int((mean_gates < 0.45).sum())}/{num_labels}")
    print(f"Balanced labels      (0.45–0.55): "
          f"{int(((mean_gates >= 0.45) & (mean_gates <= 0.55)).sum())}/{num_labels}")
    print(f"Param-dominant labels (σ > 0.55): "
          f"{int((mean_gates > 0.55).sum())}/{num_labels}")

    return {
        'dataset': dataset_name,
        'num_labels': num_labels,
        'mean_gates': mean_gates.tolist(),
        'std_gates': std_gates.tolist(),
        'global_mean': float(mean_gates.mean()),
        'global_std': float(mean_gates.std()),
        'n_proto_dominant': int((mean_gates < 0.45).sum()),
        'n_balanced': int(((mean_gates >= 0.45) & (mean_gates <= 0.55)).sum()),
        'n_param_dominant': int((mean_gates > 0.55).sum()),
        'folds_raw': gates_array.tolist(),
    }


def main():
    datasets_to_run = sys.argv[1:] if len(sys.argv) > 1 else ['scene']
    if datasets_to_run == ['all']:
        datasets_to_run = ['scene', 'reuters', 'letter_carroll', 'letter_frost', 'MSCV2']

    results = {}
    for ds_name in datasets_to_run:
        if ds_name not in CONFIGS:
            print(f"Unknown dataset: {ds_name}. Skipping.")
            continue
        config = dict(CONFIGS[ds_name])
        # Override for fast extraction: 1 seed, reduced folds
        config['ensemble_seeds'] = None
        try:
            # Gate extraction schedule:
            # Phase 1: 50 epochs (representation warm-up)
            # Phase 2: 160 epochs, patience=100 (must pass ramp@70 + full fusion@80+)
            # Gates only converge after Phase2 epoch 70+ when full fused loss is active
            config['phase1_epochs'] = 50
            config['phase2_max_epochs'] = 160
            config['early_stop_patience'] = 100   # disable premature stopping
            config['swa_start'] = 130
            result = extract_gates_for_dataset(ds_name, config, n_folds=3, seed=42)
            results[ds_name] = result
        except Exception as e:
            print(f"ERROR on {ds_name}: {e}")
            import traceback; traceback.print_exc()

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results', 'gate_values.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Merge with any existing file so a partial run never clobbers other datasets
    if os.path.exists(out_path):
        with open(out_path) as f:
            merged = json.load(f)
    else:
        merged = {}
    merged.update(results)
    with open(out_path, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f"\nGate values saved to: {out_path}")

    # Print LaTeX-ready summary table
    print("\n\n=== LaTeX-ready gate summary ===")
    for ds_name, r in results.items():
        mg = r['global_mean']
        sg = r['global_std']
        np_dom = r['n_proto_dominant']
        bal = r['n_balanced']
        pa_dom = r['n_param_dominant']
        L = r['num_labels']
        print(f"{ds_name:20s}: mean σ = {mg:.3f}±{sg:.3f}  "
              f"proto-dom={np_dom}/{L}  balanced={bal}/{L}  param-dom={pa_dom}/{L}")


if __name__ == '__main__':
    main()
