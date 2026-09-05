#!/usr/bin/env python3
"""Train one Scene fold and dump REAL cross-attention weights for the paper.

Produces results/attention_scene_fold0.json with, for a handful of test bags:
  - the (L x N) cross-attention weight matrix,
  - true labels, predicted probabilities, and the fold's optimized thresholds.

The paper's interpretability discussion (Section 8.4) must cite measured
attention, not a conceptual illustration; this supplies the measurement.

    python extract_attention.py
"""
import copy
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import MIMLDatasetCV, collate_fn, get_cv_splits
from metrics.miml_metrics import optimize_thresholds, AveragePrecision as AP_fn
from training.train_cv import create_model, _init_model
from training.trainer import ProtoMIMLTrainer
from run_experiment import CONFIGS

SCENE_LABELS = ['desert', 'mountains', 'sea', 'sunset', 'trees']
N_BAGS = 6  # how many test bags to record


def main():
    config = CONFIGS['scene'].copy()
    config.pop('ensemble_seeds', None)
    config['dataset_name'] = 'scene'

    device = (torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    dataset = MIMLDatasetCV(mat_path=config['data_path'],
                            bag_key='bags', label_key='labels')
    config['input_dim'] = dataset.input_dim
    config['num_labels'] = dataset.num_labels

    splits = get_cv_splits(config['cv_mat_path'], len(dataset), 10)
    train_idx, test_idx = splits[0]

    ds = copy.deepcopy(dataset)
    ds.normalize_features(train_idx)

    # Same 90/10 carve-out as train_cv.run_fold, fold_idx = 0
    np.random.seed(0)
    arr = np.array(train_idx); np.random.shuffle(arr)
    val_size = max(len(arr) // 10, 1)
    val_idx, actual_train = arr[:val_size].tolist(), arr[val_size:].tolist()

    sampler = ds.get_balanced_sampler(actual_train)
    tl = DataLoader(Subset(ds, actual_train), batch_size=config['batch_size'],
                    sampler=sampler, collate_fn=collate_fn)
    vl = DataLoader(Subset(ds, val_idx), batch_size=config['batch_size'],
                    shuffle=False, collate_fn=collate_fn)
    te = DataLoader(Subset(ds, test_idx), batch_size=config['batch_size'],
                    shuffle=False, collate_fn=collate_fn)

    torch.manual_seed(42); np.random.seed(42)
    model = create_model(config, device)
    _init_model(model, config, ds, actual_train, device)
    trainer = ProtoMIMLTrainer(model, config, device)
    trainer.train_phase1(tl, vl, num_epochs=config.get('phase1_epochs', 80),
                         lr=config.get('phase1_lr', 1e-3))
    trainer.train_phase2(tl, vl, max_epochs=config.get('phase2_max_epochs', 200),
                         lr=config.get('phase2_lr', 3e-4),
                         patience=config.get('early_stop_patience', 35))

    # Thresholds from validation, as in the pipeline
    val_labels, val_raw, val_fused = trainer.get_predictions(vl)
    probs_src = val_fused if AP_fn(val_labels, val_fused) >= AP_fn(val_labels, val_raw) else val_raw
    thresholds = optimize_thresholds(val_labels, probs_src, metric='hamming')

    model.eval()
    records = []
    with torch.no_grad():
        for bags, labels, mask in te:
            out = model(bags.to(device), mask.to(device))
            probs = torch.sigmoid(out['fused_logits']).cpu().numpy()
            attn = out['attn_weights'].cpu().numpy()  # (B, L, N)
            for b in range(bags.shape[0]):
                if len(records) >= N_BAGS:
                    break
                # Prefer multi-label bags: they make the per-label attention story visible
                if labels[b].sum() < 2 and len(records) < N_BAGS - 2:
                    continue
                records.append({
                    'true_labels': labels[b].numpy().astype(int).tolist(),
                    'probs': [round(float(x), 4) for x in probs[b]],
                    'attention': [[round(float(x), 4) for x in row] for row in attn[b]],
                })
            if len(records) >= N_BAGS:
                break

    out_path = 'results/attention_scene_fold0.json'
    with open(out_path, 'w') as f:
        json.dump({'label_names': SCENE_LABELS,
                   'thresholds': [round(float(t), 3) for t in thresholds],
                   'bags': records}, f, indent=1)
    print(f"Saved {len(records)} bags to {out_path}")


if __name__ == '__main__':
    main()
