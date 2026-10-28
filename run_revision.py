#!/usr/bin/env python3
"""Revision experiments (KAIS round 1): one harness, every prediction archived.

Unlike run_experiment.py, which keeps only the five fold metrics, this runner
saves the per-seed validation AND test probabilities of all three pathways
(parametric, prototype-only, fused) for every fold.  Every post-hoc question
the reviewers raised -- which pathway to deploy, how thresholds are chosen,
how much the result depends on the 13-15-bag validation split -- can then be
answered from the same trained models by revision_analysis.py, instead of by
re-training.

Models (all trained by the SAME ProtoMIMLTrainer recipe: two-phase SAM,
SWA, label smoothing, consistency, mixup where configured, 5-seed ensemble,
per-label Hamming thresholds):
    protomiml          the full model
    protomiml_noproto  prototype bank removed (purely parametric), same recipe
    protomiml_nosam    SAM radius set to 0 (plain AdamW), everything else identical
    protomiml_fullfuse          early stopping / checkpoint selection only after the
                                Phase-2 fusion ramp completes (epoch 70), so every
                                deployed checkpoint has trained the prototype pathway
    protomiml_noproto_fullfuse  the same schedule without the prototype memory (control)
    abmil              ABMIL, shared gated attention   (Ilse et al., 2018)
    abmil_per_label    ABMIL, one gated attention head per label

Protocols:
    standard   the paper's protocol: outer folds, 90/10 train/val carve-out
               seeded by fold index, 5 seeds trained on the same carve-out
    cvens      cross-validated ensemble: the 5 members are trained on the 5
               inner folds of the outer training partition, so every training
               bag is held out exactly once; early stopping uses each member's
               own held-out fold, while pathway selection and thresholds use
               the pooled out-of-fold predictions (the WHOLE training partition,
               ~10x the standard validation set).  Same compute as standard.
    holdout    the protocol of the published Scene/Reuters baselines: repeated
               random 1500/500 train/test splits.

    python run_revision.py --model protomiml --dataset letter_frost
    python run_revision.py --model abmil --dataset all
    python run_revision.py --model protomiml --protocol cvens --dataset MSCV2
    python run_revision.py --model protomiml --dataset letter_frost --val-seed 1
    python run_revision.py --model protomiml --protocol holdout --dataset scene --repeats 10

Writes results/revision/<tag>/fold_<k>.npz (resumable: finished folds are
skipped) plus run_config.json.
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
from data.flat_knn_dataset import FlatKNNBagDataset
from metrics.miml_metrics import AveragePrecision as AP_fn
from models.abmil import ABMIL
from run_experiment import CONFIGS
from training.train_cv import create_model, _init_model
from training.trainer import ProtoMIMLTrainer

MODELS = ['protomiml', 'protomiml_noproto', 'protomiml_nosam', 'abmil', 'abmil_per_label',
          'protomiml_fullfuse', 'protomiml_noproto_fullfuse']
PROTOCOLS = ['standard', 'cvens', 'holdout']
DATASETS = ['scene', 'reuters', 'letter_carroll', 'letter_frost', 'MSCV2', 'yeast', 'yeast_si']
# Yeast is flat multi-label data; bags are rebuilt leak-free inside every fold
# (data/flat_knn_dataset.py): 'yeast' = sample + 4 nearest fitting neighbours,
# 'yeast_si' = singleton bags (ordinary single-instance multi-label learning).
FLAT_K = {'yeast': 4, 'yeast_si': 0}
OUT_ROOT = os.path.join('results', 'revision')


class ABMILForTrainer(nn.Module):
    """Expose ABMIL through the ProtoMIML forward contract.

    The trainer reads 'logits' (phase 1) and 'fused_logits' (phase 2,
    consistency).  ABMIL has a single pathway, so both are the same tensor and
    ABMIL receives exactly the optimisation recipe ProtoMIML receives.
    """

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, bags, mask=None, bag_labels=None, update_prototypes=True):
        out = self.net(bags, mask)
        out['fused_logits'] = out['logits']
        return out


def build(model_name, config, dataset, train_idx, device):
    if model_name.startswith('abmil'):
        net = ABMIL(
            input_dim=config['input_dim'], num_labels=config['num_labels'],
            embed_dim=config.get('embed_dim', 64), attn_dim=config.get('embed_dim', 64),
            dropout=config.get('dropout', 0.15),
            per_label_attention=(model_name == 'abmil_per_label'))
        return ABMILForTrainer(net).to(device)
    model = create_model(config, device)
    _init_model(model, config, dataset, train_idx, device)
    return model


@torch.no_grad()
def predict_all(model, loader, device):
    """Labels plus parametric / prototype / fused probabilities over a loader."""
    model.eval()
    ys, raw, proto, fused = [], [], [], []
    for bags, labels, mask in loader:
        bags, mask = bags.to(device), mask.to(device)
        out = model(bags, mask=mask)
        raw.append(torch.sigmoid(out['logits']).cpu().numpy())
        fused.append(torch.sigmoid(out['fused_logits']).cpu().numpy())
        p = out.get('proto_scores', out['logits'])
        proto.append(torch.sigmoid(p).cpu().numpy())
        ys.append(labels.numpy())
    return (np.concatenate(ys), np.concatenate(raw),
            np.concatenate(proto), np.concatenate(fused))


def gate_values(model):
    bank = getattr(model, 'proto_bank', None)
    if bank is None:
        return np.array([])
    return torch.sigmoid(bank.gate.detach()).cpu().numpy()


def loader(dataset, idx, config, sampler=None):
    return DataLoader(Subset(dataset, idx), batch_size=config['batch_size'],
                      sampler=sampler, shuffle=False, collate_fn=collate_fn,
                      num_workers=0)


def train_member(model_name, config, dataset, tr_idx, val_loader, train_loader, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build(model_name, config, dataset, tr_idx, device)
    trainer = ProtoMIMLTrainer(model, config, device)
    trainer.train_phase1(train_loader, val_loader,
                         num_epochs=config.get('phase1_epochs', 80),
                         lr=config.get('phase1_lr', 1e-3))
    trainer.train_phase2(train_loader, val_loader,
                         max_epochs=config.get('phase2_max_epochs', 120),
                         lr=config.get('phase2_lr', 3e-4),
                         patience=config.get('early_stop_patience', 25))
    return model


def run_standard_fold(model_name, fold_idx, dataset_orig, train_idx, test_idx,
                      config, device, carve_seed):
    """Mirror of train_cv._run_fold_ensemble, archiving every prediction."""
    dataset = copy.deepcopy(dataset_orig)
    dataset.normalize_features(train_idx)

    np.random.seed(carve_seed)
    train_arr = np.array(train_idx)
    np.random.shuffle(train_arr)
    val_size = max(len(train_arr) // 10, 1)
    val_idx = train_arr[:val_size].tolist()
    tr_idx = train_arr[val_size:].tolist()
    if hasattr(dataset, 'prepare_fold'):
        dataset.prepare_fold(tr_idx)

    sampler = dataset.get_balanced_sampler(tr_idx)
    train_loader = loader(dataset, tr_idx, config, sampler)
    val_loader = loader(dataset, val_idx, config)
    test_loader = loader(dataset, test_idx, config)

    rec = {k: [] for k in ['val_raw', 'val_proto', 'val_fused',
                           'test_raw', 'test_proto', 'test_fused', 'gate']}
    for seed in config['ensemble_seeds']:
        print(f"  [fold {fold_idx + 1}] seed {seed}", flush=True)
        model = train_member(model_name, config, dataset, tr_idx,
                             val_loader, train_loader, seed, device)
        yv, vr, vp, vf = predict_all(model, val_loader, device)
        yt, tr, tp, tf = predict_all(model, test_loader, device)
        for k, v in zip(rec, [vr, vp, vf, tr, tp, tf, gate_values(model)]):
            rec[k].append(v)
        print(f"    val AP raw={AP_fn(yv, vr):.4f} fused={AP_fn(yv, vf):.4f}", flush=True)

    out = {k: np.stack(v) for k, v in rec.items()}
    out.update(val_labels=yv, test_labels=yt,
               val_idx=np.array(val_idx), test_idx=np.array(test_idx),
               train_idx=np.array(tr_idx))
    return out


def run_cvens_fold(model_name, fold_idx, dataset_orig, train_idx, test_idx,
                   config, device, n_inner=5):
    """Cross-validated ensemble: member k holds out inner fold k."""
    dataset = copy.deepcopy(dataset_orig)
    dataset.normalize_features(train_idx)

    rng = np.random.RandomState(fold_idx)
    parts = np.array_split(rng.permutation(np.array(train_idx)), n_inner)
    test_loader = loader(dataset, test_idx, config)
    seeds = config['ensemble_seeds']

    rec = {k: [] for k in ['oof_raw', 'oof_proto', 'oof_fused', 'oof_labels', 'oof_idx',
                           'test_raw', 'test_proto', 'test_fused', 'gate']}
    for k in range(n_inner):
        val_idx = parts[k].tolist()
        tr_idx = np.concatenate([parts[j] for j in range(n_inner) if j != k]).tolist()
        if hasattr(dataset, 'prepare_fold'):
            dataset.prepare_fold(tr_idx)
        sampler = dataset.get_balanced_sampler(tr_idx)
        train_loader = loader(dataset, tr_idx, config, sampler)
        val_loader = loader(dataset, val_idx, config)
        print(f"  [fold {fold_idx + 1}] inner {k + 1}/{n_inner} seed {seeds[k]}", flush=True)
        model = train_member(model_name, config, dataset, tr_idx,
                             val_loader, train_loader, seeds[k], device)
        yv, vr, vp, vf = predict_all(model, val_loader, device)
        yt, tr, tp, tf = predict_all(model, test_loader, device)
        for key, v in zip(['oof_raw', 'oof_proto', 'oof_fused', 'oof_labels', 'oof_idx',
                           'test_raw', 'test_proto', 'test_fused', 'gate'],
                          [vr, vp, vf, yv, np.array(val_idx), tr, tp, tf, gate_values(model)]):
            rec[key].append(v)

    out = {key: np.concatenate(rec[key]) for key in
           ['oof_raw', 'oof_proto', 'oof_fused', 'oof_labels', 'oof_idx']}
    out['oof_member'] = np.concatenate([np.full(len(p), k) for k, p in enumerate(parts)])
    for key in ['test_raw', 'test_proto', 'test_fused', 'gate']:
        out[key] = np.stack(rec[key])
    out.update(test_labels=yt, test_idx=np.array(test_idx))
    return out


def make_splits(config, dataset, protocol, repeats):
    n = len(dataset)
    if protocol == 'holdout':
        splits = []
        for r in range(repeats):
            perm = np.random.RandomState(r).permutation(n)
            splits.append((perm[:1500].tolist(), perm[1500:].tolist()))
        return splits
    if config.get('cv_mat_path'):
        return get_cv_splits(config['cv_mat_path'], n, 10)
    # identical to training/train_cv.run_cv for datasets without a 10CV file
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // 10
    return [(np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]]).tolist(),
             indices[i * fold_size:(i + 1) * fold_size].tolist()) for i in range(10)]


def tag_for(args):
    tag = f"{args.model}_{args.protocol}_{args.dataset}"
    if args.val_seed:
        tag += f"_valseed{args.val_seed}"
    return tag


def run(args, dataset_name, device):
    args.dataset = dataset_name
    config = CONFIGS['yeast' if dataset_name in FLAT_K else dataset_name].copy()
    config['dataset_name'] = dataset_name
    if args.model in ('protomiml_noproto', 'protomiml_noproto_fullfuse'):
        config['use_prototypes'] = False
    if args.model.endswith('_fullfuse'):
        # only checkpoints after the fusion ramp (Phase-2 epoch 70) are eligible
        config['phase2_select_from'] = 70
    if args.model == 'protomiml_nosam':
        config['phase1_sam_rho'] = 0.0
        config['phase2_sam_rho'] = 0.0
    if args.model.startswith('abmil'):
        config['use_gcn'] = False
    if not config.get('ensemble_seeds'):
        config['ensemble_seeds'] = [42, 123, 456, 789, 1024]
    if args.smoke:
        config.update(phase1_epochs=2, phase2_max_epochs=2)

    if dataset_name in FLAT_K:
        dataset = FlatKNNBagDataset(config['data_path'], k_neighbors=FLAT_K[dataset_name])
    else:
        dataset = MIMLDatasetCV(mat_path=config['data_path'],
                                bag_key=config.get('bag_key', 'bags'),
                                label_key=config.get('label_key', 'labels'),
                                dd_format=config.get('dd_format', False))
    config['input_dim'] = dataset.input_dim
    config['num_labels'] = dataset.num_labels

    out_dir = os.path.join(OUT_ROOT, tag_for(args))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'run_config.json'), 'w') as f:
        json.dump({'model': args.model, 'protocol': args.protocol,
                   'dataset': dataset_name, 'val_seed': args.val_seed,
                   'config': {k: v for k, v in config.items()
                              if isinstance(v, (int, float, str, bool, list, type(None)))}},
                  f, indent=2)

    # Same global seeding as run_experiment.main before run_cv
    torch.manual_seed(42)
    np.random.seed(42)
    splits = make_splits(config, dataset, args.protocol, args.repeats)
    if args.folds:
        splits = splits[:args.folds]

    print(f"{tag_for(args)}: {len(dataset)} bags, {dataset.num_labels} labels, "
          f"{len(splits)} splits -> {out_dir}", flush=True)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        path = os.path.join(out_dir, f'fold_{fold_idx}.npz')
        if os.path.exists(path):
            print(f"  fold {fold_idx + 1} done, skipping", flush=True)
            continue
        if args.protocol == 'cvens':
            rec = run_cvens_fold(args.model, fold_idx, dataset, train_idx, test_idx,
                                 config, device)
        else:
            carve_seed = fold_idx + 1000 * args.val_seed
            rec = run_standard_fold(args.model, fold_idx, dataset, train_idx, test_idx,
                                    config, device, carve_seed)
        np.savez_compressed(path + '.tmp.npz', **rec)
        os.replace(path + '.tmp.npz', path)
        print(f"  [saved {path}]", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='protomiml', choices=MODELS)
    p.add_argument('--protocol', default='standard', choices=PROTOCOLS)
    p.add_argument('--dataset', default='letter_frost', choices=DATASETS + ['all'])
    p.add_argument('--val-seed', type=int, default=0,
                   help='0 reproduces the paper carve-out; k>0 draws a different validation subset')
    p.add_argument('--repeats', type=int, default=10, help='holdout protocol only')
    p.add_argument('--folds', type=int, default=None, help='limit number of splits (smoke tests)')
    p.add_argument('--device', default=None)
    p.add_argument('--threads', type=int, default=None)
    p.add_argument('--smoke', action='store_true', help='2+2 epochs, 2 seeds, separate output dir')
    args = p.parse_args()
    if args.smoke:
        global OUT_ROOT
        OUT_ROOT = os.path.join('results', 'revision_smoke')

    if args.device:
        device = torch.device(args.device)
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    if args.threads:
        torch.set_num_threads(args.threads)

    datasets = DATASETS[:5] if args.dataset == 'all' else [args.dataset]
    for d in datasets:
        run(args, d, device)


if __name__ == '__main__':
    main()
