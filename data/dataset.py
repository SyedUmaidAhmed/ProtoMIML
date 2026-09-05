"""Dataset loader for MIML .mat files.

Supports two formats:
  Standard (bags/labels): scene.mat, reuters_MIML.mat, yeast.mat, birdsong.mat, proteins
    bags: (n_samples, 1) cell array, each cell is (n_instances, features)
    labels: (n_samples, n_labels) matrix with {-1, 1} values

  DD format: letter_carroll.mat, letter_frost.mat, MSCV2.mat
    DD.X: (n_instances, features) - all instances
    DD.XtB: (n_instances, 1) - instance-to-bag mapping
    DD.B: (n_bags, 1) - bag IDs
    DD.YB: (n_bags, n_labels) - bag-level labels
"""

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from scipy.io import loadmat
import numpy as np
import os


class MIMLDatasetCV(Dataset):
    """MIML dataset with 10-fold CV support from 10CV.mat permutation splits."""

    def __init__(self, mat_path, bag_key='bags', label_key='labels', dd_format=False):
        super().__init__()
        self.bags = []
        self.labels = []
        self.num_instances = []

        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"Data file not found: {mat_path}")

        if dd_format:
            self._load_dd_format(mat_path)
        else:
            self._load_from_mat(mat_path, bag_key, label_key)
        self.num_labels = self.labels[0].shape[0]
        self.input_dim = self.bags[0].shape[1]

    def _load_dd_format(self, mat_path):
        """Load DD-format .mat files (letter_carroll, letter_frost, MSCV2)."""
        mat = loadmat(mat_path)
        dd = mat['DD'][0, 0]
        X = dd['X'].astype(np.float32)       # (n_instances, features)
        XtB = dd['XtB'].flatten()             # instance-to-bag mapping
        B = dd['B'].flatten()                 # bag IDs
        YB = dd['YB']                         # (n_bags, n_labels)

        for i, bag_id in enumerate(B):
            inst_mask = XtB == bag_id
            bag = X[inst_mask]
            bag_tensor = torch.tensor(bag, dtype=torch.float32)

            label = YB[i]
            label = (label > 0).astype(np.float32)
            label_tensor = torch.tensor(label, dtype=torch.float32)

            self.bags.append(bag_tensor)
            self.labels.append(label_tensor)
            self.num_instances.append(bag_tensor.shape[0])

    def _load_from_mat(self, mat_path, bag_key, label_key):
        mat = loadmat(mat_path)
        raw_bags = mat[bag_key]
        raw_labels = mat[label_key]

        for i in range(raw_bags.shape[0]):
            bag = raw_bags[i, 0]
            bag_tensor = torch.tensor(bag, dtype=torch.float32)

            label = raw_labels[i]
            label = (label > 0).astype(np.float32)
            label_tensor = torch.tensor(label, dtype=torch.float32)

            self.bags.append(bag_tensor)
            self.labels.append(label_tensor)
            self.num_instances.append(bag_tensor.shape[0])

    def normalize_features(self, train_indices):
        """Standardize features using training set statistics only."""
        all_instances = []
        for i in train_indices:
            all_instances.append(self.bags[i])
        all_instances = torch.cat(all_instances, dim=0)
        self.feat_mean = all_instances.mean(dim=0)
        self.feat_std = all_instances.std(dim=0).clamp(min=1e-6)

        for i in range(len(self.bags)):
            self.bags[i] = (self.bags[i] - self.feat_mean) / self.feat_std

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        return self.bags[idx], self.labels[idx]

    def get_label_cooccurrence(self, indices=None):
        """Compute label co-occurrence matrix: C[i,j] = P(label_j=1 | label_i=1)."""
        if indices is None:
            indices = range(len(self))
        labels = torch.stack([self.labels[i] for i in indices])
        co = labels.T @ labels
        counts = labels.sum(dim=0).clamp(min=1)
        co = co / counts.unsqueeze(1)
        co.fill_diagonal_(0)
        return co

    def get_balanced_sampler(self, indices):
        """Create WeightedRandomSampler that balances label-combination frequency."""
        labels = torch.stack([self.labels[i] for i in indices])
        combo_strings = [''.join(str(int(x)) for x in row) for row in labels.numpy()]
        from collections import Counter
        combo_counts = Counter(combo_strings)
        weights = [1.0 / combo_counts[s] for s in combo_strings]
        weights = torch.tensor(weights, dtype=torch.float64)
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def get_cv_splits(cv_mat_path, num_samples, num_folds=10):
    """Load 10-fold CV splits from 10CV.mat.

    Returns list of (train_indices, test_indices) tuples.
    """
    mat = loadmat(cv_mat_path)

    for key in ['perm', 'permutation', 'indices', 'cv']:
        if key in mat:
            perm = mat[key].flatten() - 1  # MATLAB 1-indexed to 0-indexed
            break
    else:
        data_keys = [k for k in mat.keys() if not k.startswith('__')]
        if data_keys:
            perm = mat[data_keys[0]].flatten() - 1
        else:
            raise ValueError(f"Cannot find permutation data in {cv_mat_path}")

    fold_size = num_samples // num_folds
    splits = []

    for fold in range(num_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size
        test_idx = perm[test_start:test_end].tolist()
        train_idx = np.concatenate([perm[:test_start], perm[test_end:]]).tolist()
        splits.append((train_idx, test_idx))

    return splits


def collate_fn(batch):
    """Collate bags of variable length with padding. mask: 1=padded, 0=real."""
    bags, labels = zip(*batch)
    max_len = max(bag.shape[0] for bag in bags)
    dim = bags[0].shape[1]

    padded_bags = torch.zeros((len(bags), max_len, dim))
    mask = torch.ones((len(bags), max_len))

    for i, bag in enumerate(bags):
        n = bag.shape[0]
        padded_bags[i, :n] = bag
        mask[i, :n] = 0

    labels = torch.stack(labels)
    return padded_bags, labels, mask
