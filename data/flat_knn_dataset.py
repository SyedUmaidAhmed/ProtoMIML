"""Leak-free k-NN bagging for the flat multi-label Yeast dataset.

yeast.mat is NOT a native MIML benchmark.  It is the single-instance Yeast
gene-function dataset (2,417 genes x 103 features, 14 labels; Elisseeff &
Weston, 2001) turned into bags by taking each sample plus its 4 nearest
neighbours.  The shipped file fits the scaler and the neighbour index on the
WHOLE dataset before any split, so test samples sit inside training bags
(transductive leakage).

This loader keeps only the flat matrix (instance 0 of every stored bag is the
sample itself) and rebuilds the bags inside each fold:

  * standardisation statistics come from the fold's training rows only;
  * the neighbour pool is the fitting split only (training minus validation);
  * a fitting bag is the sample plus its k nearest OTHER fitting samples;
  * a validation or test bag is the sample plus its k nearest fitting samples.

k = 0 gives singleton bags, i.e. ordinary single-instance multi-label learning.

Usage contract:
    ds.normalize_features(train_idx)   # always from the raw matrix
    ds.prepare_fold(fit_idx)           # rebuilds every bag; call before reading ds.bags
"""

import numpy as np
import torch
from scipy.io import loadmat

from data.dataset import MIMLDatasetCV


class FlatKNNBagDataset(MIMLDatasetCV):

    def __init__(self, mat_path, bag_key='bags', label_key='labels', k_neighbors=4):
        # Skip MIMLDatasetCV.__init__ on purpose: the stored bags are leaky.
        torch.utils.data.Dataset.__init__(self)
        mat = loadmat(mat_path)
        raw_bags = mat[bag_key]
        self.k_neighbors = int(k_neighbors)
        self.X_raw = np.stack([np.asarray(raw_bags[i, 0][0], dtype=np.float64)
                               for i in range(raw_bags.shape[0])])
        y = (np.asarray(mat[label_key]) > 0).astype(np.float32)
        self.labels = [torch.tensor(row) for row in y]
        self.num_labels = y.shape[1]
        self.input_dim = self.X_raw.shape[1]
        self.X = self.X_raw.astype(np.float32)
        self.bags = [torch.tensor(self.X[i:i + 1]) for i in range(len(self.X))]
        self.fold_ready = False

    def normalize_features(self, train_indices):
        tr = self.X_raw[np.asarray(train_indices)]
        self.feat_mean = tr.mean(axis=0)
        std = tr.std(axis=0, ddof=1)
        self.feat_std = np.where(std < 1e-6, 1.0, std)
        self.X = ((self.X_raw - self.feat_mean) / self.feat_std).astype(np.float32)
        self.bags = [torch.tensor(self.X[i:i + 1]) for i in range(len(self.X))]
        self.fold_ready = False

    def prepare_fold(self, fit_indices):
        k = self.k_neighbors
        if k == 0:
            self.bags = [torch.tensor(self.X[i:i + 1]) for i in range(len(self.X))]
            self.fold_ready = True
            return
        fit = np.asarray(fit_indices)
        # Exact Euclidean k+1 nearest fitting samples, nearest first.  (A plain
        # NumPy search: sklearn's NearestNeighbors segfaults when its OpenMP
        # runtime meets torch's in the same process on macOS.)
        X, F = self.X.astype(np.float64), self.X[fit].astype(np.float64)
        d = (X ** 2).sum(1)[:, None] + (F ** 2).sum(1)[None, :] - 2.0 * X @ F.T
        part = np.argpartition(d, k, axis=1)[:, :k + 1]
        order = np.argsort(np.take_along_axis(d, part, axis=1), axis=1, kind='stable')
        nbr = fit[np.take_along_axis(part, order, axis=1)]
        in_fit = np.zeros(len(self.X), dtype=bool)
        in_fit[fit] = True
        bags = []
        for i in range(len(self.X)):
            cand = nbr[i]
            if in_fit[i]:
                cand = cand[cand != i][:k] if (cand == i).any() else cand[:k]
            else:
                cand = cand[:k]
            bags.append(torch.tensor(self.X[np.concatenate([[i], cand])]))
        self.bags = bags
        self.fold_ready = True

    def __getitem__(self, idx):
        if not self.fold_ready:
            raise RuntimeError("FlatKNNBagDataset: call prepare_fold(fit_idx) first")
        return self.bags[idx], self.labels[idx]
