"""Label GCN: 2-layer Graph Convolutional Network over label co-occurrence graph.

Captures higher-order label dependencies by message passing on the label graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """Single GCN layer: H' = ReLU(A_hat @ H @ W) + residual."""

    def __init__(self, in_dim, out_dim, dropout=0.15):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = (in_dim == out_dim)

    def forward(self, H, A_hat):
        """
        Args:
            H: (B, L, D) node features
            A_hat: (L, L) normalized adjacency matrix
        Returns:
            (B, L, D) updated features
        """
        # GCN propagation: A_hat @ H @ W
        support = self.weight(H)  # (B, L, D_out)
        out = torch.matmul(A_hat, support)  # (B, L, D_out)
        out = self.norm(out)
        out = F.gelu(out)
        out = self.dropout(out)

        if self.use_residual:
            out = out + H
        return out


class LabelGCN(nn.Module):
    """2-layer GCN over label co-occurrence graph with residual connection.

    Usage:
        gcn = LabelGCN(embed_dim=64, hidden_dim=64, num_labels=5)
        gcn.init_from_cooccurrence(co_matrix)  # co_matrix from dataset
        refined = gcn(label_repr)  # (B, L, D) -> (B, L, D)
    """

    def __init__(self, embed_dim, hidden_dim=64, num_labels=5, dropout=0.15):
        super().__init__()
        self.gcn1 = GCNLayer(embed_dim, hidden_dim, dropout)
        self.gcn2 = GCNLayer(hidden_dim, embed_dim, dropout)

        # Normalized adjacency as buffer
        self.register_buffer('A_hat', torch.eye(num_labels))

    def init_from_cooccurrence(self, co_matrix):
        """Initialize adjacency from label co-occurrence matrix.

        Args:
            co_matrix: (L, L) tensor, C[i,j] = P(label_j=1 | label_i=1)
        """
        A = co_matrix.clone()
        # Add self-loops
        A = A + torch.eye(A.size(0), device=A.device)
        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        D = A.sum(dim=1)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.clamp(min=1e-8)))
        A_hat = D_inv_sqrt @ A @ D_inv_sqrt
        self.A_hat = A_hat

    def forward(self, label_repr):
        """
        Args:
            label_repr: (B, L, D)
        Returns:
            (B, L, D) refined label representations
        """
        H = self.gcn1(label_repr, self.A_hat)
        H = self.gcn2(H, self.A_hat)
        # Global residual
        return label_repr + H
