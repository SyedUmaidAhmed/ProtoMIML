"""Set Attention Encoder using Induced Set Attention Blocks (ISAB).

Implements the Set Transformer architecture (Lee et al., 2019) for
permutation-equivariant processing of instance sets.
"""

import torch
import torch.nn as nn
import math


class MultiheadAttentionBlock(nn.Module):
    """Pre-norm multi-head attention block with FFN.

    MAB(X, Y) = LayerNorm(H + FFN(H))
    where H = LayerNorm(X + MHA(X, Y, Y))
    """

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.15):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, X, Y, key_padding_mask=None):
        """
        Args:
            X: queries (B, N_x, D)
            Y: keys/values (B, N_y, D)
            key_padding_mask: (B, N_y), True where padded
        Returns:
            (B, N_x, D)
        """
        X_norm = self.norm1(X)
        Y_norm = self.norm1(Y) if Y is not X else X_norm
        attn_out, _ = self.mha(X_norm, Y_norm, Y_norm, key_padding_mask=key_padding_mask)
        H = X + attn_out
        H = H + self.ffn(self.norm2(H))
        return H


class InducedSetAttentionBlock(nn.Module):
    """ISAB: Induced Set Attention Block.

    Uses M learnable inducing points to reduce O(N^2) self-attention to O(NM).
    ISAB(X) = MAB(X, H) where H = MAB(I, X) and I are inducing points.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, num_inducing, dropout=0.15):
        super().__init__()
        self.inducing_points = nn.Parameter(
            torch.randn(1, num_inducing, embed_dim) * (1.0 / math.sqrt(embed_dim))
        )
        self.mab1 = MultiheadAttentionBlock(embed_dim, num_heads, ff_dim, dropout)
        self.mab2 = MultiheadAttentionBlock(embed_dim, num_heads, ff_dim, dropout)

    def forward(self, X, key_padding_mask=None):
        """
        Args:
            X: (B, N, D)
            key_padding_mask: (B, N), True where padded
        Returns:
            (B, N, D)
        """
        B = X.size(0)
        I = self.inducing_points.expand(B, -1, -1)
        H = self.mab1(I, X, key_padding_mask=key_padding_mask)
        return self.mab2(X, H)


class SetAttentionEncoder(nn.Module):
    """Stack of ISAB blocks for permutation-equivariant instance encoding."""

    def __init__(self, embed_dim, num_heads=4, ff_dim=128, num_inducing=4,
                 num_layers=1, dropout=0.15):
        super().__init__()
        self.layers = nn.ModuleList([
            InducedSetAttentionBlock(embed_dim, num_heads, ff_dim, num_inducing, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, X, key_padding_mask=None):
        """
        Args:
            X: (B, N, D)
            key_padding_mask: (B, N), True where padded
        Returns:
            (B, N, D)
        """
        for layer in self.layers:
            X = layer(X, key_padding_mask=key_padding_mask)
        return X
