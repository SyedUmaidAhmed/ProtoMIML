"""Cross-Attention Label Pooling.

Labels as queries, instances as keys/values via multi-head cross-attention.
More expressive than gated attention for label-specific instance aggregation.
"""

import torch
import torch.nn as nn
import math


class CrossAttentionLabelPooling(nn.Module):
    """Pool instance embeddings into per-label representations via cross-attention.

    Each label has a learnable query embedding. These queries attend to the
    instance embeddings (keys/values) to produce label-specific bag representations.
    """

    def __init__(self, num_labels, embed_dim, num_heads=4, ff_dim=128, dropout=0.15):
        super().__init__()
        self.num_labels = num_labels
        self.embed_dim = embed_dim

        # Learnable label query embeddings
        self.label_queries = nn.Parameter(
            torch.randn(1, num_labels, embed_dim) * (1.0 / math.sqrt(embed_dim))
        )

        # Cross-attention: Q=label queries, K=V=instances
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # FFN after cross-attention
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, instance_embeddings, key_padding_mask=None):
        """
        Args:
            instance_embeddings: (B, N, D) encoded instances
            key_padding_mask: (B, N), True where padded
        Returns:
            label_repr: (B, L, D) per-label bag representations
            attn_weights: (B, L, N) attention weights
        """
        B = instance_embeddings.size(0)
        Q = self.label_queries.expand(B, -1, -1)

        Q_norm = self.norm1(Q)
        K_norm = self.norm1(instance_embeddings)

        attn_out, attn_weights = self.cross_attn(
            Q_norm, K_norm, K_norm,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )

        # Residual + FFN
        H = Q + attn_out
        label_repr = H + self.ffn(self.norm2(H))

        return label_repr, attn_weights
