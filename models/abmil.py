"""Attention-based Multiple Instance Learning (ABMIL) baseline for MIML.

Reference implementation of the gated-attention pooling operator of
Ilse et al. (2018), "Attention-based Deep Multiple Instance Learning",
extended to the multi-label setting in the two natural ways:

  shared     -- one attention distribution over instances, shared across all
                labels, producing a single bag embedding that feeds an L-way
                head.  This is ABMIL as published (Section 2.4 of the paper
                describes exactly this: "shared attention across all classes").

  per_label  -- L attention distributions, one per label, producing L
                label-specific bag embeddings.  This is the stronger variant
                and is the direct point of comparison for ProtoMIML's
                cross-attention label pooling.

Both share the same gated attention form:

    a_n = w^T ( tanh(V h_n) (*) sigmoid(U h_n) )

with (*) elementwise, softmax-normalised over the (unpadded) instances.
"""

import torch
import torch.nn as nn


class ABMIL(nn.Module):
    """Gated-attention MIL baseline with shared or per-label attention.

    Args:
        input_dim: raw instance feature dimension
        num_labels: number of labels L
        embed_dim: instance embedding dimension
        attn_dim: hidden dimension of the gated attention branches
        dropout: dropout rate in the feature extractor
        per_label_attention: if True, learn one attention head per label
    """

    def __init__(self, input_dim, num_labels, embed_dim=64, attn_dim=64,
                 dropout=0.15, per_label_attention=False):
        super().__init__()
        self.num_labels = num_labels
        self.embed_dim = embed_dim
        self.per_label_attention = per_label_attention

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Gated attention: tanh branch (V) modulated by sigmoid branch (U)
        self.attention_V = nn.Linear(embed_dim, attn_dim)
        self.attention_U = nn.Linear(embed_dim, attn_dim)

        num_heads = num_labels if per_label_attention else 1
        self.attention_w = nn.Linear(attn_dim, num_heads)

        if per_label_attention:
            # One linear classifier per label, applied to that label's bag embedding
            self.cls_weight = nn.Parameter(torch.empty(num_labels, embed_dim))
            self.cls_bias = nn.Parameter(torch.zeros(num_labels))
            nn.init.xavier_uniform_(self.cls_weight)
        else:
            self.classifier = nn.Linear(embed_dim, num_labels)

    def forward(self, bags, mask=None):
        """
        Args:
            bags: (B, N, D_in) padded instance features
            mask: (B, N) with 1 = padded, 0 = real (matching collate_fn)

        Returns:
            dict with 'logits' (B, L) and 'attn_weights' (B, heads, N)
        """
        h = self.feature_extractor(bags)                       # (B, N, D)

        gated = torch.tanh(self.attention_V(h)) * torch.sigmoid(self.attention_U(h))
        scores = self.attention_w(gated)                       # (B, N, heads)

        if mask is not None:
            # mask==1 marks padding; exclude those positions from the softmax
            scores = scores.masked_fill(mask.bool().unsqueeze(-1), float('-inf'))

        attn = torch.softmax(scores, dim=1)                    # (B, N, heads)
        # A fully padded row would yield NaN; replace with zeros defensively.
        attn = torch.nan_to_num(attn, nan=0.0)

        if self.per_label_attention:
            # (B, N, L) x (B, N, D) -> (B, L, D)
            z = torch.einsum('bnl,bnd->bld', attn, h)
            logits = (z * self.cls_weight).sum(-1) + self.cls_bias   # (B, L)
        else:
            z = torch.einsum('bnk,bnd->bkd', attn, h).squeeze(1)     # (B, D)
            logits = self.classifier(z)                              # (B, L)

        return {'logits': logits, 'attn_weights': attn.transpose(1, 2)}
