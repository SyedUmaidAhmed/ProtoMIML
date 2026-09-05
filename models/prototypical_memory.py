"""Prototypical Memory Bank with EMA-updated per-label prototypes.

Bridges parametric (neural) and non-parametric (kNN/SVM) methods by maintaining
learnable prototypes and fusing prototype-based scores with parametric logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypicalMemoryBank(nn.Module):
    """EMA-updated per-label prototypes with differentiable scoring and gated fusion.

    For each label, maintains positive and negative prototype vectors.
    During training, prototypes are updated via exponential moving average.
    Scoring: score = (neg_dist - pos_dist) / temperature.
    Fusion: final = sigmoid(gate) * parametric + (1 - sigmoid(gate)) * proto_score.
    """

    def __init__(self, num_labels, embed_dim, temperature=0.1, ema_decay=0.999,
                 learnable_gate=True):
        super().__init__()
        self.num_labels = num_labels
        self.learnable_gate = learnable_gate
        self.embed_dim = embed_dim
        self.temperature = temperature
        self.ema_decay = ema_decay

        # Prototype projection: project label representations into prototype space
        self.proto_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        # Prototypes as buffers (not trainable, updated via EMA)
        self.register_buffer('pos_prototypes', torch.randn(num_labels, embed_dim) * 0.01)
        self.register_buffer('neg_prototypes', torch.randn(num_labels, embed_dim) * 0.01)
        self.register_buffer('initialized', torch.zeros(num_labels, dtype=torch.bool))

        # Learned per-label fusion gate.  With learnable_gate=False the gate is
        # frozen at its zero initialisation, i.e. sigmoid(0) = 0.5 exactly, giving
        # a fixed equal-weight average of the parametric and prototype pathways.
        self.gate = nn.Parameter(torch.zeros(num_labels), requires_grad=learnable_gate)

    @torch.no_grad()
    def update_prototypes(self, label_repr, labels):
        """Update prototypes via EMA using batch statistics.

        Args:
            label_repr: (B, L, D) per-label representations (already projected)
            labels: (B, L) binary labels
        """
        for l in range(self.num_labels):
            pos_mask = labels[:, l] > 0.5
            neg_mask = labels[:, l] < 0.5

            if pos_mask.any():
                pos_mean = label_repr[pos_mask, l, :].mean(dim=0)
                if self.initialized[l]:
                    self.pos_prototypes[l] = (
                        self.ema_decay * self.pos_prototypes[l] +
                        (1 - self.ema_decay) * pos_mean
                    )
                else:
                    self.pos_prototypes[l] = pos_mean
                    self.initialized[l] = True

            if neg_mask.any():
                neg_mean = label_repr[neg_mask, l, :].mean(dim=0)
                if self.initialized[l]:
                    self.neg_prototypes[l] = (
                        self.ema_decay * self.neg_prototypes[l] +
                        (1 - self.ema_decay) * neg_mean
                    )
                else:
                    self.neg_prototypes[l] = neg_mean

    def compute_proto_scores(self, label_repr):
        """Compute prototype-based scores using normalized Euclidean distance.

        Args:
            label_repr: (B, L, D) per-label representations

        Returns:
            proto_scores: (B, L) scores from prototype distances, clamped to [-5, 5]
        """
        projected = self.proto_proj(label_repr)  # (B, L, D)

        # L2-normalize to make distances comparable across dimensions
        proj_norm = F.normalize(projected, dim=-1)
        pos_norm = F.normalize(self.pos_prototypes.unsqueeze(0), dim=-1)
        neg_norm = F.normalize(self.neg_prototypes.unsqueeze(0), dim=-1)

        # Euclidean distance on unit sphere: range [0, 2]
        pos_dist = torch.sum((proj_norm - pos_norm) ** 2, dim=-1)  # (B, L)
        neg_dist = torch.sum((proj_norm - neg_norm) ** 2, dim=-1)  # (B, L)

        # Score: closer to positive → higher score
        proto_scores = (neg_dist - pos_dist) / self.temperature
        # Clamp to prevent extreme values
        proto_scores = proto_scores.clamp(-5.0, 5.0)
        return proto_scores

    def fuse(self, parametric_logits, label_repr):
        """Fuse parametric logits with prototype scores.

        Args:
            parametric_logits: (B, L) from per-label MLP classifiers
            label_repr: (B, L, D) per-label representations

        Returns:
            fused_logits: (B, L) gated fusion of parametric and prototype scores
            proto_scores: (B, L) raw prototype scores
        """
        proto_scores = self.compute_proto_scores(label_repr)
        gate = torch.sigmoid(self.gate)  # (L,)
        fused_logits = gate * parametric_logits + (1 - gate) * proto_scores
        return fused_logits, proto_scores
