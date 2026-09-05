"""ProtoMIML: Prototypical Multi-Instance Multi-Label Learning.

Main model assembling all components:
  InstanceEncoder -> SetAttentionEncoder (ISAB) -> CrossAttentionLabelPooling
  -> LabelGCN -> DualClassifier (parametric MLP + PrototypicalMemoryBank)

Architecture Flow:
  Input bags (B, N, D_in)
    -> Instance Encoder (MLP + LayerNorm + GELU + residual) -> (B, N, 64)
    -> Set Attention Encoder (ISAB with inducing points) -> (B, N, 64)
    -> Cross-Attention Label Pooling (label queries x instance KV) -> (B, L, 64)
    -> Label GCN Refinement (2-layer GCN over co-occurrence graph) -> (B, L, 64)
    -> Dual Classification:
        Path A: Per-label MLP -> parametric logits (B, L)
        Path B: Prototype distance -> non-parametric scores (B, L)
        Fusion: sigma(gate_l) x parametric + (1-sigma(gate_l)) x proto_score
"""

import torch
import torch.nn as nn

from models.instance_encoder import InstanceEncoder
from models.set_attention import SetAttentionEncoder
from models.cross_attention_pooling import CrossAttentionLabelPooling
from models.label_gnn import LabelGCN
from models.prototypical_memory import PrototypicalMemoryBank


class PerLabelClassifier(nn.Module):
    """Per-label MLP classifiers: Linear(D, D//2) -> GELU -> Dropout -> Linear(D//2, 1)."""

    def __init__(self, num_labels, embed_dim, dropout=0.15):
        super().__init__()
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, 1),
            )
            for _ in range(num_labels)
        ])

    def forward(self, label_repr):
        """
        Args:
            label_repr: (B, L, D)
        Returns:
            logits: (B, L)
        """
        logits = []
        for l, clf in enumerate(self.classifiers):
            logits.append(clf(label_repr[:, l, :]))  # (B, 1)
        return torch.cat(logits, dim=-1)  # (B, L)


class ProtoMIML(nn.Module):
    """Prototypical Multi-Instance Multi-Label Learning model.

    Args:
        input_dim: Raw feature dimension (15 for Scene, 243 for Reuters)
        num_labels: Number of labels (5 for both datasets)
        embed_dim: Embedding dimension (default 64)
        num_heads: Attention heads (default 4)
        ff_dim: FFN hidden dim (default 128)
        num_inducing: ISAB inducing points (4 for Scene, 6 for Reuters)
        dropout: Dropout rate (default 0.15)
        proto_temperature: Prototype scoring temperature (default 0.1)
        proto_ema_decay: EMA decay for prototypes (default 0.999)
        use_isab: Enable ISAB set attention (default True)
        use_gcn: Enable label GCN (default True)
        use_prototypes: Enable prototypical memory (default True)
    """

    def __init__(self, input_dim, num_labels, embed_dim=64, num_heads=4,
                 ff_dim=128, num_inducing=4, dropout=0.15,
                 proto_temperature=0.1, proto_ema_decay=0.999,
                 use_isab=True, use_gcn=True, use_prototypes=True,
                 embedding_layers=2, learnable_gate=True):
        super().__init__()
        self.num_labels = num_labels
        self.embed_dim = embed_dim
        self.use_isab = use_isab
        self.use_gcn = use_gcn
        self.use_prototypes = use_prototypes

        # Component 1: Instance Encoder
        self.instance_encoder = InstanceEncoder(
            input_dim, embed_dim, dropout=dropout, num_layers=embedding_layers
        )

        # Component 2: Set Attention Encoder (ISAB)
        if use_isab:
            self.set_attention = SetAttentionEncoder(
                embed_dim, num_heads=num_heads, ff_dim=ff_dim,
                num_inducing=num_inducing, num_layers=1, dropout=dropout
            )

        # Component 3: Cross-Attention Label Pooling
        self.label_pooling = CrossAttentionLabelPooling(
            num_labels, embed_dim, num_heads=num_heads,
            ff_dim=ff_dim, dropout=dropout
        )

        # Component 4: Label GCN
        if use_gcn:
            self.label_gcn = LabelGCN(
                embed_dim, hidden_dim=embed_dim, num_labels=num_labels,
                dropout=dropout
            )

        # Component 5: Per-label MLP classifiers (parametric path)
        self.classifier = PerLabelClassifier(num_labels, embed_dim, dropout)

        # Component 6: Prototypical Memory Bank (non-parametric path)
        if use_prototypes:
            self.proto_bank = PrototypicalMemoryBank(
                num_labels, embed_dim,
                temperature=proto_temperature, ema_decay=proto_ema_decay,
                learnable_gate=learnable_gate
            )

    def init_label_gcn(self, co_matrix):
        """Initialize label GCN adjacency from co-occurrence matrix."""
        if self.use_gcn:
            self.label_gcn.init_from_cooccurrence(co_matrix)

    def forward(self, bags, mask=None, bag_labels=None, update_prototypes=True):
        """
        Args:
            bags: (B, N, D_in) input instance features
            mask: (B, N) padding mask, 1=padded, 0=real
            bag_labels: (B, L) labels for prototype update (training only)
            update_prototypes: whether to update prototype bank

        Returns:
            dict with keys:
                'logits': (B, L) parametric logits
                'fused_logits': (B, L) fused parametric+prototype logits
                'coupled_logits': (B, L) alias for fused_logits (API compat)
                'proto_scores': (B, L) raw prototype scores
                'attn_weights': (B, L, N) cross-attention weights
                'label_repr': (B, L, D) label representations
        """
        # Convert mask: our convention is 1=padded=True for key_padding_mask
        key_padding_mask = mask.bool() if mask is not None else None

        # 1. Instance encoding
        instance_emb = self.instance_encoder(bags)  # (B, N, D)

        # 2. Set attention (ISAB)
        if self.use_isab:
            instance_emb = self.set_attention(instance_emb, key_padding_mask=key_padding_mask)

        # 3. Cross-attention label pooling
        label_repr, attn_weights = self.label_pooling(
            instance_emb, key_padding_mask=key_padding_mask
        )  # (B, L, D), (B, L, N)

        # 4. Label GCN refinement
        if self.use_gcn:
            label_repr = self.label_gcn(label_repr)

        # 5. Parametric classification
        logits = self.classifier(label_repr)  # (B, L)

        result = {
            'logits': logits,
            'attn_weights': attn_weights,
            'label_repr': label_repr,
        }

        # 6. Prototypical memory bank
        if self.use_prototypes:
            # Update prototypes during training
            if self.training and bag_labels is not None and update_prototypes:
                self.proto_bank.update_prototypes(label_repr.detach(), bag_labels)

            fused_logits, proto_scores = self.proto_bank.fuse(logits, label_repr)
            result['fused_logits'] = fused_logits
            result['coupled_logits'] = fused_logits  # API compatibility
            result['proto_scores'] = proto_scores
        else:
            result['fused_logits'] = logits
            result['coupled_logits'] = logits
            result['proto_scores'] = logits

        return result
