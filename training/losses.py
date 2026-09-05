"""Loss functions for ProtoMIML.

Includes:
  - Multi-View Consistency Loss: enforce prediction consistency across dropout views
"""

import torch
import torch.nn.functional as F


def consistency_loss(model, bags, mask=None, bag_labels=None):
    """Multi-view consistency training via dual dropout forward passes.

    Two forward passes through the model with dropout active produce two
    different predictions. We enforce agreement between them via symmetric BCE.

    consistency = 0.5 * (BCE(p1, p2.detach()) + BCE(p2, p1.detach()))

    Args:
        model: ProtoMIML model (must be in train mode for dropout)
        bags: (B, N, D) input bags
        mask: (B, N) padding mask
        bag_labels: (B, L) labels for prototype updates

    Returns:
        loss: scalar consistency loss
    """
    assert model.training, "Model must be in train mode for consistency loss"

    # View 1
    result1 = model(bags, mask=mask, bag_labels=bag_labels, update_prototypes=False)
    logits1 = result1['fused_logits']

    # View 2
    result2 = model(bags, mask=mask, bag_labels=bag_labels, update_prototypes=False)
    logits2 = result2['fused_logits']

    # Symmetric BCE consistency loss with clamping for numerical stability
    eps = 1e-6
    p1 = torch.sigmoid(logits1).clamp(eps, 1 - eps)
    p2 = torch.sigmoid(logits2).clamp(eps, 1 - eps)

    loss = 0.5 * (
        F.binary_cross_entropy(p1, p2.detach(), reduction='mean') +
        F.binary_cross_entropy(p2, p1.detach(), reduction='mean')
    )

    return loss
