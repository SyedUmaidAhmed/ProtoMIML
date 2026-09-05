"""ProtoMIML Two-Phase Trainer with SAM optimizer.

Phase 1 (Foundation): SAM(AdamW) + CosineAnnealingWarmRestarts
  - BCE on parametric logits only
  - Prototype bank updates in background (EMA, no loss contribution)

Phase 2 (Full Training): SAM(AdamW) + CosineAnnealingLR + SWA
  - BCE on fused logits + consistency loss
  - Prototype fusion active
  - Early stopping on val AP
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
import numpy as np

from training.sam_optimizer import SAM
from training.losses import consistency_loss
from metrics.miml_metrics import AllFive

# Import augmentation from sibling project
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'MIML_CLAM_Release'))
try:
    from miml_clam.training.augmentation import bag_mixup
except ImportError:
    def bag_mixup(bags, labels, mask, alpha=0.2, fixed_size=True):
        """Fallback: inline bag mixup."""
        B = bags.size(0)
        if B < 2:
            return bags, labels, mask
        lam = max(np.random.beta(alpha, alpha), 1 - np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
        indices = torch.randperm(B, device=bags.device)
        if fixed_size:
            return lam * bags + (1 - lam) * bags[indices], lam * labels + (1 - lam) * labels[indices], mask
        else:
            mixed_labels = lam * labels + (1 - lam) * labels[indices]
            mixed_bags_list, mixed_mask_list = [], []
            for i in range(B):
                j = indices[i].item()
                vi = (~mask[i].bool()).sum().item()
                vj = (~mask[j].bool()).sum().item()
                combined = torch.cat([bags[i, :vi], bags[j, :vj]], dim=0)
                mixed_bags_list.append(combined)
                mixed_mask_list.append(torch.zeros(combined.shape[0], device=bags.device))
            max_len = max(b.shape[0] for b in mixed_bags_list)
            D = bags.shape[2]
            mixed_bags = torch.zeros(B, max_len, D, device=bags.device)
            mixed_mask = torch.ones(B, max_len, device=bags.device)
            for i, (b, m) in enumerate(zip(mixed_bags_list, mixed_mask_list)):
                n = b.shape[0]
                mixed_bags[i, :n] = b
                mixed_mask[i, :n] = 0
            return mixed_bags, mixed_labels, mixed_mask


class ProtoMIMLTrainer:
    """Two-phase trainer for ProtoMIML with SAM optimizer."""

    def __init__(self, model, config, device='cpu'):
        self.model = model
        self.config = config
        self.device = device
        self.label_smooth = config.get('label_smoothing', 0.0)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def _smooth_labels(self, labels):
        if self.label_smooth > 0:
            return labels * (1 - self.label_smooth) + (1 - labels) * self.label_smooth
        return labels

    def _maybe_mixup(self, bags, labels, mask):
        if self.config.get('use_mixup', False):
            fixed = self.config.get('fixed_bag_size', True)
            alpha = self.config.get('mixup_alpha', 0.2)
            return bag_mixup(bags, labels, mask, alpha=alpha, fixed_size=fixed)
        return bags, labels, mask

    def _create_sam_optimizer(self, lr, rho):
        """Create SAM optimizer wrapping AdamW."""
        return SAM(
            self.model.parameters(),
            optim.AdamW,
            rho=rho,
            lr=lr,
            weight_decay=self.config.get('weight_decay', 5e-4),
        )

    def _sam_step(self, optimizer, loss_fn, bags, labels, mask, smooth_labels, phase='phase1'):
        """Perform a single SAM step (two forward-backward passes).

        Args:
            optimizer: SAM optimizer
            loss_fn: callable returning loss given model output
            bags, labels, mask, smooth_labels: batch data
            phase: 'phase1' or 'phase2'
        Returns:
            loss value (from first pass)
        """
        # First forward-backward pass
        result = self.model(bags, mask=mask, bag_labels=labels)
        loss = loss_fn(result, smooth_labels, bags, mask, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.get('gradient_clip', 1.0)
        )
        optimizer.first_step(zero_grad=True)

        # Second forward-backward pass (at perturbed point)
        result2 = self.model(bags, mask=mask, bag_labels=labels)
        loss2 = loss_fn(result2, smooth_labels, bags, mask, labels)
        loss2.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.get('gradient_clip', 1.0)
        )
        optimizer.second_step(zero_grad=True)

        return loss.item()

    def train_phase1(self, train_loader, val_loader, num_epochs=80, lr=1e-3):
        """Phase 1: Foundation training on parametric logits with SAM."""
        self.model.train()
        rho = self.config.get('phase1_sam_rho', 0.05)
        optimizer = self._create_sam_optimizer(lr, rho)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer.base_optimizer, T_0=max(num_epochs // 3, 10), T_mult=1
        )

        best_ap = 0.0
        best_state = None

        def phase1_loss(result, smooth_labels, bags, mask, labels):
            return self.bce_loss(result['logits'], smooth_labels)

        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for bags, labels, mask in train_loader:
                bags = bags.to(self.device)
                labels = labels.to(self.device)
                mask = mask.to(self.device)

                bags, labels_mix, mask = self._maybe_mixup(bags, labels, mask)
                smooth_labels = self._smooth_labels(labels_mix)

                loss_val = self._sam_step(
                    optimizer, phase1_loss, bags, labels_mix, mask, smooth_labels, 'phase1'
                )
                scheduler.step(epoch + num_batches / max(len(train_loader), 1))

                epoch_loss += loss_val
                num_batches += 1

            val_metrics = self.evaluate(val_loader, use_raw_logits=True)
            val_ap = val_metrics['AveragePrecision']
            if val_ap > best_ap:
                best_ap = val_ap
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if (epoch + 1) % 10 == 0:
                print(f"  Phase1 Epoch {epoch+1}/{num_epochs} | "
                      f"Loss: {epoch_loss/max(num_batches,1):.4f} | "
                      f"Val AP: {val_ap:.4f} | Best: {best_ap:.4f}", flush=True)

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        return best_ap

    def train_phase2(self, train_loader, val_loader, max_epochs=120, lr=3e-4,
                     patience=25):
        """Phase 2: Full training with fused logits, consistency, SWA."""
        self.model.train()
        rho = self.config.get('phase2_sam_rho', 0.03)
        optimizer = self._create_sam_optimizer(lr, rho)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer.base_optimizer, T_max=max_epochs
        )

        # SWA
        swa_model = AveragedModel(self.model)
        swa_start = self.config.get('swa_start', 30)
        swa_scheduler = SWALR(optimizer.base_optimizer, swa_lr=lr * 0.5, anneal_epochs=5)
        swa_active = False

        cons_weight = self.config.get('consistency_weight', 0.3)
        proto_warmup = 30  # epochs of pure parametric before any fused
        fused_ramp = 40  # epochs to ramp fused_ratio from 0 to 1
        cons_start = proto_warmup + fused_ramp + 10  # consistency after stabilization
        best_score = float('-inf')
        patience_counter = 0
        best_state = None

        for epoch in range(max_epochs):
            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            # Gradually transition from parametric to fused logits
            fused_ratio = min(1.0, max(0.0, (epoch - proto_warmup) / fused_ramp))

            def phase2_loss_fn(result, smooth_labels, bags, mask, labels):
                if fused_ratio <= 0:
                    return self.bce_loss(result['logits'], smooth_labels)
                elif fused_ratio < 1.0:
                    param_loss = self.bce_loss(result['logits'], smooth_labels)
                    fused_loss = self.bce_loss(result['fused_logits'], smooth_labels)
                    return (1 - fused_ratio) * param_loss + fused_ratio * fused_loss
                return self.bce_loss(result['fused_logits'], smooth_labels)

            for bags, labels, mask in train_loader:
                bags = bags.to(self.device)
                labels = labels.to(self.device)
                mask = mask.to(self.device)

                # Mixup
                mixed_bags, mixed_labels, mixed_mask = self._maybe_mixup(bags, labels, mask)
                smooth_labels = self._smooth_labels(mixed_labels)

                # SAM step for main BCE loss
                loss_val = self._sam_step(
                    optimizer, phase2_loss_fn, mixed_bags, mixed_labels,
                    mixed_mask, smooth_labels, 'phase2'
                )

                # Consistency loss — separate lightweight step (no SAM)
                if cons_weight > 0 and epoch >= cons_start:
                    self.model.train()
                    cons = consistency_loss(
                        self.model, bags, mask=mask, bag_labels=labels
                    )
                    scaled_cons = cons_weight * cons
                    scaled_cons.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.get('gradient_clip', 1.0))
                    optimizer.base_optimizer.step()
                    optimizer.zero_grad()
                    loss_val += scaled_cons.item()

                epoch_loss += loss_val
                num_batches += 1

            # SWA
            if epoch >= swa_start:
                swa_model.update_parameters(self.model)
                swa_scheduler.step()
                swa_active = True
            else:
                scheduler.step()

            # Validation
            eval_model = swa_model.module if swa_active else self.model
            val_metrics = self._evaluate_model(eval_model, val_loader)
            val_ap = val_metrics['AveragePrecision']

            if (epoch + 1) % 5 == 0:
                swa_str = " [SWA]" if swa_active else ""
                print(f"  Phase2 Epoch {epoch+1}/{max_epochs} | "
                      f"Loss: {epoch_loss/max(num_batches,1):.4f} | "
                      f"Val AP: {val_ap:.4f}{swa_str}", flush=True)

            if val_ap > best_score:
                best_score = val_ap
                patience_counter = 0
                state_dict = swa_model.module.state_dict() if swa_active else self.model.state_dict()
                best_state = {k: v.cpu().clone() for k, v in state_dict.items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stop at epoch {epoch+1} (best AP: {best_score:.4f})", flush=True)
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(self.device)
        return best_score

    def _evaluate_model(self, model, data_loader, use_raw_logits=False):
        """Evaluate a model on data loader."""
        model.eval()
        all_labels, all_raw, all_fused = [], [], []

        with torch.no_grad():
            for bags, labels, mask in data_loader:
                bags, mask = bags.to(self.device), mask.to(self.device)
                result = model(bags, mask=mask)
                all_raw.append(torch.sigmoid(result['logits']).cpu().numpy())
                all_fused.append(torch.sigmoid(result['fused_logits']).cpu().numpy())
                all_labels.append(labels.numpy())

        all_labels = np.concatenate(all_labels)
        all_raw = np.concatenate(all_raw)
        all_fused = np.concatenate(all_fused)

        metrics_raw = AllFive(all_labels, all_raw)
        metrics_fused = AllFive(all_labels, all_fused)

        if use_raw_logits:
            return metrics_raw
        return metrics_fused if metrics_fused['AveragePrecision'] >= metrics_raw['AveragePrecision'] else metrics_raw

    @torch.no_grad()
    def evaluate(self, data_loader, use_raw_logits=False):
        return self._evaluate_model(self.model, data_loader, use_raw_logits)

    @torch.no_grad()
    def get_predictions(self, data_loader):
        """Get raw predictions and labels for threshold optimization."""
        self.model.eval()
        all_labels, all_raw, all_fused = [], [], []

        for bags, labels, mask in data_loader:
            bags, mask = bags.to(self.device), mask.to(self.device)
            result = self.model(bags, mask=mask)
            all_raw.append(torch.sigmoid(result['logits']).cpu().numpy())
            all_fused.append(torch.sigmoid(result['fused_logits']).cpu().numpy())
            all_labels.append(labels.numpy())

        return (np.concatenate(all_labels),
                np.concatenate(all_raw),
                np.concatenate(all_fused))
