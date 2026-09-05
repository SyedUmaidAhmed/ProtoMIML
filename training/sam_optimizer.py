"""Sharpness-Aware Minimization (SAM) optimizer wrapper.

SAM finds parameters that lie in neighborhoods having uniformly low loss,
leading to flatter minima and better generalization on small datasets.

Reference: Foret et al., "Sharpness-Aware Minimization for Efficiently
Improving Generalization", ICLR 2021.
"""

import torch


class SAM(torch.optim.Optimizer):
    """SAM optimizer wrapper around a base optimizer (e.g., AdamW).

    Two forward-backward passes per step:
    1. first_step(): compute gradient, perturb weights by rho in gradient direction
    2. second_step(): compute gradient at perturbed point, restore weights, apply update

    Usage:
        optimizer = SAM(model.parameters(), AdamW, lr=1e-3, rho=0.05)
        # In training loop:
        loss = compute_loss(model(x), y)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        compute_loss(model(x), y).backward()  # second forward-backward
        optimizer.second_step(zero_grad=True)
    """

    def __init__(self, params, base_optimizer_cls, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """Perturb weights in the gradient direction."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                # Store original parameters
                self.state[p]['old_p'] = p.data.clone()
                # Compute perturbation
                e_w = (torch.pow(p, 2) if group.get('adaptive', False) else 1.0) * p.grad * scale
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """Restore original weights and apply base optimizer update."""
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.data = self.state[p]['old_p']
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self):
        """Compute the L2 norm of all gradients."""
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group.get('adaptive', False) else 1.0) * p.grad)
                .norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ]),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
