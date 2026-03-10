"""Loss functions for imitation learning.

TODO (students implement all three):
    - bc_loss: MSE regression loss for behavior cloning.
    - gaussian_nll_loss: Negative log-likelihood for Gaussian BC.
    - diffusion_loss: MSE loss between predicted and true noise.
"""

import torch
import torch.nn as nn


def bc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the behavior cloning loss (MSE).

    Args:
        pred: predicted actions, shape (B, action_dim).
        target: expert actions, shape (B, action_dim).

    Returns:
        Scalar MSE loss (mean over batch and action dimensions).
    """
    return nn.MSELoss()(pred, target)


def gaussian_nll_loss(mean: torch.Tensor, log_var: torch.Tensor,
                      target: torch.Tensor) -> torch.Tensor:
    """Compute the Gaussian negative log-likelihood loss.

    For a Gaussian N(mean, exp(log_var)):
        NLL = 0.5 * (log_var + (target - mean)^2 / exp(log_var))
    averaged over the batch.

    Args:
        mean: predicted mean, shape (B, action_dim).
        log_var: predicted log-variance, shape (B, action_dim).
        target: expert actions, shape (B, action_dim).

    Returns:
        Scalar NLL loss (mean over batch and action dimensions).
    """
    return 0.5 * (log_var + (target - mean).pow(2) / log_var.exp()).mean()


def diffusion_loss(pred_noise: torch.Tensor,
                   target_noise: torch.Tensor) -> torch.Tensor:
    """Compute the diffusion denoising loss (MSE on noise prediction).

    Args:
        pred_noise: noise predicted by the denoising network, shape (B, action_dim).
        target_noise: actual noise that was added, shape (B, action_dim).

    Returns:
        Scalar MSE loss (mean over batch and action dimensions).
    """
    return nn.MSELoss()(pred_noise, target_noise)
