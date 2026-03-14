"""Loss functions for imitation learning.

Each loss function takes the model, a batch of states, and a batch of expert
actions, and returns a scalar loss.  This keeps the training loop generic.

TODO (students implement all four):
    - bc_loss: MSE regression loss for behavior cloning.
    - gaussian_nll_loss: Negative log-likelihood for Gaussian BC.
    - diffusion_loss: MSE loss between predicted and true noise.
    - flow_matching_loss: MSE loss between predicted and target velocity.
"""

import torch
import torch.nn as nn


def bc_loss(policy, s_batch: torch.Tensor,
            a_batch: torch.Tensor) -> torch.Tensor:
    """Compute the behavior cloning loss (MSE).

    Args:
        policy: BCPolicy network.
        s_batch: states, shape (B, state_dim).
        a_batch: expert actions, shape (B, action_dim).

    Returns:
        Scalar MSE loss (mean over batch and action dimensions).
    """
    pred = policy(s_batch)
    return nn.MSELoss()(pred, a_batch)


def gaussian_nll_loss(policy, s_batch: torch.Tensor,
                      a_batch: torch.Tensor) -> torch.Tensor:
    """Compute the Gaussian negative log-likelihood loss.

    For a Gaussian N(mean, exp(log_var)):
        NLL = 0.5 * (log_var + (target - mean)^2 / exp(log_var))
    averaged over the batch.

    Args:
        policy: GaussianBCPolicy network (returns mean, log_var).
        s_batch: states, shape (B, state_dim).
        a_batch: expert actions, shape (B, action_dim).

    Returns:
        Scalar NLL loss (mean over batch and action dimensions).
    """
    mean, log_var = policy(s_batch)
    return 0.5 * (log_var + (a_batch - mean).pow(2) / log_var.exp()).mean()


def diffusion_loss(policy, s_batch: torch.Tensor,
                   a_batch: torch.Tensor) -> torch.Tensor:
    """Compute the diffusion denoising loss (MSE on noise prediction).

    The policy (DiffusionPolicy) carries its own schedule and T.

    Args:
        policy: DiffusionPolicy (model + schedule + T).
        s_batch: states, shape (B, state_dim).
        a_batch: expert actions, shape (B, action_dim).

    Returns:
        Scalar MSE loss (mean over batch and action dimensions).
    """
    bsz = s_batch.size(0)
    t = torch.randint(0, policy.T, (bsz,), device=s_batch.device)
    noisy_a, noise = policy.schedule.q_sample(a_batch, t)
    pred_noise = policy(noisy_a, s_batch, t)
    return nn.MSELoss()(pred_noise, noise)


def flow_matching_loss(policy, s_batch: torch.Tensor,
                       a_batch: torch.Tensor) -> torch.Tensor:
    """Compute the flow matching loss (MSE on velocity prediction).

    The policy (FlowMatchingPolicy) carries its own schedule.

    Args:
        policy: FlowMatchingPolicy (model + schedule).
        s_batch: states, shape (B, state_dim).
        a_batch: expert actions, shape (B, action_dim).

    Returns:
        Scalar MSE loss (mean over batch and action dimensions).
    """
    bsz = s_batch.size(0)
    t = torch.rand(bsz, device=s_batch.device)
    x_t, velocity = policy.schedule.interpolate(a_batch, t)
    pred_v = policy(x_t, s_batch, t)
    return nn.MSELoss()(pred_v, velocity)
