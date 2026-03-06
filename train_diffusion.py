"""Diffusion Policy (DDPM) for behavior cloning.

Uses the same approach as the Diffusion Policy paper (Chi et al.):
- DDPMScheduler from HuggingFace diffusers with squared cosine beta schedule
- Epsilon-prediction
- EMA on model weights
- Cosine LR schedule with warmup

Usage:
    python train_diffusion.py --data data/easy_expert.npz --output models/diffusion_easy.pt
    python train_diffusion.py --data data/hard_expert.npz --output models/diffusion_hard.pt
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding (same as diffusion policy paper)
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# ---------------------------------------------------------------------------
# Denoising network
# ---------------------------------------------------------------------------

class NoisePredictor(nn.Module):
    """MLP noise predictor with FiLM-style conditioning from diffusion step.

    Following the diffusion policy paper, the timestep embedding goes through
    a small MLP (sinusoidal -> linear -> Mish -> linear) and is then used
    to condition the main network.
    """

    def __init__(self, state_dim: int = 4, action_dim: int = 1,
                 time_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.time_dim = time_dim

        # Timestep encoder (matches paper: sinusoidal -> expand -> Mish -> compress)
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        input_dim = action_dim + state_dim + time_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.Mish(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, noisy_action, state, timestep):
        t_emb = self.time_encoder(timestep)
        x = torch.cat([noisy_action, state, t_emb], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(data_path: str, output_path: str, epochs: int = 200,
          batch_size: int = 256, lr: float = 1e-4,
          num_diffusion_iters: int = 100):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    data = np.load(data_path)
    states = torch.tensor(data["states"], dtype=torch.float32)
    actions = torch.tensor(data["actions"], dtype=torch.float32)

    dataset = TensorDataset(states, actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = NoisePredictor().to(device)

    # DDPMScheduler with squared cosine schedule (same as paper)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon',
    )

    # EMA (same as paper)
    ema = EMAModel(parameters=model.parameters(), power=0.75)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        n = 0
        for s_batch, a_batch in loader:
            s_batch = s_batch.to(device)
            a_batch = a_batch.to(device)
            bsz = s_batch.size(0)

            # Sample noise
            noise = torch.randn_like(a_batch)

            # Sample random timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (bsz,), device=device
            ).long()

            # Add noise (forward diffusion)
            noisy_actions = noise_scheduler.add_noise(a_batch, noise, timesteps)

            # Predict noise
            pred_noise = model(noisy_actions, s_batch, timesteps)
            loss = loss_fn(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update EMA
            ema.step(model.parameters())

            total_loss += loss.item() * bsz
            n += bsz

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}  loss={total_loss/n:.6f}")

    # Copy EMA weights to model for inference
    ema.copy_to(model.parameters())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved diffusion policy to {output_path}")


@torch.no_grad()
def sample(model, noise_scheduler, state, num_diffusion_iters=100):
    """Sample actions from the diffusion model (same as paper inference loop)."""
    batch = state.size(0)
    device = state.device
    noisy_action = torch.randn(batch, 1, device=device)

    noise_scheduler.set_timesteps(num_diffusion_iters)
    for k in noise_scheduler.timesteps:
        noise_pred = model(
            noisy_action, state,
            torch.full((batch,), k, device=device, dtype=torch.long)
        )
        noisy_action = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=noisy_action
        ).prev_sample

    return noisy_action.clamp(0.0, 1.0)


class DDPMSchedule:
    """Lightweight self-contained DDPM schedule for quick demos (arbitrary T and beta range)."""

    def __init__(self, T=100, beta_start=0.0001, beta_end=0.02, device='cpu',
                 action_dim=1):
        self.T = T
        self.device = device
        self.action_dim = action_dim
        betas = torch.linspace(beta_start, beta_end, T, device=device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bar = alpha_bar
        self.sqrt_alpha_bar = alpha_bar.sqrt()
        self.sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()

    def q_sample(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(-1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        return sqrt_ab * x0 + sqrt_omab * noise, noise

    @torch.no_grad()
    def sample(self, model, state):
        batch = state.size(0)
        device = state.device
        x = torch.randn(batch, self.action_dim, device=device)
        for t_val in reversed(range(self.T)):
            t = torch.full((batch,), t_val, device=device, dtype=torch.long)
            pred_noise = model(x, state, t)
            alpha = self.alphas[t_val]
            alpha_bar = self.alpha_bar[t_val]
            beta = self.betas[t_val]
            x = (1.0 / alpha.sqrt()) * (x - (beta / (1.0 - alpha_bar).sqrt()) * pred_noise)
            if t_val > 0:
                x = x + beta.sqrt() * torch.randn_like(x)
        return x.clamp(0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="models/diffusion.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--T", type=int, default=100)
    args = parser.parse_args()
    train(args.data, args.output, args.epochs, args.batch_size, args.lr, args.T)


if __name__ == "__main__":
    main()
