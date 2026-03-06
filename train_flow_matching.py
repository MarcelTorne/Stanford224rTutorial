"""Flow Matching policy for behavior cloning.

Conditional flow matching: learn a velocity field that transports noise to data.
Much simpler than DDPM — no noise schedule, just linear interpolation.

Reference: Lipman et al., "Flow Matching for Generative Modeling" (2023)

Usage:
    python train_flow_matching.py --data data/hard_expert.npz --output models/flow_hard.pt
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
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
# Velocity network
# ---------------------------------------------------------------------------

class VelocityNet(nn.Module):
    """Predicts the velocity field v(x_t, state, t) for flow matching."""

    def __init__(self, state_dim: int = 4, action_dim: int = 1,
                 time_dim: int = 64, hidden: int = 256):
        super().__init__()
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

    def forward(self, x_t, state, t):
        t_emb = self.time_encoder(t)
        inp = torch.cat([x_t, state, t_emb], dim=-1)
        return self.net(inp)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(data_path: str, output_path: str, epochs: int = 200,
          batch_size: int = 256, lr: float = 1e-4):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    data = np.load(data_path)
    states = torch.tensor(data["states"], dtype=torch.float32)
    actions = torch.tensor(data["actions"], dtype=torch.float32)

    dataset = TensorDataset(states, actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = VelocityNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)

    for epoch in range(epochs):
        total_loss = 0.0
        n = 0
        for s_batch, a_batch in loader:
            s_batch = s_batch.to(device)
            a_batch = a_batch.to(device)  # x_1 (data)
            bsz = s_batch.size(0)

            # Sample noise x_0 ~ N(0, 1)
            x_0 = torch.randn_like(a_batch)

            # Sample time t ~ U(0, 1)
            t = torch.rand(bsz, device=device)

            # Interpolate: x_t = (1 - t) * x_0 + t * x_1
            t_expand = t.unsqueeze(-1)
            x_t = (1 - t_expand) * x_0 + t_expand * a_batch

            # Target velocity: v = x_1 - x_0
            target_v = a_batch - x_0

            # Predict velocity
            pred_v = model(x_t, s_batch, t)
            loss = nn.MSELoss()(pred_v, target_v)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * bsz
            n += bsz

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}  loss={total_loss/n:.6f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved flow matching policy to {output_path}")


@torch.no_grad()
def sample(model, state, num_steps: int = 20):
    """Generate actions by integrating the velocity field (Euler method)."""
    batch = state.size(0)
    device = state.device

    # Start from noise
    x = torch.randn(batch, 1, device=device)
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.full((batch,), i * dt, device=device)
        v = model(x, state, t)
        x = x + v * dt

    return x.clamp(0.0, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="models/flow.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    train(args.data, args.output, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
